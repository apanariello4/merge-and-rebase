# Reproducing the Proposal-1 testing method on the LLM path

Audience: an agent who has to evaluate ARIADNE Proposal 1 (target residual
completion) on the **decoder/LLM** side of `merge-and-rebase`, matching the
methodology already established on the vision side.

This is not a description of the code. It is the *protocol*: what to run, what to
check before believing a number, and which parts of the vision method cannot be
carried over. Read §1 before planning anything — it will change your design.

---

## 1. Read this first: what ports now, and what still does not

**Superseded as of 2026-09-21.** Both Proposal-1 arms are now wired on the LLM
path. What follows replaces the original §1, which said the transport-free arm
did not exist there; it did not, and now it does.

| | vision | LLM |
|---|---|---|
| imports `complete_residuals` | yes | yes (`llm_rebase.py:76`) |
| imports `complete_residuals_direct` | yes (`vision_rebase.py:80`) | yes (`llm_rebase.py:79`) |
| branches on `config.mode` | yes (`vision_rebase.py:1289-1312`) | yes (`llm_rebase.py`, `_maybe_complete_target_residual_task_vector`) |

So `"mode": "direct_target"` now runs the transport-free solve it names. Both
arms are honestly testable:

- **`transport_residual`** — complete the residual an already-transported task
  vector left behind. Unchanged.
- **`direct_target`** — transport-free. `method.prepare` and `method.transport`
  are both skipped for that task, the transported delta is empty by
  construction, and the fitted correction is the entire task vector, scaled
  against an explicit zero baseline so `strength: 0` is an exact
  native-target-base control.

`target_trajectory: "interpolate"` and `components: ["attn.out_proj",
"mlp.c_proj"]` are reachable in `direct_target` mode, on decoders as well as on
ViTs. The decoder `o_proj` hook is no longer untested: a forward hook on a real
`nn.Linear` `o_proj` fires once per batch and its captured rows reproduce the
attention output, pinned in `tests/test_llm_direct_target_p1.py`.

**Three decisions baked into the LLM direct arm.** They change what the arm
measures, so read them before quoting a number:

1. **Passthrough keys are dropped.** The non-transportable source keys
   (embeddings, `lm_head`, norms) are folded into the output on the transport
   arm; the direct arm discards them, because folding raw source parameters into
   the target would make the arm not transport-free and would stop `strength: 0`
   being an exact control. The summary records this as
   `residual_completion.passthrough_policy` with a dropped-key count.
2. **Bias materialization covers every fitted component.** Qwen's `o_proj` is
   bias-free like `down_proj`, so the two-component exact arm materializes six
   zero biases per block-pair, not three. `materialize_missing_projection_biases`
   now takes `components`; its default is the historical MLP-only set.
3. **The block-extension pre-step still runs.** It is what produces the realized
   layout and the native reference banks; only the *transport fit* is skipped.
   That is also where the arm's cost saving comes from.

**Two silent no-ops — one of them now fails loudly instead.** P1 needs both:

1. the block-extension pre-step — `method ∈ {theseus, theseus_gqa, bico}`,
   `block_extension_enabled: true`, **and a source/target depth mismatch**;
2. execution reaching the transported-body branch.

(1) is now a hard error: an enabled `target_residual_completion` with no
pre-step raises before any model is resized, naming which condition failed.
Still confirm P1 actually fired (§5) — the gate costs nothing and (2) is not
guarded.

## 2. Prerequisites

**The venv is Python 3.14 — always use `.venv/bin/python`.** System `python3` is
3.10 and lacks torch.

**Install the harness extras. They are missing and IFEval will not run without
them.** Verified in this checkout:

```
lm_eval        OK
nltk           OK
langdetect     MISSING   <- IFEval instruction checkers import this
immutabledict  MISSING   <- likewise
math_verify    MISSING
```

```bash
uv sync --extra harness            # or: .venv/bin/pip install -e '.[harness]'
.venv/bin/python -m nltk.downloader punkt_tab
```

**Models are already cached; datasets are not.** `HF_HOME=/home/aba/.cache/huggingface`
(note: not the invoking user's home). Qwen2.5-0.5B, 0.5B-Instruct and 1.5B are all
present, so the reference campaign needs **no model download**. `google/IFEval` is
**not** cached — first run needs network.

**Hardware:** 2 × RTX 4080 SUPER, 16 GB VRAM each. The only historical LLM failure
logged on this box is a CUDA OOM on exactly this card. Budget accordingly (§7).

**Every shipped LLM config writes to `/leonardo_scratch/...`, which does not exist
here.** Override `logging.local_log_dir` or pass `--local-log-dir`, or the run dies
on startup.

---

## 3. The reference setup

Entry point — there is **no LLM launcher script** in this repo (the `scripts/*`
runners are vision-only), so invoke directly:

```bash
.venv/bin/python -m merge_and_rebase.eval.llm_rebase --config <config.json>
```

Reference pair (a genuine depth-up **and** width-up rebase):

| | model | layers | hidden |
|---|---|---|---|
| source base | `Qwen/Qwen2.5-0.5B` | 24 | 896 |
| source tuned | `Qwen/Qwen2.5-0.5B-Instruct` | 24 | 896 |
| target base | `Qwen/Qwen2.5-1.5B` | 28 | 1536 |

Start from these, which are the only runnable P1 configs shipped:

- `configs/llm_rebase_qwen0.5b_to_1.5b_ifeval_theseus_p1exact_ridge100_s1.json`
  — exact affine form, `missing_bias: "materialize"`
- `configs/llm_rebase_qwen0.5b_to_1.5b_ifeval_theseus_p1reduced_ridge100_s1.json`
  — `exact_form: false`, `missing_bias: "skip"` (the documented ablation)
- `configs/llm_rebase_qwen0.5b_to_1.5b_ifeval_theseus_skipcorr_true.json`
  — the no-P1 baseline

> **Do not use the six `..._p1_ridge100_s*.json` configs.** They omit
> `missing_bias` and `exact_form`, so they take the defaults (`exact_form=true`,
> `missing_bias="error"`). Decoder MLPs are bias-free, so the exact form has
> nowhere to put its intercept and the run raises at strength > 0. At
> `strength: 0` it returns early and never notices — which is worse, because it
> looks like it worked.

---

## 4. Metric, and what it is not

There is **no `avg_rebased`** on this path. Scoring goes through
lm-evaluation-harness (`eval/lm_harness_runner.py`):

- per-task metrics are flattened to `{task}_{metric}` — for IFEval:
  `ifeval_prompt_level_strict_acc`, `ifeval_inst_level_strict_acc`, and the two
  `loose` variants;
- the selection score is `score_by_task()` — the mean over tasks of each task's
  mean metric. **This averages strict and loose together**, so quote the specific
  metric you care about, not just the scalar the search optimised.

Alpha search is the same machinery as vision, driven by
`hyperparam_search.alpha.values` (the P1 configs use
`[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]`). Every alpha re-runs the whole
harness, so cost scales with the number of alphas.

Calibration/eval disjointness is handled for you: with `calibration_split: "val"`
the task's docs are shuffled by `seed`, a prefix is used for calibration, and the
held-out indices become `harness_samples`. For the reference config that is
100 batches × 4 = 400 of IFEval's 541 docs for calibration, leaving **~141 eval
docs**. Note how small that is — see §6.

---

## 5. Per-run acceptance gate

Run these before believing any number. This is the LLM translation of the vision
gate, and it exists because the failure modes here are silent.

```bash
.venv/bin/python - <<'EOF'
import json, sys
d = json.load(open(sys.argv[1] if len(sys.argv)>1 else "SUMMARY.json"))
rl = d["run_logging"]
print("status          :", rl["status"], "| error:", rl["error"])
rc = (d.get("task_vectors") or {}).get("residual_completion") or {}
print("P1 enabled      :", rc.get("enabled"))
print("target_scope    :", rc.get("target_scope"), "| strength:", rc.get("strength"))
print("materialized    :", rc.get("materialized_bias_keys"))
diag = rc.get("diagnostics") or {}
rows = [r for v in diag.values() for r in (v or [])]
print("blocks fitted   :", len(rows), "  <- MUST be > 0, or P1 silently no-oped")
if rows:
    print("first block rel_residual_before:", round(rows[0]["relative_residual_before"], 6))
    bad = [r for r in rows if r["residual_norm_after"] >= r["residual_norm_before"]]
    print("blocks not improved:", len(bad))
md = d.get("merged_delta") or {}
print("merged delta    : keys", md.get("key_count"), "nonzero", md.get("nonzero_key_count"),
      "rel_norm", md.get("merged_delta_rel_norm"))
print("best_alpha      :", d.get("best_alpha"))
EOF
```

Required:

1. `run_logging.status == "success"`.
2. **`blocks fitted > 0`** — the single most important check. Zero means P1 never
   ran (depth match, or the pre-step branch was skipped) and the run measured the
   baseline while claiming to be a P1 arm.
3. First fitted block `relative_residual_before == 1.0` (±1e-4) — proves the
   temporary model started from the untouched base.
4. `residual_norm_after < residual_norm_before` for every block.
5. `merged_delta.nonzero_key_count > 0` — a zero delta raises, but check anyway.
6. For an `exact_form: true` run, `materialized_bias_keys` is non-empty (the
   `down_proj.bias` keys that had to be created).

---

## 6. The methodology: measure the noise floor *first*

This is the part most worth carrying over, because on the vision side it
invalidated a conclusion that looked solid.

On vision, three runs differing **only in the calibration seed** gave 84.20 /
86.62 / 83.67 (2-task average): **σ_emp = 1.28 pp**, against a binomial
prediction of 0.34 pp. Calibration resampling dominated test sampling by ~4×, and
an entire ridge sweep spanning 1.89 pp turned out to be **smaller than 1.5 σ**.

The LLM setup is *more* exposed to this, not less: the eval hold-out is **~141
IFEval docs** (vs 2430 EuroSAT images), and the calibration draw is 400 of 541
docs from the same pool. Binomial noise alone on 141 docs at p≈0.4 is ~4 pp per
run, ~6 pp on a difference.

**Therefore, before sweeping anything:**

1. Run the reference config at **≥3 seeds** (`seed: 0, 17, 89`), changing nothing
   else. Record σ_emp.
2. Pre-commit to ignoring any later difference smaller than ~2 σ_emp.
3. If σ_emp is large relative to the effects you care about, fix it before
   sweeping — raise `harness_limit`/hold-out size, add tasks beyond IFEval, or
   average over seeds. Do not proceed to a fine grid on a noisy metric.

Everything else follows the vision discipline:

- **One factor at a time.** A new variant gets its own campaign namespace, never
  an edited cell in an existing one.
- **Write-once results.** Never overwrite a completed summary; make a new
  namespace instead.
- **Track diagnostics next to the metric**, not just accuracy: per block,
  `relative_residual_after`, `unreachable_residual_norm / residual_norm_before`,
  and the fraction of blocks whose achieved residual *exceeds* the unreachable
  floor. On vision these revealed that accuracy rose as the fit got *worse* at its
  own objective — the kind of finding a scalar accuracy column hides.

---

## 7. Axes worth sweeping (and ones that are inert)

| axis | config path | notes |
|---|---|---|
| `ridge_relative` (ρ) | `...target_residual_completion.ridge_relative` | the lead axis on vision; monotone and unbracketed there. Sweep in decades. |
| `strength` (γ) | `...target_residual_completion.strength` | **only meaningful if alpha is NOT searched.** With alpha search, the model is `θ + α·γ·Δτ` and γ merely rescales the α grid. The shipped `s0/s0p5/s1` configs *do* search alpha, so they are near-redundant — and `s0` is a pure no-P1 control. |
| `exact_form` | `...target_residual_completion.exact_form` | `true` = affine + intercept (needs `missing_bias: materialize`); `false` = reduced (needs `skip`). A real ablation here, unlike vision where the bias always exists. |
| `target_scope` | `...target_residual_completion.target_scope` | `all` vs `inserted`. |
| `num_batches` | `...target_residual_completion.num_batches` | calibration rows for the fit. Interacts with ρ — sweep jointly, not separately. |
| calibration source | `block_extension_params.calibration_dataset` | e.g. the hellaswag-calibrated variant, which pairs with explicit `harness_samples` to keep the eval hold-out identical. Cleanly separates "fit data" from "eval data". |

**Inert — do not sweep:** `method_params.*`, `ridge_identity`, `n_cascade_iters`,
`lmc_mode`, `extension_strategy` affect the ARIADNE-resized source. On the
*transport* arm (the only LLM arm) they are **not** inert the way they are for
vision's direct arm — the transported vector does depend on them — so treat them
as real but secondary, and hold them fixed for comparability.

---

## 8. Cost and memory

Hold live at once: the resized source (24→28 blocks, bf16), the 1.5B target, two
calibration loaders (source and target tokenizers), and 100-batch activation
banks. On a 16 GB card this is the binding constraint.

- Start with `calibration_batch_size: 4`, `harness_batch_size: 4` as shipped; lower
  before raising.
- Each alpha re-runs the full harness — 9 alphas means 9 IFEval passes.
- **Run one LLM arm at a time per GPU.** Do not overlap two; the historical failure
  here is a CUDA OOM.
- Smoke first with `configs/llm_rebase_it_smoke_qwen.json` or
  `llm_rebase_qwen0.5b_samesize_harness_smoke.json` to shake out environment drift
  (the venv ships `transformers 5.0.0` / `datasets 4.5.0` while the code was
  written against 4.x).

---

## 9. Landmines, collected

1. ~~`mode: "direct_target"` silently runs the transport path.~~ **Fixed** —
   it now runs the transport-free solve. See §1.
2. P1 is still a **silent no-op** if the transported-body branch is skipped —
   check `blocks fitted > 0` every time. The *other* half of this (no
   block-extension pre-step) now raises up front instead.
3. The six `..._p1_ridge100_s*.json` configs are unrunnable at γ>0 and silently
   vacuous at γ=0.
4. Harness extras (`langdetect`, `immutabledict`, `math_verify`) and NLTK
   `punkt_tab` are missing — install before anything.
5. `google/IFEval` is not cached; first run needs network.
6. Every config's `local_log_dir` points at a Leonardo path that does not exist here.
7. `HF_HOME` points at `/home/aba/.cache/huggingface`, not the invoking user's home.
   If that becomes unreadable, everything re-downloads.
8. `method_params.n_batches` is rejected — the key is `num_batches`.
9. The eval hold-out is ~141 docs. Treat single-run differences with suspicion
   until §6 is done.
10. No LLM run in this checkout has ever completed with P1 — every historical LLM
    summary here is either a pre-P1 smoke run or a failure. You are the first; do
    not assume a green run means a correct one.

---

## 10. File map

| what | where |
|---|---|
| LLM rebase entrypoint | `src/merge_and_rebase/eval/llm_rebase.py` |
| P1 capture / completion helpers | same file, `_maybe_capture_target_residual_references` (225), `_maybe_complete_target_residual_task_vector` (260) |
| solver (architecture-agnostic) | `src/merge_and_rebase/eval/target_residual_completion.py` |
| orchestration + decoder layout | `src/merge_and_rebase/eval/target_informed_runtime.py` (`_DecoderLayout` ~266) |
| decoder depth resize | `src/merge_and_rebase/eval/block_extension_llm.py` |
| family adapters | `src/merge_and_rebase/rebase/model_families/hf_decoder.py`, `registry.py` |
| harness wrapper | `src/merge_and_rebase/eval/lm_harness_runner.py` |
| calibration / hold-out split | `src/merge_and_rebase/data/llm_calibration.py` |
| decoder P1 tests | `tests/test_target_informed_runtime_decoder.py` |
| vision precedent for the method | `scripts/run_direct_p1_sweep.py` (staged sweep, memory guard, resumable, live report) |

`scripts/run_direct_p1_sweep.py` is vision-only but is the working reference for
the *driver* pattern — staged stages, a gate computed from one stage to build the
next, a memory guard before each launch, idempotent skip-if-done, and a live
`REPORT.md`. Port that shape rather than inventing one.

---

## 11. The `direct_target` wiring, as built

**Done as of 2026-09-21.** This section described a trap and scoped a fix; both
are now in the tree. What follows is what was actually built, so the next reader
can check it rather than re-derive it.

### 11a. The guard that was proposed — and what replaced it

The proposed guard refused `mode != "transport_residual"` outright. That is no
longer the right shape: the mode is implemented, so refusing it would refuse a
working arm. The guard that went in instead closes the *other* silent no-op, the
one no arm survives — an enabled `target_residual_completion` whose
block-extension pre-step will never run:

```python
if residual_completion_cfg.enabled and not run_block_extension_prestep:
    raise ValueError(...)   # names which condition failed
```

It sits immediately after `run_block_extension_prestep` is computed, so a bad
config dies in seconds rather than after a resize and a transport fit.

### 11b. How the arm is wired

All four obstacles the original section listed, and what each became:

1. **Branch gating.** The completion branch reads
   `if method_name in ("theseus","theseus_gqa","bico") and transport_keys:`.
   `transport_keys` comes from the family adapter's transportable-key set, not
   from a fit, so it is non-empty in direct mode too and the branch is entered
   normally. No second branch was needed — the skip happens *inside*, around the
   `method.prepare` / `method.transport` pair.
2. **Empty-delta contract.** `fitted_prepared = None` and
   `transported_body = {}` in direct mode. The helper asserts the delta is empty
   **before** the materialized-bias seeding, which would otherwise populate it
   and make a genuinely transport-free vector look transported.
3. **Passthrough keys — decided: dropped.** See §1, decision 1. Recorded in the
   summary as `passthrough_policy` + `dropped_passthrough_key_count`.
4. **`projection_transforms` is not called** on the direct path, matching
   vision's `prepared=None`.

Two extras the original scope did not anticipate:

- `delta_norm_match: "uncorrected"` is refused in direct mode. It rescales the
  transported vector to the *source* task vector's norm; the direct correction
  is fitted natively in target coordinates, so the ratio is meaningless and
  would silently override γ.
- A zero merged delta is fatal everywhere **except** direct-mode γ=0, where an
  identically zero task vector is exactly what the native-target-base control
  looks like. There it prints and proceeds.

### 11c. Coverage

`tests/test_llm_direct_target_p1.py`, 11 tests, all passing. The decoder
plumbing the vision suite could not reach:

- `attn_proj_input` capture on a decoder — the hook fires once per batch, and
  pushing the captured rows back through `o_proj` reproduces the attention
  output. (Decoders need **none** of the recompute machinery vision required:
  there `nn.MultiheadAttention` applies `out_proj` functionally and a hook on it
  never fires at all.)
- `materialize_missing_projection_biases` over both components, and its default
  still being the historical MLP-only set.
- Direct completion end to end on a width-up, depth-up decoder pair (8→12 wide,
  3→4 deep, one duplicated ancestry group): only the projection keys are
  written, the first fitted block reports `relative_residual_before == 1.0`,
  every block improves on its own objective, `interpolate` yields the expected
  fractional coordinates `{0: 0.0, 1: 0.5, 2: 1.0, 3: 2.0}`, and the solver
  restores the model bit-for-bit.
- γ=0 against the explicit zero baseline is identically zero; γ=1 is not.
- `llm_rebase`'s own helper refuses a non-empty delta in direct mode, and
  returns a task vector containing **no** transported keys.

### 11d. Configs

- `configs/llm_rebase_qwen0.5b_to_1.5b_ifeval_theseus_p1direct_smoke.json`
  — plumbing smoke: 8 activation batches, 4 completion batches, 16 IFEval docs,
  2 alphas. Launcher: `scripts/run_p1direct_qwen0.5b_to_1.5b_smoke.slurm`.
- `..._p1direct_s1.json` — the reference direct arm, γ=1, 9 alphas.
- `..._p1direct_s0.json` — the γ=0 control. `alpha_search` is off: the task
  vector is identically zero, so nine alphas would be nine identical IFEval
  passes.
