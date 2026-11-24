# Synthetic calibration data, IFEval (2026-11-19)

Question: does calibration data that resembles the evaluation distribution, but is not taken from it, recover the
gain that generic calibration loses (vision KS2_track_O2_generic: EuroSAT +2 instead of +13)? Does it help the
kernel more than ARIADNE?

## Setting (identical across all 12 runs; only the calibration set and the method vary)
- Transfer: Qwen2.5-0.5B -> Qwen2.5-1.5B, task vector = Qwen2.5-0.5B-Instruct - Qwen2.5-0.5B (ARIADNE's best LLM cell).
- Calibration: 40 texts (10 batches x 4), max 448 tokens. Answers appended by the source fine-tuned model
  (Qwen2.5-0.5B-Instruct, 256 new tokens, plain prompt), as in ARIADNE's best setting; none for WikiText.
  - real: 40 IFEval prompts (the calibration part of `holdout_split`, seed 0)
  - claude: 40 of 50 prompts written by Claude from the IFEval instruction *types* (`calib/claude_prompts_raw.json`;
    overlap with IFEval only on the instruction-template wording, `calib/overlap_check.json`)
  - self: 40 prompts written by Qwen2.5-0.5B-Instruct (`gen_self_prompts.py`, meta v2; v1 output mostly JSON-wrapped)
  - wiki: 40 WikiText-103 *test* paragraphs (disjoint from the anchor pools)
- Methods: `kern` = D3 kernel (canonical_soft, held-out weights, T = 65,536, k = 16/doc, WikiText-103 train pool,
  sequential, ridge); `ariadne` / `ariadneO2` = ARIADNE via the bridge (procrustes_cross, map_rows task, T = 4,096),
  plain and + O2 (ridge_preserve, rho_g 0.1).
- alpha protocol: IFEval 541 prompts = 40 calibration (excluded everywhere) / 150 selection / 351 test.
  Selection metric = mean of instruction- and prompt-level strict accuracy; grid 0..3; alpha = 1 also scored on test.
- References on the same prompts: Qwen2.5-1.5B (alpha 0), Qwen2.5-1.5B-Instruct (native fine-tune).
- Code: `results/_code_snapshots/20261119e_svdfix` (commit 1d22888 + gesvd fallback of the canonical SVD + IFEval avg
  in the split eval). Single seed.

## Files
- `chain_gpu1.sh` (fits, GPU 1, holds the launch lock), `evals_gpu0.sh` (all split evals, GPU 0)
- `logs/chain_gpu1.log`, `logs/evals_gpu0.log`, `logs/split_<run>.json`
- 2026-11-20 05:03 LRT_kern65k_seq_mathprompt_e fit failed: CUDA OOM (512-prompt bank used whole). Rerun with 40-prompt subset (calib/llama_mathprompt_40.json) + idle-GPU1 IFEval evals from list end: gpu1_after_fits.sh
- 06:24 mathprompt40 (batch 4) OOMed again; v1 worker replaced by gpu1_after_fits_v2.sh: batch 2 x 20 (config LRT_kern65k_seq_mathprompt40b2.json)

## Experiment 5: calibration size (user, 2026-11-20)
- Question: does more calibration data help the kernel more than ARIADNE? Calibration = 40 / 160 / 320 Claude-written
  prompts, nested (calib/claude_{40,160,320}.json; 270 new prompts from calib/make_claude_more.py, overlap with IFEval
  only on instruction wording, calib/overlap_check_more.json: max common string 16 words, max TF-IDF cos 0.67).
  Answers by Qwen2.5-0.5B-Instruct as before (new cache per size). num_batches = n/4.
- Methods (user): kernel + O2 and ARIADNE + O2, both depth_fit independent (O2 on decoders exists only for
  'independent'; ARIADNE + O2 already ran that way). So kernel + O2 differs from IF_kern_* (sequential, plain ridge);
  its own 40-prompt point is IFS_kernO2_claude40. ARIADNE + O2 40 = IF_ariadneO2_claude.
- Runs: IFS_kernO2_claude{40,160,320}, IFS_ariadneO2_claude{160,320}; worker gpu1_scaling_v3.sh (GPU 1).

### 2026-11-20 13:30 — IFS_kernO2_claude160 killed out of memory
rc=137 after 13 min, peak RSS 119.5 GB (logs/IFS_kernO2_claude160.time). Same config as claude40 except 160 sequences
(num_batches 40). Peak RSS at 40 prompts: kernO2 48.1 GB, kern 43.4 GB, ariadneO2 33.8 GB. The fit keeps the
calibration activations in RAM, so memory grows with the number of calibration tokens; the code has no token cap.
gpu1_scaling_v3.sh continues with IFS_ariadneO2_claude160 (started 13:29:55); kernO2_claude320 is expected to fail
the same way. Fix (fixed token budget, or streaming/lower precision of the activations) waits for the user.

### 2026-11-24 — experiment 5 restarted with calibration_streaming
Code: results/_code_snapshots/20261124_streaming (new anchors/calibration_streaming.py, flag `calibration_streaming`,
default off; implementer + reviewer 24/11). Configs IFS_*_claude{40,160,320}s = old configs + calibration_streaming true,
chunk 4. Worker gpu1_streaming.sh (log logs/gpu1_streaming.log): 40-prompt equivalence vs the resident runs
(compare_tvs.py, gate global rel L2 < 1e-3 kernel / 1e-2 ARIADNE), then 160/320 fits and split evals of passing methods.
Reviewer caveat: the ARIADNE Procrustes map may amplify float32-vs-float64 differences on badly conditioned blocks.
