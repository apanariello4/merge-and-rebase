# Anchors (shared-anchor rebasin) — experimental

`anchors` transfers a fine-tuning effect through one coordinate frame shared by every model: `T` anchor inputs
that all models read. Each model is registered once (two forward passes over a generic image pool); an expert is
encoded as its displacement's similarities to the source anchors and decoded on any registered target. The
decoded displacement is compiled into the target with **Ariadne's ridge, unchanged** (`fit_direct_residual`).

Status: experimental (branch `exp/un_reb`), vision (OpenCLIP) only. LLM configs with `method: anchors` are
rejected.

## Pipeline

Per target block `j`, with source block `i = round(j (L_A - 1) / (L_B - 1))` (Ariadne's relative pairing):

| Step | Quantity | Code |
| --- | --- | --- |
| Register (once per model) | `mu`, `c = ‖S_reg − mu‖_F / √n`, `P = (S_anc − mu) / c`, per block | `AnchorState.register_model` |
| Encode | `R = (E / c_A) P_Aᵀ`, `E = S_ft − S_base` **not centred** | `core.encode_rows` |
| Decode | `D = c_B · R (K_B + εI)⁻¹ P_B`, `ε = eps_rel · tr(K_B) / T` | `core.decode_rows` |
| Compile | ridge on the target base inputs of `mlp.c_proj` (bias recovers the mean of `D`) | `fit_direct_residual` |

Token rows live on a canonical grid (`canonical_tokens`, default CLS + 14×14). Each model's tokens are
bilinearly resampled to it, and `D` is resampled back to the target's native grid before the ridge. All of these
maps are linear row maps, so their order does not matter.

## Data

- **Frame** (anchors, `mu`, `c`): a generic pool, `anchor_pool: tiny_imagenet` (default) or `vision8_mix`. The
  first `anchor_batches` batches of the seeded pool draw hold the anchor images; the next `registration_batches`
  hold the registration images. The two sets are disjoint.
- **Task calibration** (`E` and the ridge rows): the task's own `paired_calibration` draw, with the same
  `num_batches` and `seed` as Ariadne.

## State

With `state_dir` set, the frame and the registered models persist:

- `frame.json` holds the frame config, the pool draw and the anchor rows. It is written once.
- `models/<id>/blocks.pt` and `meta.json` hold the per-block data (`P`, `mu`, `c`) and a sha256 of `blocks.pt`.

Registering a new model writes only under `models/<id>/`. Opening a state with a different frame config fails.

## Kernel diagnostics (M2)

`python -m merge_and_rebase.eval.diagnostics.anchors_kernels --config configs/anchors/m2_kernels_b16_l14.json`
reports, per pool, pair, `T` and depth:

- the CKA of the centred anchor kernels and `‖K_A − K_B‖_F / ‖K_A‖_F`;
- a permutation null and bootstrap intervals;
- kernel conditioning;
- the cross-depth CKA matrix.
