"""Configuration of the shared-anchor method (Step 0 of the reference document).

Two groups of fields:

- **frame fields** (``FRAME_FIELDS``) define the common coordinate system: which pool the anchor and
  registration images come from, how many anchors, the canonical token grid, the anchor draw. They are stored
  with the method state and must be identical for every model ever registered in it (a later model C reuses
  them verbatim);
- **per-run fields** (decode and ridge options, task calibration budget) may change between runs on the same
  state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

from .._ariadne.config import DirectResidualConfig
from .core import KERNEL_DECODES

ANCHOR_POOLS = ("tiny_imagenet", "vision8_mix", "wikitext103")
#: Generic text pools of a decoder (LLM) frame, whose anchor rows are tokens (AGENT_NEXT_PHASE.md L1).
TEXT_ANCHOR_POOLS = ("wikitext103",)
ANCHOR_UNITS = ("token", "image_mean", "cls", "pool2", "pool4")
K_SAMPLINGS = ("k_per_image", "k_per_document")
CANONICAL_SOFT_WEIGHTS = ("in_sample", "heldout")
ANCHOR_SAMPLINGS = ("random", "whole_images", "one_per_image", *K_SAMPLINGS)

FRAME_FIELDS = (
    "anchor_pool",
    "num_anchors",
    "anchor_batches",
    "registration_batches",
    "pool_seed",
    "anchor_seed",
    "canonical_tokens",
    "anchor_positions",
    "anchor_sampling",
    "anchor_tokens_per_image",
    "anchor_unit",
    "text_units",
)
#: Row units of a decoder (text) frame and calibration: tokens (identical tokenizers) or text-defined words.
TEXT_UNITS = ("token", "word")


@dataclass(frozen=True)
class AnchorsConfig:
    # ---- frame (stored with the state, identical for every registered model)
    # Generic image pool for anchors and registration statistics (mu, c); the task data never enters the frame.
    anchor_pool: str = "tiny_imagenet"
    # T: dimension of the common space. T <= d of every model gives exact round trips and cycle consistency.
    num_anchors: int = 512
    # The pool draw is anchor_batches + registration_batches batches from paired_calibration(seed=pool_seed):
    # the first anchor_batches hold the anchor images, the rest the registration images (disjoint by construction).
    anchor_batches: int = 4
    registration_batches: int = 10
    pool_seed: int = 0
    # Seeded choice of the T (image, token position) anchor rows among the anchor images' canonical tokens.
    anchor_seed: int = 0
    # Token rows live on one fixed grid (CLS + square patch grid): 197 = CLS + 14x14 (ViT-B/16 at 224px).
    canonical_tokens: int = 197
    # 'all' (CLS and patch positions) or 'patch' (anchor rows drawn from patch positions only).
    anchor_positions: str = "all"
    # How the T anchor rows are drawn from the anchor images x positions. 'random': T rows uniformly at random
    # (a few tokens from each of many images). 'whole_images': every position of T / n_positions images (seeded
    # image choice; T must be a multiple of n_positions). 'one_per_image': one random position from each of T images.
    # 'k_per_image' (vision) / 'k_per_document' (decoder): exactly anchor_tokens_per_image = k rows from each of T / k
    # seeded images (documents), the k rows drawn at random within the image (document); T must be a multiple of k, and
    # a document needs >= k candidate rows. The rows then carry the image / document id used by the half splits of
    # canonical_stats (a split is by id, never by row).
    anchor_sampling: str = "random"
    anchor_tokens_per_image: int | None = None
    # What one anchor row is (AGENT_KERNEL.md R1). 'token': one canonical token (image, position). Centroid anchors,
    # the same members in every model: 'image_mean' (mean of the image's patch tokens), 'cls' (the CLS token),
    # 'pool2' (mean over 2 x 2 patch regions: a 7 x 7 grid on 14 x 14), 'pool4' (adaptive average pooling of the
    # patch grid to 4 x 4 regions of 3-4 patches). Centroid rows are drawn at random over (image, region) and centred
    # by the mean of the same unit over the registration images; c stays the token spread (the scale of E).
    anchor_unit: str = "token"
    # Decoder rows (text pools only). 'token': one row per content token, every model must see identical token ids
    # (hash guard). 'word': one row per word of the raw text (maximal non-whitespace run), represented in each model
    # by the last token of the word in its own tokenization (character offsets); a word whose last token crosses the
    # word boundary, or is cut by truncation, is missing in that model only. Tokenizers may then differ: the anchors
    # are drawn among the words present in every model registered with the frame, the calibration rows are the
    # words present in both source and target (anchors/text_units.py).
    text_units: str = "token"
    # ---- per run
    # eps = eps_rel * tr(K) / T in the decode solve.
    eps_rel: float = 1e-3
    # True: D carries the target scale c_k; False: the source scale c_m (Ariadne leaves scale to alpha).
    rescale_target: bool = True
    # Map between the source and target anchor frames. 'pairwise': the encode/decode of the kernel (with
    # spectral_filter). 'gpa': generalized Procrustes (AGENT_NEXT.md section 4, universe.py): one map A_m (d_U x d_m,
    # orthonormal columns) per model into a common centre G(s) estimated on the stored anchors;
    # D = (scale / c_m) E A_m[:r]^T A_k[:r]. gpa_dim: d_U (None = widest model). gpa_rank: r, the nested truncation
    # to the top-r axes of G (None = d_U). gpa_models: further registered models fitted into G with source and target
    # (G3). gpa_add_models: registered models added afterwards by a Procrustes towards the fixed G (G4/G5).
    frame_map: str = "pairwise"
    # High-norm anchor filter (AGENT_NEXT step 2, variant F1). None: off. 'norm': drop the anchors whose centred norm
    # exceeds anchor_filter_k x the block median in the source OR the target block (any fitted model for GPA), and
    # use mu, c recomputed without such registration rows (AnchorState.norm_filter, stored once per model).
    anchor_filter: str | None = None
    anchor_filter_k: float = 3.0
    gpa_dim: int | None = None
    gpa_rank: int | None = None
    gpa_models: tuple[str, ...] = ()
    gpa_add_models: tuple[str, ...] = ()
    # frame_map='consensus' (AGENT_ADDENDUM_CONSENSUS section 2, universe.fit_consensus): the centre is the consensus
    # kernel K* of the double-centred, Frobenius-normalised anchor kernels of the fitted models (gpa_models with source
    # and target; gpa_add_models are registered afterwards against the FROZEN K*). consensus: mean | logeuclid | robust.
    # Z* = V_k Lambda_k^(1/2) (consensus_k axes). consensus_register: rot (polar(P^T Z*), isometric) | correct
    # (P^T (K + eps I)^-1 Z*: the anchors land on Z*; decode by least squares from Z*) | zca_rot (rotation between the
    # ZCA-whitened frame and the whitened consensus V_k). consensus_eps: relative to tr(K)/T.
    consensus: str = "mean"
    consensus_k: int = 512
    consensus_register: str = "rot"
    consensus_eps: float = 1e-2
    # Spectral filter of the decode: 'tikhonov' (eps above) or a hard cut at spectral_rank directions,
    # 'truncate_target' | 'truncate_cross' | 'procrustes_cross' (core.spectral_transport_map). None = full rank.
    spectral_filter: str = "tikhonov"
    spectral_rank: int | None = None
    # Kernel decode of the pairwise map (AGENT_KERNEL.md; V1 = spectral_filter 'truncate_target' + spectral_rank).
    # 'tikhonov': D = scale R (K_k + eps I)^{-1} P_k (V0). 'zca' (V2): each model whitened by its own kernel,
    # D = scale (E / c_m) Q_m^T Q_k, Q = (K + eps I)^{-1/2} P, eps = eps_rel tr(K) / T per model (pseudo-inverse
    # root at eps_rel = 0). 'agreement' (V3): V2 restricted to the directions where the two whitened kernels agree,
    # 1/agreement_tau <= gamma_i <= agreement_tau (core.whitened_transport_map).
    kernel_decode: str = "tikhonov"
    agreement_tau: float | None = None
    # 'canonical_hard' / 'canonical_soft' (core.canonical_transport_map): Phi = (scale / c_m) V_m F V_k^T with
    # M = U_m^T U_k = Y Gamma Z^T (P = U S V^T); hard F = Y_r Z_r^T (canonical_rank = r, None = all pairs), soft
    # F = Y g(Gamma) Z^T with g(G) = G for G >= canonical_null_threshold and 0 below. Refused when T <= r_m + r_k
    # (>= r_m + r_k - T canonical correlations are 1 from noise alone); the ranks are only known at the block pair
    # (core.check_canonical_widths). The config path of canonical_soft is IN-SAMPLE (g from Gamma of the fitting rows;
    # info 'canonical_soft_weights' = 'in_sample', UserWarning): it is NOT plan decoder D3 (held-out Gamma_out, passed
    # as core.canonical_core(weights=...), 'explicit'), which is wired as canonical_soft_weights='heldout' below.
    # canonical_soft_weights (canonical_soft only): 'in_sample' (default, not recorded) or 'heldout' = plan decoder D3
    # (PIANO_METODO_K.md Fase 2): the directions Y, Z and V of the decoder come from ALL anchor rows, the weights are
    # g_i = clip(Gamma_out_i, 0, 1), Gamma_out_i = the correlation of the canonical pair i estimated on the half of the
    # anchor images / documents not used to find the directions, averaged over the two directions of the split (A -> B,
    # B -> A: the Phase 1 estimate, canonical_stats.heldout_gamma_out), matched to the full-set pairs by rank index;
    # canonical_null_threshold then zeroes the weights below it. The split is by image / document id (never by row),
    # seeded by anchor_seed; needs anchor_sampling 'k_per_image' / 'k_per_document' (rows carry ids) and no anchor_dedup_cos.
    canonical_soft_weights: str = "in_sample"
    # canonical_source: where the held-out canonical statistics come from. 'rows' (default, not recorded): the stored anchor
    # frames (num_anchors rows). 'sigma_all_tokens' (vision, canonical_soft + heldout): streamed second moments over EVERY
    # canonical token of the frame's anchor images (canonical_images = anchor_batches x batch size, checked), per half of the
    # image ids and per block pair, cached on disk (anchors/canonical_stream.py); the E rows and the rest are unchanged.
    canonical_source: str = "rows"
    canonical_images: int | None = None
    canonical_rank: int | None = None
    canonical_null_threshold: float | None = None
    # Anchor post-processing (AGENT_KERNEL.md K1b), applied to the stored frames at use time (no new registration).
    # anchor_row_norm (R3): the centred anchor rows of every model scaled to unit norm before P, K and any rotation;
    # E is not normalised and the decode scale stays c_k / c_m. anchor_dedup_cos (R2): keep only anchors whose pairwise
    # cosine, measured in the run's SOURCE model at the source block (after the norm filter), is below this value
    # (greedy, seeded by anchor_seed); anchor_dedup_max caps how many are kept (None: all that pass).
    anchor_row_norm: bool = False
    anchor_dedup_cos: float | None = None
    anchor_dedup_max: int | None = None
    # Expert library (AGENT_NEXT_PHASE.md O1). 'data': D from the source displacement rows (the method so far).
    # 'operator': per source block a ridge operator e ~ h W_E + b_E (h, e centred/scaled by the source frame), carried
    # into the GPA space (W_U = A_m W_E A_m^T) and applied to the TARGET's own base activations (anchors/operators.py).
    # operator_rank: data-weighted rank of W_E (None: full). operator_holdout_batches: extra calibration batches (the
    # tail of the same seeded draw, disjoint) for the held-out R^2. target_calibration 'disjoint': the target's H and
    # h_k come from num_batches further batches of that draw, never seen by the source (O1.3).
    library: str = "data"
    operator_rank: int | None = None
    operator_holdout_batches: int = 2
    target_calibration: str = "same"
    # operator_norm_filter (needs anchor_filter='norm'): the source rows whose centred norm exceeds anchor_filter_k x
    # the block median are left out of the operator fit and its R^2, and the target rows above that bound in the
    # target block get D = 0 (the GPA maps were fitted without such rows, so A_k^T is not defined on them).
    operator_norm_filter: bool = False
    # operator_space 'native': fit e ~ h in the source's own coordinates, then W_U = A_m W_E A_m^T; 'common': fit
    # directly e A_m^T ~ h A_m^T in the GPA coordinates (the operator sees the inputs it will be applied to).
    # operator_input_map: how the target's common coordinates u_k = h_k A_k^T are carried onto the source's before
    # the operator: 'gpa' (identity, the shared frame alone), 'anchor_ridge' (ridge u_k -> u_m on the shared anchor
    # frames of the block pair: still no task data from the target side), 'paired' (the same ridge on the task
    # calibration rows of both models: a ceiling, it needs the source and target on the same task images),
    # 'pool_ridge' (that ridge on all tokens of the frame's anchor images, both models: task-free, many rows).
    operator_space: str = "native"
    operator_input_map: str = "gpa"
    # operator_cls_split (needs operator_space='common'): one operator for the CLS row and one for the patch rows
    # (the readout's CLS row is poorly predicted by an operator shared with ~200 patch rows per image), and, for the
    # 'pool_ridge' / 'paired' input maps, one input map each as well; decode: target CLS row by the CLS pair.
    operator_cls_split: bool = False
    # Optional: save each expert's operators {grid point j: (W_U, b_U)} as <dir>/<universe version>/<source>__<task>.pt
    # (with the source scales), for decode-only fusion in the common space (O4 F2/F3).
    operator_library_dir: str | None = None
    # O4 F2/F3, decode-only fusion: operator_fusion_entries = paths of target-side library entries
    # (<source>__<task>__composed.pt, all towards this run's target and in this run's GPA frame); no source operator
    # is fitted. Per grid point the entries are fused by operator_fusion 'sum' (sum_i w_i W_i, sum_i w_i b_i) or
    # 'tsv' (per entry the top r/K singular directions of w_i W_i, stacked, U and V orthogonalised as in TSV; biases
    # averaged), then ONE decode D = (u W + b) A_k and ONE ridge. operator_decode_rows: the target rows of that
    # decode and ridge, 'task' (this task's calibration draw) or 'pool' (the frame's anchor images: task-free, the
    # same update for every task of the run).
    operator_fusion_entries: tuple[str, ...] = ()
    operator_fusion: str = "sum"
    operator_fusion_weights: tuple[float, ...] = ()
    operator_decode_rows: str = "task"
    # Update synthesis (AGENT_NEXT_PHASE.md O2; the default since AGENT_SHARED_SPACE.md S0). 'ridge': Ariadne's ridge
    # (unchanged). 'ridge_preserve' (D2): weight and bias solved jointly with an extra (rho_g / m)||G dW + 1 beta^T||^2
    # term, G = the target base's c_proj (decoders: mlp.down_proj) inputs on the frame's anchor data: the anchor images
    # of the Tiny-ImageNet pool (vision: preserve_tokens tokens per block, high-norm tokens dropped when the anchor
    # filter is on) or the content tokens of the anchor sequences of the text pool (decoders, any text_units: BOS /
    # first token and padding excluded, at most preserve_tokens per block, G^T G built one block at a time inside the
    # solve; anchors/synthesis.py). rho_g = 0 is Ariadne's ridge.
    # Default rule (None = not given, preserve_by_default): synthesis resolves to 'ridge_preserve' wherever it is
    # implemented (preserve_supported: components = (mlp.c_proj,), depth_fit 'independent' or 'spread', anchor_pool
    # 'tiny_imagenet' or 'wikitext103'; decoders also need depth_fit 'independent') except on the bridge to Ariadne
    # map_rows='task' (stays Ariadne by default; explicit ridge_preserve there = Ariadne + O2), and to 'ridge'
    # elsewhere (depth_fit 'sequential', attn.out_proj, the mixed Vision8 pool). rho_g resolves to DEFAULT_RHO_G = 0.1
    # under 'ridge_preserve' and to 0 under 'ridge'. The resolved values are the ones stored by to_dict(). An explicit
    # 'ridge_preserve' on an unsupported combination, or an explicit nonzero rho_g with 'ridge', is an error that
    # names the fix (synthesis: ridge).
    synthesis: str | None = None
    rho_g: float | None = None
    preserve_tokens: int = 32768
    # D5 (O4): the compiled c_proj update reduced to update_rank by the data-weighted projection dW V_r V_r^T (V_r
    # from the fitted displacement's scatter dW^T H~^T H~ dW), intercept recomputed; None: full rank.
    update_rank: int | None = None
    # Rows the transport map is estimated from. 'frame': the registered anchors of the shared frame (the method).
    # 'task': the task's own paired-calibration rows (base-model boundaries, same seeded draw as E), the bridge to
    # Ariadne: with procrustes_cross at full rank, rescale_target=False and the target's token grid as canonical
    # grid it is Ariadne's Procrustes. map_num_batches: batches of that draw (None = num_batches; the draw is
    # prefix-consistent, so the first num_batches are E's images); map_rows_per_image: token rows kept per image
    # (None = all); map_num_rows: rows kept in total after that (None = all); map_seed: seed of both subsamples.
    map_rows: str = "frame"
    map_num_batches: int | None = None
    map_rows_per_image: int | None = None
    map_num_rows: int | None = None
    map_seed: int = 0
    # Task calibration (E and ridge rows), same budget and draw as Ariadne's paired_calibration.
    num_batches: int = 10
    seed: int = 89
    # Where the calibration batches (E capture, the synthesis fit, gain / refine holdouts) come from. 'task' (default,
    # unchanged): the task's own val images. 'generic': num_batches x batch_size images of the anchor pool
    # (tiny_imagenet), drawn with `seed` from the part of the pool that the frame (registration + anchor images),
    # ridge_preserve's G and refine's keep loss never touch; labels are not used. Vision, map_rows='frame', library='data'.
    # 'precomputed': stored preprocessed image tensors <calibration_images>/<Task>.pt (task-data-free sets built by
    # eval/vision_rebase/build_calibration_images.py: retrieval / DeepInversion), the first num_batches * batch_size in
    # stored order, the same tensors for source and target (their preprocessing must agree); sha256 recorded.
    calibration_source: str = "task"
    calibration_images: str | None = None
    components: tuple[str, ...] = ("mlp.c_proj",)
    ridge_estimator: str = "empirical_bayes"
    ridge_relative: float = 0.01
    exact_form: bool = True
    missing_bias: str = "error"
    strength: float = 1.0
    depth_pairing: str = "relative"
    # How the per-block D_j are compiled across depth. 'independent': every block j regressed on the cumulative D_j
    # (Ariadne's resident fit; upstream contributions add up, the realized change overshoots D_j). 'sequential':
    # blocks fitted in order on D_j - U_j, U_j = the change the already-fitted upstream blocks produce at j
    # (Ariadne's sequential streaming fit with the anchor decode as desired_fn; host memory O(1) in depth).
    depth_fit: str = "independent"
    # depth_fit 'tracking' (S2, anchors/synthesis.py fit_tracking): error-feedback fit in depth order. D_j is already the
    # CUMULATIVE stream-level change at block j's boundary (E = S_ft,i - S_base,i), so block j writes what is still
    # missing, D_j - U_j (U_j = realized change at boundary j of the already updated upstream blocks), fitted on the
    # c_proj inputs of the updated model, through the same synthesis hook (plain ridge or ridge_preserve). With the plain
    # ridge it is Ariadne's sequential fit, resident, with the hook; 'sequential' + ridge_preserve (S1) runs the same loop.
    # depth_fit 'spread' (D3, AGENT_NEXT_PHASE.md O3): D is used only at the measurement depths measure_depths
    # (target blocks round(s (L_k - 1))); the blocks of each interval are fitted in order to equal shares of what is
    # still missing at its measurement block (anchors/synthesis.py fit_spread); blocks after the last one: no update.
    measure_depths: tuple[float, ...] = ()
    # S1/S2 (sequential/tracking with ridge_preserve): which inputs the preserve statistics G are taken on. 'base': the
    # target base's c_proj inputs on the anchor images, captured once (as in the independent O2); 'updated': the inputs
    # of the model with blocks 0..j-1 already updated, re-captured for every block (one anchor sweep per block).
    preserve_inputs: str = "base"
    # S3 (label-free gain calibration): 'none' (default, unchanged) or 'lsq': after the fit of each block, in depth order,
    # a scalar g_j = <own, D_j - U_j> / <own, own> (own = the block's own realized contribution H dW + 1 beta^T on the
    # current inputs, U_j = upstream realized change) on the last gain_holdout_batches calibration batches, which are
    # then NOT used by the fit (0: in-sample, all rows); dW, beta are multiplied by g_j (clipped to GAIN_BOUNDS).
    gain_calibration: str = "none"
    gain_holdout_batches: int = 2
    # Implementation of the native closed-loop depth fit (fit_tracking: depth_fit 'tracking', or 'sequential' with
    # ridge_preserve / gain). 'resident' (default, unchanged): every block re-runs a full target forward per calibration
    # batch (N (N + 1) / 2 block forwards per batch). 'incremental' (vision open_clip ViTs only): the residual stream of the
    # updated model is carried from block to block (2 block forwards per batch and block); numerically equivalent.
    sequential_impl: str = "resident"
    # refine 'joint' (anchors/refine.py): AFTER the depth fit, all (dW_j, beta_j) are trained jointly, label-free, by Adam on
    # the stream-level losses of the module docstring (L_task at the measurement blocks refine_depths, a closed-loop keep
    # term rho on generic anchor-pool rows, optional proximity to the fit), best checkpoint on refine_holdout_batches
    # held-out calibration batches (the last ones). 'none' (default): nothing changes. refine_depths: 'all' or relative
    # depths (blocks round(s (L - 1))); default 6 evenly spaced, last included. refine_target 'final': only the last
    # measurement block enters L_task. refine_lr is relative to rms(dW_j) of the fit (per element, Adam), with a linear warmup of refine_warmup * steps;
    # refine_target 'final' scales rho by |task blocks| / |M|; with refine != none the depth fit uses only the calibration
    # batches that are not held out. refine_rho (0.1 = DEFAULT_RHO_G) / refine_prox are dimensionless.
    refine: str = "none"
    refine_depths: str | tuple[float, ...] = (1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0)
    refine_target: str = "all_depths"
    refine_rho: float = 0.1
    refine_prox: float = 0.0
    refine_steps: int = 300
    refine_lr: float = 1e-3
    refine_warmup: float = 0.1
    refine_batch: int = 8
    refine_holdout_batches: int = 2
    refine_eval_every: int = 25
    refine_generic_batches: int = 8
    refine_amp: bool = False
    refine_checkpoint: bool = True
    # None: every target block is compiled. A float s_max in (0, 1]: only target blocks j with relative depth
    # j / (L_k - 1) < s_max keep their update, the others are zero (exact: the ridge fits blocks independently).
    compile_max_depth: float | None = None
    # Optional on-disk state directory (frame + registered models); None keeps the state in memory.
    state_dir: str | None = None
    # Analysis-only: per-position coverage of E by the source anchors (one d x d scatter per position).
    coverage_diagnostics: bool = True
    # Streaming calibration (anchors/calibration_streaming.py; decoders, depth_fit 'independent', library 'data'): host
    # RAM independent of the number of calibration tokens. False (default): every boundary activation of the draw is
    # stored, then decoded and solved (unchanged). True: the calibration batches are swept once per group of
    # calibration_streaming_chunk fitted blocks and only the sufficient statistics of the ridge (X^T X, X^T Y, sums, in
    # float64) are accumulated; the task vector is the same up to float summation order. map_rows='task' takes one
    # extra sweep for the Gram statistics of the frame (spectral_filter procrustes_cross / truncate_* only).
    calibration_streaming: bool = False
    calibration_streaming_chunk: int = 4

    def __post_init__(self) -> None:
        # Resolve the S0 defaults (see the synthesis comment); frozen dataclass, hence object.__setattr__.
        if self.synthesis is None:
            object.__setattr__(self, "synthesis", "ridge_preserve" if preserve_by_default(self) else "ridge")
        if self.rho_g is None:
            object.__setattr__(self, "rho_g", DEFAULT_RHO_G if self.synthesis == "ridge_preserve" else 0.0)

    def frame(self) -> dict[str, Any]:
        out = {name: getattr(self, name) for name in FRAME_FIELDS}
        if out["anchor_sampling"] == "random":
            del out["anchor_sampling"]  # the default leaves the frame of existing states unchanged
        if out["anchor_tokens_per_image"] is None:
            del out["anchor_tokens_per_image"]  # None leaves the frame of existing states unchanged
        if out["anchor_unit"] == "token":
            del out["anchor_unit"]
        if out["text_units"] == "token":
            del out["text_units"]
        return out

    def ridge_config(self) -> DirectResidualConfig:
        """The Ariadne config of the shared ridge: resident independent fit, or the sequential streaming fit."""
        return DirectResidualConfig(
            ridge_relative=self.ridge_relative,
            ridge_estimator=self.ridge_estimator,
            num_batches=self.num_batches,
            seed=self.seed,
            components=self.components,
            exact_form=self.exact_form,
            missing_bias=self.missing_bias,
            activation_storage="streaming" if self.depth_fit == "sequential" else "resident",
            # 'spread' / 'tracking' compile block by block in anchors itself; only the ridge parameters are read here
            depth_fit="independent" if self.depth_fit in ("spread", "tracking") else self.depth_fit,
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["components"] = list(self.components)
        if out["calibration_source"] == "task":
            del out["calibration_source"]  # the default leaves the recorded config of existing runs unchanged
        if out["calibration_images"] is None:
            del out["calibration_images"]
        if out["sequential_impl"] == "resident":
            del out["sequential_impl"]  # the default leaves the recorded config of existing runs unchanged
        if out["canonical_source"] == "rows":
            del out["canonical_source"]  # the default leaves the recorded config of existing runs unchanged
        if out["canonical_images"] is None:
            del out["canonical_images"]
        if not out["calibration_streaming"]:  # the default leaves the recorded config of existing runs unchanged
            del out["calibration_streaming"], out["calibration_streaming_chunk"]
        if out["canonical_soft_weights"] == "in_sample":
            del out["canonical_soft_weights"]  # the default leaves the recorded config of existing runs unchanged
        out["gpa_models"] = list(self.gpa_models)
        out["gpa_add_models"] = list(self.gpa_add_models)
        out["measure_depths"] = [float(x) for x in self.measure_depths]
        if not isinstance(self.refine_depths, str):
            out["refine_depths"] = [float(x) for x in self.refine_depths]
        out["operator_fusion_entries"] = list(self.operator_fusion_entries)
        out["operator_fusion_weights"] = [float(x) for x in self.operator_fusion_weights]
        return out


#: rho_g of the default synthesis (AGENT_SHARED_SPACE.md S0).
DEFAULT_RHO_G = 0.1
#: Anchor pools whose anchor data serve as the generic inputs G of ridge_preserve (Vision8's mix holds task images).
PRESERVE_POOLS = ("tiny_imagenet", "wikitext103")
#: Bounds of the S3 gain g_j (the least-squares gain is clipped into them; recorded in the diagnostics).
GAIN_BOUNDS = (0.0, 2.0)


def preserve_supported(cfg: AnchorsConfig) -> bool:
    """Whether ``synthesis='ridge_preserve'`` is implemented for ``cfg`` (decoders additionally refuse depth_fit
    'spread' in ``validate_text_config``: spread is vision-only)."""
    return (
        tuple(cfg.components) == ("mlp.c_proj",)
        and cfg.depth_fit in {"independent", "spread", "tracking", "sequential"}
        and cfg.anchor_pool in PRESERVE_POOLS
    )


def preserve_by_default(cfg: AnchorsConfig) -> bool:
    """The S0 default rule: ridge_preserve wherever it is implemented, except the bridge to Ariadne
    (``map_rows='task'``), which keeps Ariadne's plain ridge unless ridge_preserve is asked for explicitly
    (Ariadne + O2, the second Ariadne reference of S0, uses the frame's anchor data as G)."""
    # sequential keeps the plain ridge by default (S1 is opt-in: explicit ridge_preserve)
    return preserve_supported(cfg) and cfg.map_rows == "frame" and cfg.depth_fit != "sequential"


_ALLOWED = {f.name for f in fields(AnchorsConfig)}


def parse_anchors_config(value: Mapping[str, Any] | None) -> AnchorsConfig:
    """Parse ``method.params`` for ``anchors``; unknown keys are rejected."""
    payload = dict(value or {})
    unknown = sorted(set(payload) - _ALLOWED)
    if unknown:
        raise ValueError(f"unknown anchors config keys: {unknown}; allowed: {sorted(_ALLOWED)}")
    for key in ("gpa_models", "gpa_add_models", "operator_fusion_entries"):
        if isinstance(payload.get(key), (list, str)):
            value = payload[key]
            payload[key] = (value,) if isinstance(value, str) else tuple(value)
    if isinstance(payload.get("operator_fusion_weights"), (list, tuple)):
        payload["operator_fusion_weights"] = tuple(float(x) for x in payload["operator_fusion_weights"])
    if isinstance(payload.get("measure_depths"), (list, tuple)):
        payload["measure_depths"] = tuple(float(x) for x in payload["measure_depths"])
    if isinstance(payload.get("refine_depths"), (list, tuple)):
        payload["refine_depths"] = tuple(float(x) for x in payload["refine_depths"])
    if isinstance(payload.get("components"), (list, str)):
        comps = payload["components"]
        payload["components"] = (comps,) if isinstance(comps, str) else tuple(comps)
    cfg = AnchorsConfig(**payload)
    validate_anchors_config(cfg)
    return cfg


def validate_anchors_config(cfg: AnchorsConfig) -> None:
    if cfg.anchor_pool not in ANCHOR_POOLS:
        raise ValueError(f"anchor_pool must be one of {ANCHOR_POOLS}, got {cfg.anchor_pool!r}")
    for name in ("num_anchors", "anchor_batches", "registration_batches", "num_batches", "canonical_tokens"):
        if int(getattr(cfg, name)) < 1:
            raise ValueError(f"{name} must be >= 1")
    if cfg.anchor_positions not in {"all", "patch"}:
        raise ValueError("anchor_positions must be 'all' or 'patch'")
    if cfg.calibration_source not in {"task", "generic", "precomputed"}:
        raise ValueError(f"calibration_source must be 'task', 'generic' or 'precomputed', got {cfg.calibration_source!r}")
    if (cfg.calibration_source == "precomputed") != (cfg.calibration_images is not None):
        raise ValueError("calibration_images (a directory of <Task>.pt) is required by, and only allowed with, "
                         "calibration_source='precomputed'")
    if cfg.calibration_source != "task":
        if cfg.map_rows != "frame":
            raise ValueError(f"calibration_source={cfg.calibration_source!r} needs map_rows='frame' "
                             "(map_rows='task' reads task images)")
        if cfg.library != "data":
            raise ValueError(f"calibration_source={cfg.calibration_source!r} is not supported with library='operator'")
        if cfg.anchor_pool in TEXT_ANCHOR_POOLS:
            raise ValueError(f"calibration_source={cfg.calibration_source!r} is vision-only (text pools unsupported)")
    if cfg.calibration_source == "generic":
        if cfg.anchor_pool != "tiny_imagenet":
            raise ValueError("calibration_source='generic' needs the generic image pool anchor_pool='tiny_imagenet' "
                             "(vision8_mix holds task images; text pools are not supported)")
    if cfg.text_units not in TEXT_UNITS:
        raise ValueError(f"text_units must be one of {TEXT_UNITS}, got {cfg.text_units!r}")
    if cfg.text_units != "token" and cfg.anchor_pool not in TEXT_ANCHOR_POOLS:
        raise ValueError("text_units='word' applies to decoder (text) anchor pools only")
    if cfg.anchor_unit not in ANCHOR_UNITS:
        raise ValueError(f"anchor_unit must be one of {ANCHOR_UNITS}")
    if cfg.anchor_unit != "token" and (cfg.anchor_sampling != "random" or cfg.anchor_filter is not None):
        raise ValueError("centroid anchor_unit needs anchor_sampling='random' and no anchor_filter (token semantics)")
    if cfg.anchor_sampling not in ANCHOR_SAMPLINGS:
        raise ValueError(f"anchor_sampling must be one of {ANCHOR_SAMPLINGS}")
    if (cfg.anchor_sampling in K_SAMPLINGS) != (cfg.anchor_tokens_per_image is not None):
        raise ValueError("anchor_tokens_per_image is required by, and only allowed with, anchor_sampling 'k_per_image' / 'k_per_document'")
    if cfg.anchor_tokens_per_image is not None:
        k = cfg.anchor_tokens_per_image
        if isinstance(k, bool) or not isinstance(k, int) or k < 1 or cfg.num_anchors % k:
            raise ValueError("anchor_tokens_per_image must be a positive integer dividing num_anchors")
    if not cfg.eps_rel >= 0.0:
        raise ValueError("eps_rel must be >= 0")
    if cfg.ridge_estimator not in {"fixed_relative", "empirical_bayes", "none"}:
        raise ValueError("ridge_estimator must be 'fixed_relative', 'empirical_bayes' or 'none'")
    if not set(cfg.components) <= {"mlp.c_proj", "attn.out_proj"} or not cfg.components:
        raise ValueError("components must be a non-empty subset of {'mlp.c_proj', 'attn.out_proj'}")
    if cfg.compile_max_depth is not None and not 0.0 < float(cfg.compile_max_depth) <= 1.0:
        raise ValueError("compile_max_depth must be None or in (0, 1]")
    if cfg.spectral_filter not in {"tikhonov", "truncate_target", "truncate_cross", "procrustes_cross"}:
        raise ValueError(
            "spectral_filter must be 'tikhonov', 'truncate_target', 'truncate_cross' or 'procrustes_cross'"
        )
    if cfg.spectral_rank is not None:
        if cfg.spectral_filter == "tikhonov":
            raise ValueError("spectral_rank applies to the truncating spectral filters only")
        if isinstance(cfg.spectral_rank, bool) or not isinstance(cfg.spectral_rank, int) or cfg.spectral_rank < 1:
            raise ValueError("spectral_rank must be None or a positive integer")
    if cfg.kernel_decode not in KERNEL_DECODES:
        raise ValueError(f"kernel_decode must be one of {KERNEL_DECODES}")
    if cfg.kernel_decode != "tikhonov" and (cfg.frame_map != "pairwise" or cfg.spectral_filter != "tikhonov"):
        raise ValueError("kernel_decode 'zca'/'agreement'/canonical needs frame_map='pairwise' and spectral_filter='tikhonov'")
    if cfg.canonical_rank is not None and (
        cfg.kernel_decode != "canonical_hard"
        or isinstance(cfg.canonical_rank, bool) or not isinstance(cfg.canonical_rank, int) or cfg.canonical_rank < 1
    ):
        raise ValueError("canonical_rank must be a positive integer and is allowed with kernel_decode='canonical_hard' only")
    if cfg.canonical_null_threshold is not None and (
        cfg.kernel_decode != "canonical_soft" or isinstance(cfg.canonical_null_threshold, bool)
        or not 0.0 <= float(cfg.canonical_null_threshold) <= 1.0
    ):
        raise ValueError("canonical_null_threshold must lie in [0, 1] and is allowed with kernel_decode='canonical_soft' only")
    if cfg.canonical_soft_weights not in CANONICAL_SOFT_WEIGHTS:
        raise ValueError(f"canonical_soft_weights must be one of {CANONICAL_SOFT_WEIGHTS}, got {cfg.canonical_soft_weights!r}")
    if cfg.canonical_source not in ("rows", "sigma_all_tokens"):
        raise ValueError("canonical_source must be 'rows' or 'sigma_all_tokens'")
    if (cfg.canonical_source == "sigma_all_tokens") != (cfg.canonical_images is not None):
        raise ValueError("canonical_images is required by, and only allowed with, canonical_source='sigma_all_tokens'")
    if cfg.canonical_source == "sigma_all_tokens":
        if cfg.canonical_soft_weights != "heldout":
            raise ValueError("canonical_source='sigma_all_tokens' needs canonical_soft_weights='heldout'")
        if isinstance(cfg.canonical_images, bool) or not isinstance(cfg.canonical_images, int) or cfg.canonical_images < 2:
            raise ValueError("canonical_images must be an integer >= 2")
        if cfg.anchor_filter is not None or cfg.anchor_dedup_cos is not None or cfg.anchor_row_norm \
                or cfg.anchor_unit != "token" or cfg.map_rows != "frame":
            raise ValueError("canonical_source='sigma_all_tokens' needs token anchors without anchor_filter, "
                             "anchor_dedup_cos, anchor_row_norm and map_rows='frame'")
    if cfg.canonical_soft_weights == "heldout":
        if cfg.kernel_decode != "canonical_soft":
            raise ValueError("canonical_soft_weights='heldout' needs kernel_decode='canonical_soft'")
        if cfg.canonical_source == "rows" and cfg.anchor_sampling not in (*K_SAMPLINGS, "whole_images"):
            raise ValueError("canonical_soft_weights='heldout' needs anchor_sampling 'k_per_image' / 'k_per_document' / "
                             "'whole_images' "
                             "(the split is by image / document id, so every anchor row must carry one)")
        if cfg.anchor_dedup_cos is not None:
            raise ValueError("canonical_soft_weights='heldout' is not supported with anchor_dedup_cos (the dedup "
                             "drops rows after the id bookkeeping)")
    if (cfg.kernel_decode == "agreement") != (cfg.agreement_tau is not None):
        raise ValueError("agreement_tau is required by, and only allowed with, kernel_decode='agreement'")
    if cfg.agreement_tau is not None and not float(cfg.agreement_tau) >= 1.0:
        raise ValueError("agreement_tau must be >= 1")
    if cfg.synthesis not in {"ridge", "ridge_preserve"}:
        raise ValueError("synthesis must be 'ridge' or 'ridge_preserve'")
    if cfg.synthesis == "ridge_preserve" and not preserve_supported(cfg):
        raise ValueError(
            "synthesis='ridge_preserve' needs components=('mlp.c_proj',), depth_fit 'independent'/'spread'/'tracking'/'sequential' "
            f"and anchor_pool in {PRESERVE_POOLS}: set synthesis: ridge for this configuration"
        )
    if cfg.synthesis == "ridge" and float(cfg.rho_g) != 0.0:
        raise ValueError("rho_g applies to synthesis='ridge_preserve' only (this configuration resolves to 'ridge')")
    if cfg.update_rank is not None and (
        isinstance(cfg.update_rank, bool) or not isinstance(cfg.update_rank, int) or cfg.update_rank < 1
        or cfg.depth_fit == "sequential" or tuple(cfg.components) != ("mlp.c_proj",)
    ):
        raise ValueError("update_rank must be a positive integer, with depth_fit 'independent'/'spread'/'tracking', mlp.c_proj")
    if not float(cfg.rho_g) >= 0.0 or int(cfg.preserve_tokens) < 1:
        raise ValueError("rho_g must be >= 0 and preserve_tokens >= 1")
    if cfg.library not in {"data", "operator"}:
        raise ValueError("library must be 'data' or 'operator'")
    if cfg.target_calibration not in {"same", "disjoint"}:
        raise ValueError("target_calibration must be 'same' or 'disjoint'")
    if cfg.library == "operator" and (
        cfg.frame_map != "gpa" or cfg.depth_fit == "sequential" or cfg.map_rows != "frame"
    ):
        raise ValueError("library='operator' needs frame_map='gpa', depth_fit 'independent'/'spread', map_rows='frame'")
    if cfg.library == "data" and (
        cfg.operator_rank is not None or cfg.target_calibration != "same" or cfg.operator_library_dir is not None
    ):
        raise ValueError("operator_rank / target_calibration apply to library='operator' only")
    if cfg.operator_rank is not None and (
        isinstance(cfg.operator_rank, bool) or not isinstance(cfg.operator_rank, int) or cfg.operator_rank < 1
    ):
        raise ValueError("operator_rank must be None or a positive integer")
    if cfg.operator_space not in {"native", "common"} or cfg.operator_input_map not in {
        "gpa", "anchor_ridge", "paired", "pool_ridge"
    }:
        raise ValueError("operator_space must be native|common, operator_input_map gpa|anchor_ridge|paired|pool_ridge")
    if cfg.library != "operator" and (cfg.operator_space != "native" or cfg.operator_input_map != "gpa"):
        raise ValueError("operator_space / operator_input_map apply to library='operator' only")
    if cfg.operator_input_map == "paired" and cfg.target_calibration != "same":
        raise ValueError("operator_input_map='paired' needs target_calibration='same'")
    if cfg.operator_norm_filter and (cfg.library != "operator" or cfg.anchor_filter != "norm"):
        raise ValueError("operator_norm_filter needs library='operator' and anchor_filter='norm'")
    if cfg.operator_fusion not in {"sum", "tsv"} or cfg.operator_decode_rows not in {"task", "pool"}:
        raise ValueError("operator_fusion must be sum|tsv, operator_decode_rows task|pool")
    if (cfg.operator_fusion_entries or cfg.operator_fusion != "sum" or cfg.operator_fusion_weights
            or cfg.operator_decode_rows != "task") and (cfg.library != "operator" or not cfg.operator_fusion_entries):
        raise ValueError("operator_fusion* / operator_decode_rows need library='operator' and operator_fusion_entries")
    if cfg.operator_fusion_weights and len(cfg.operator_fusion_weights) != len(cfg.operator_fusion_entries):
        raise ValueError("operator_fusion_weights: one weight per entry")
    if cfg.operator_fusion_entries and (cfg.operator_cls_split or cfg.operator_library_dir is not None
                                        or cfg.depth_fit == "sequential"):
        raise ValueError("operator_fusion_entries: no cls split, no library_dir, no sequential fit")
    if cfg.operator_cls_split and (cfg.library != "operator" or cfg.operator_space != "common"):
        raise ValueError("operator_cls_split needs library='operator' and operator_space='common'")
    if int(cfg.operator_holdout_batches) < 1:
        raise ValueError("operator_holdout_batches must be >= 1")
    if cfg.anchor_dedup_cos is not None and not -1.0 < float(cfg.anchor_dedup_cos) <= 1.0:
        raise ValueError("anchor_dedup_cos must be None or in (-1, 1]")
    if cfg.anchor_dedup_max is not None:
        if cfg.anchor_dedup_cos is None:
            raise ValueError("anchor_dedup_max applies with anchor_dedup_cos only")
        if isinstance(cfg.anchor_dedup_max, bool) or not isinstance(cfg.anchor_dedup_max, int) or cfg.anchor_dedup_max < 1:
            raise ValueError("anchor_dedup_max must be None or a positive integer")
    if (cfg.anchor_row_norm or cfg.anchor_dedup_cos is not None) and cfg.map_rows != "frame":
        raise ValueError("anchor_row_norm / anchor_dedup_cos apply to the registered frame (map_rows='frame')")
    if cfg.anchor_filter not in {None, "norm"}:
        raise ValueError("anchor_filter must be None or 'norm'")
    if not float(cfg.anchor_filter_k) > 0.0:
        raise ValueError("anchor_filter_k must be > 0")
    if cfg.anchor_filter is not None and cfg.map_rows != "frame":
        raise ValueError("anchor_filter applies to the registered frame (map_rows='frame')")
    if cfg.frame_map not in {"pairwise", "gpa", "consensus"}:
        raise ValueError("frame_map must be 'pairwise', 'gpa' or 'consensus'")
    if cfg.consensus not in {"mean", "logeuclid", "robust"} or cfg.consensus_register not in {"rot", "correct", "zca_rot"}:
        raise ValueError("consensus must be mean|logeuclid|robust, consensus_register rot|correct|zca_rot")
    if isinstance(cfg.consensus_k, bool) or not isinstance(cfg.consensus_k, int) or cfg.consensus_k < 1 \
            or not float(cfg.consensus_eps) > 0.0:
        raise ValueError("consensus_k must be a positive integer and consensus_eps > 0")
    if cfg.frame_map != "consensus" and (
        cfg.consensus != "mean" or cfg.consensus_k != 512 or cfg.consensus_register != "rot" or cfg.consensus_eps != 1e-2
    ):
        raise ValueError("consensus / consensus_k / consensus_register / consensus_eps apply to frame_map='consensus' only")
    if cfg.frame_map == "consensus" and (
        cfg.map_rows != "frame" or cfg.spectral_filter != "tikhonov" or cfg.library != "data"
        or cfg.gpa_dim is not None or cfg.gpa_rank is not None
    ):
        raise ValueError("frame_map='consensus' needs map_rows='frame', spectral_filter='tikhonov', library='data' "
                         "and no gpa_dim / gpa_rank (the rank is consensus_k)")
    for name in ("gpa_dim", "gpa_rank"):
        value = getattr(cfg, name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{name} must be None or a positive integer")
    if cfg.frame_map == "pairwise" and (
        cfg.gpa_dim is not None or cfg.gpa_rank is not None or cfg.gpa_models or cfg.gpa_add_models
    ):
        raise ValueError("gpa_dim / gpa_rank / gpa_models / gpa_add_models apply to frame_map='gpa'|'consensus' only")
    if cfg.frame_map == "gpa" and cfg.map_rows != "frame":
        raise ValueError("frame_map='gpa' needs the registered frame (map_rows='frame')")
    if cfg.frame_map == "gpa" and cfg.spectral_filter != "tikhonov":
        raise ValueError("frame_map='gpa' replaces the decode; leave spectral_filter='tikhonov'")
    if cfg.map_rows not in {"frame", "task"}:
        raise ValueError("map_rows must be 'frame' or 'task'")
    for name in ("map_num_batches", "map_rows_per_image", "map_num_rows"):
        value = getattr(cfg, name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{name} must be None or a positive integer")
    if cfg.map_rows == "frame" and any(
        getattr(cfg, n) is not None for n in ("map_num_batches", "map_rows_per_image", "map_num_rows")
    ):
        raise ValueError("map_num_batches / map_rows_per_image / map_num_rows apply to map_rows='task' only")
    if cfg.map_rows == "task" and cfg.depth_fit != "independent":
        raise ValueError("map_rows='task' is implemented for depth_fit='independent' only")
    if cfg.depth_fit not in {"independent", "sequential", "spread", "tracking"}:
        raise ValueError("depth_fit must be 'independent', 'sequential', 'spread' or 'tracking'")
    if cfg.sequential_impl not in {"resident", "incremental"}:
        raise ValueError("sequential_impl must be 'resident' or 'incremental'")
    if cfg.sequential_impl == "incremental" and not (
        cfg.depth_fit == "tracking"
        or (cfg.depth_fit == "sequential" and (cfg.synthesis == "ridge_preserve" or cfg.gain_calibration != "none"))
    ):
        raise ValueError("sequential_impl='incremental' needs the native closed-loop fit: depth_fit='tracking', or "
                         "'sequential' with synthesis='ridge_preserve' or a gain_calibration")
    if cfg.preserve_inputs not in {"base", "updated"}:
        raise ValueError("preserve_inputs must be 'base' or 'updated'")
    if cfg.preserve_inputs == "updated" and (
        cfg.depth_fit not in {"sequential", "tracking"} or cfg.synthesis != "ridge_preserve"
    ):
        raise ValueError("preserve_inputs='updated' needs depth_fit 'sequential'/'tracking' and synthesis='ridge_preserve'")
    if cfg.gain_calibration not in {"none", "lsq"}:
        raise ValueError("gain_calibration must be 'none' or 'lsq'")
    if isinstance(cfg.gain_holdout_batches, bool) or not isinstance(cfg.gain_holdout_batches, int) \
            or cfg.gain_holdout_batches < 0 \
            or (cfg.gain_calibration != "none" and cfg.gain_holdout_batches >= cfg.num_batches):
        raise ValueError("gain_holdout_batches must be an integer in [0, num_batches)")
    if cfg.gain_calibration != "none" and (
        cfg.depth_fit == "spread" or tuple(cfg.components) != ("mlp.c_proj",) or cfg.map_rows != "frame"
    ):
        raise ValueError("gain_calibration needs mlp.c_proj, map_rows='frame' and depth_fit 'independent'/'sequential'/'tracking'")
    if (cfg.depth_fit == "spread") != bool(cfg.measure_depths):
        raise ValueError("measure_depths is required by, and only allowed with, depth_fit='spread'")
    if cfg.depth_fit == "spread" and (
        tuple(cfg.components) != ("mlp.c_proj",) or cfg.map_rows != "frame" or cfg.compile_max_depth is not None
        or not all(0.0 <= float(s) <= 1.0 for s in cfg.measure_depths)
    ):
        raise ValueError("depth_fit='spread' needs mlp.c_proj, map_rows='frame', no compile_max_depth, depths in [0, 1]")
    if cfg.depth_fit in {"sequential", "tracking"} and len(cfg.components) != 1:
        raise ValueError("depth_fit='sequential' requires exactly one component")
    if cfg.refine not in {"none", "joint"}:
        raise ValueError("refine must be 'none' or 'joint'")
    if cfg.refine == "joint":
        if cfg.refine_target not in {"all_depths", "final"}:
            raise ValueError("refine_target must be 'all_depths' or 'final'")
        if isinstance(cfg.refine_depths, str):
            if cfg.refine_depths != "all":
                raise ValueError("refine_depths must be 'all' or a non-empty list of relative depths in [0, 1]")
        elif not cfg.refine_depths or not all(0.0 <= float(s) <= 1.0 for s in cfg.refine_depths):
            raise ValueError("refine_depths must be 'all' or a non-empty list of relative depths in [0, 1]")
        for name in ("refine_steps", "refine_batch", "refine_eval_every", "refine_generic_batches"):
            v = getattr(cfg, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(cfg.refine_holdout_batches, bool) or not isinstance(cfg.refine_holdout_batches, int) \
                or not 1 <= cfg.refine_holdout_batches < cfg.num_batches:
            raise ValueError("refine_holdout_batches must be an integer in [1, num_batches)")
        if not float(cfg.refine_lr) > 0.0 or float(cfg.refine_rho) < 0.0 or float(cfg.refine_prox) < 0.0:
            raise ValueError("refine_lr must be > 0, refine_rho and refine_prox >= 0")
        if float(cfg.refine_rho) > 0.0 and cfg.anchor_pool not in PRESERVE_POOLS:
            raise ValueError("refine_rho > 0 needs the generic anchor pool (anchor_pool 'tiny_imagenet')")
        if not 0.0 <= float(cfg.refine_warmup) < 1.0:
            raise ValueError("refine_warmup must lie in [0, 1)")
        if cfg.update_rank is not None or cfg.map_rows != "frame":
            raise ValueError("refine='joint' is not implemented with update_rank or map_rows='task'")
        if cfg.depth_fit == "spread" and max(float(x) for x in cfg.measure_depths) < 1.0:
            raise ValueError("refine='joint' with depth_fit='spread' needs the last measure depth to be 1.0 "
                             "(blocks after it would be refined from a zero update)")
        if (tuple(cfg.components) != ("mlp.c_proj",) or cfg.compile_max_depth is not None
                or (cfg.depth_fit == "sequential" and cfg.synthesis != "ridge_preserve" and cfg.gain_calibration == "none")):
            raise ValueError("refine='joint' needs mlp.c_proj, no compile_max_depth, and a depth_fit that keeps the "
                             "calibration banks resident (independent, tracking, or sequential with ridge_preserve / gain)")
    if cfg.calibration_streaming:
        validate_calibration_streaming(cfg)
    if cfg.depth_pairing != "relative":
        raise ValueError("anchors supports depth_pairing='relative' only (pi_m(s) = round(s (L_m - 1)))")


def validate_calibration_streaming(cfg: AnchorsConfig) -> None:
    """``calibration_streaming``: only the combinations whose statistics are accumulated exactly as the resident fit
    would (anything else is refused, never approximated). Decoders only: the run-time check is in ``prepare``."""
    if isinstance(cfg.calibration_streaming_chunk, bool) or int(cfg.calibration_streaming_chunk) < 1:
        raise ValueError("calibration_streaming_chunk must be an integer >= 1")
    if cfg.depth_fit != "independent":
        raise ValueError("calibration_streaming needs depth_fit='independent' (sequential / tracking / spread: not streamed)")
    for name, required in (("library", "data"), ("refine", "none"), ("gain_calibration", "none"),
                           ("update_rank", None), ("calibration_source", "task"), ("text_units", "token")):
        if getattr(cfg, name) != required:
            raise ValueError(f"calibration_streaming does not support {name}={getattr(cfg, name)!r} (must be {required!r})")
    if cfg.operator_fusion_entries:
        raise ValueError("calibration_streaming does not support operator_fusion_entries")
    if cfg.map_rows == "task":
        if cfg.frame_map != "pairwise" or cfg.spectral_filter not in TASK_STREAMING_FILTERS:
            raise ValueError(
                f"calibration_streaming with map_rows='task' needs frame_map='pairwise' and spectral_filter in "
                f"{TASK_STREAMING_FILTERS} (the frame is kept as Gram statistics); got frame_map={cfg.frame_map!r}, "
                f"spectral_filter={cfg.spectral_filter!r}"
            )
        if cfg.map_rows_per_image is not None:
            raise ValueError("calibration_streaming: map_rows_per_image is a vision option")


#: Spectral filters of the task-row frame that need only the Gram / cross-Gram of the rows.
TASK_STREAMING_FILTERS = ("procrustes_cross", "truncate_cross", "truncate_target")

#: Defaults of a decoder (LLM) run for absent keys; the dataclass defaults (the vision ones) are unchanged.
DECODER_DEFAULTS: dict[str, Any] = {"anchor_pool": "wikitext103", "missing_bias": "materialize"}

#: (field, required value) of the options defined on image token grids only, or not ported to decoders yet.
_TEXT_REQUIRED: tuple[tuple[str, Any], ...] = (
    ("anchor_unit", "token"),
    ("anchor_positions", "all"),
    ("library", "data"),
    ("update_rank", None),
    ("gain_calibration", "none"),
    ("refine", "none"),
    ("preserve_inputs", "base"),
    ("sequential_impl", "resident"),
    ("map_rows_per_image", None),
    ("components", ("mlp.c_proj",)),
    ("calibration_source", "task"),
    ("calibration_images", None),
)


def validate_text_config(cfg: AnchorsConfig) -> None:
    """Decoder (text) guard. Rows are the content tokens of the pool sequences (BOS and padding excluded), so the
    grid / CLS / centroid options do not apply; the operator library, update_rank (D5) and the spread compile are
    vision-only for now. synthesis 'ridge_preserve' (D2, the default) runs with depth_fit 'independent' and either
    text_units; with depth_fit 'sequential' the default resolves to 'ridge'. depth_fit 'sequential' (Ariadne's
    sequential streaming fit, the anchor decode as desired_fn) needs token rows: source and target must read the same
    token sequence."""
    if cfg.anchor_sampling not in {"random", "k_per_document"}:
        raise ValueError(
            f"anchors on decoder models does not support anchor_sampling={cfg.anchor_sampling!r} "
            "(must be 'random' or 'k_per_document')"
        )
    if cfg.canonical_source != "rows":
        raise ValueError("canonical_source='sigma_all_tokens' is vision-only for now (decoders: not wired)")
    for name, required in _TEXT_REQUIRED:
        got = getattr(cfg, name)
        if got != required:
            raise ValueError(f"anchors on decoder models does not support {name}={got!r} (must be {required!r})")
    if cfg.depth_fit not in {"independent", "sequential"}:
        raise ValueError(f"anchors on decoder models supports depth_fit 'independent' or 'sequential': {cfg.depth_fit!r}")
    if cfg.depth_fit != "independent" and cfg.synthesis == "ridge_preserve":
        raise ValueError("ridge_preserve on decoder models needs depth_fit='independent' (S1 sequential is vision-only): "
                         "set synthesis: ridge")
    if cfg.depth_fit == "sequential" and cfg.text_units != "token":
        raise ValueError("anchors depth_fit='sequential' on decoder models needs text_units='token' (same tokenizer)")
    if cfg.anchor_pool not in TEXT_ANCHOR_POOLS:
        raise ValueError(f"anchors on decoder models needs a text anchor_pool {TEXT_ANCHOR_POOLS}: {cfg.anchor_pool!r}")


def resolve_anchors_decoder_config(value: Mapping[str, Any] | None, family_adapter: Any) -> AnchorsConfig:
    """Parse ``method.params`` of an LLM ``anchors`` run: decoder defaults for absent keys, vision-only options refused.

    ``family_adapter`` must be given. Families that normalise the MLP output before the residual add (Gemma-2) need
    Ariadne's ``post_mlp_norm`` in the shared ridge, not wired for anchors yet.
    """
    if family_adapter is None:
        raise ValueError("resolve_anchors_decoder_config requires the model family adapter")
    if bool(getattr(family_adapter, "HAS_POST_MLP_NORM", False)):
        raise NotImplementedError(
            f"anchors on {getattr(family_adapter, 'name', 'this family')} (post-MLP norm) is not supported yet"
        )
    cfg = parse_anchors_config({**DECODER_DEFAULTS, **dict(value or {})})
    validate_text_config(cfg)
    return cfg
