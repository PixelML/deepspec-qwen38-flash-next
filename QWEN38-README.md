# DFlash / DSpark for Qwen3.8-Flash-Next

DeepSpec fork carrying the Qwen3.8-Flash-Next (`qwen4_exp`) port: a vLLM
feature exporter, the DFlash/DSpark configs, the `qwen4_exp` draft-config
package, and the trainer/parser patches needed to train against this target.

Full narrative (every decision, correction, dead end, and measured number)
lives in `docs/dflash-training-log.md` (not included in this repo -- see
"What's not here" below). This file is the load-bearing summary: enough to
reproduce or extend the work without re-deriving it.

Tracking issue: `seanphan/pixelml#118`.

## Tap definition

Target: `Qwen/Qwen3.8-Flash-Next` (`qwen4_exp`, `hidden_size=2560`,
`num_hidden_layers=48`, HyperConnection width `hc_count * hidden_size =
4*2560 = 10240`). Every 4th layer (`full_attention_interval=4`) is a QSA
full-attention layer; the other 36 are GatedDeltaNet linear-attention.

**5 tapped layers, 0-indexed: `[3, 15, 23, 35, 43]`** (config:
`config/dflash/dflash_qwen38_flash_next.py::target_layer_ids`). Derivation:
DeepSpec's rule is uniform spacing from the second full-attention layer
through the third-to-last, applied to this target's 12 full-attention
ordinals `[3,7,11,15,19,23,27,31,35,39,43,47]`. (An earlier by-eye pick,
`[7,15,23,27,35]`, was wrong -- not evenly spread, and didn't apply the rule
correctly; corrected 2026-09-06.)

**What's actually captured at each tap** is *not* the raw HC-wide (10240)
residual -- it's the **HC-contracted, native-width (2560) residual**, taken
from the tapped layer's own `GatedResidual` mix, i.e. "the input to layer
`L+1`'s hyper-connection combine" == "the fully combined, contracted state
after layer `L`" in exactly the sense DeepSpec's own
`extract_context_feature` convention (`hidden_states[layer_id + 1]`) already
uses. This reuses the model's own learned HC-contraction weights rather than
introducing an new, untrained contraction module.

- HF reference: `Qwen4ExpTextGatedResidual.forward(..., use_combine=True)`
  returns `(mixed_input[2560], hyper_input, injection_weights)` -- the tap is
  return value **`[0]`**.
- vLLM: `.../qwen4_exp/nvidia/hyperconnection.py::GatedResidual.mix()` /
  `.combine_and_mix()` return `(hidden_states[10240], block_input[2560],
  injection)` -- the tap is return value **`[1]`**. Same quantity (grouped
  GemmaRMSNorm over the HC streams, low-rank silu/sigmoid gate, gated mean
  over `hc_count`); the NVIDIA variant just defers the combine to the next
  mix boundary.

Plus a 6th tensor: `target_last_hidden_states`, the **final-norm last
hidden** -- the model's own final contracted output
(`hyper_connection_mixer.combine_and_mix(...)[1]`, same quantity as HF's
`last_hidden_state`; there is no separate final norm after it in either
implementation).

Concatenated widths: `target_hidden_states` = `5 * 2560 = 12800`,
`target_last_hidden_states` = `2560`.

## How the exporter hooks vLLM (`scripts/data/export_features_vllm.py`)

Three structural traps found by running, in order:

1. **vLLM's `GatedResidual` has no `forward()`** -- the decoder layer calls
   `.mix()` / `.combine_and_mix()` directly, so `register_forward_hook` would
   never fire. Fixed by patching those two methods at class level and
   dispatching per-instance via a `_ds_tap` tag attached only to the tapped
   modules.
2. The registered top-level module is the VLM wrapper
   `Qwen4ExpForConditionalGeneration`, not `Qwen4ExpForCausalLM` -- the
   causal LM (and `.lm_head`) live at `.language_model`.
3. **`Qwen4ExpModel` carries `@support_torch_compile`, which installs its own
   `__call__` on the class and therefore never runs `nn.Module._call_impl` --
   forward hooks registered on it silently never fire.** This is the sharp
   edge: it manifests as a *clean, green run* -- "20/20 prompts processed at
   15,640 tok/s, 0 captures" -- not an exception. **A non-zero capture count
   is a mandatory assertion**, not an optional sanity check; the exporter
   raises `RuntimeError(f"taps not captured this forward: {missing}")` on
   every batch for exactly this reason. The fix was moving both hooks
   (`register_forward_pre_hook` / `register_forward_hook`, `with_kwargs=True`)
   onto the undecorated top-level wrapper, which receives the same
   `input_ids` / `positions` / `query_start_loc` the decorated inner module
   would have.

Batch mechanics: vLLM V1 flattens the whole scheduled batch to
`[num_tokens, ...]`. Per-request spans come from `query_start_loc`
(best-effort cross-checked against `positions == 0`, which was observed
degenerate/all-zero in this configuration and so is not authoritative); each
span is then *identified* by its exact token-id tuple against the pending
request table, and a span matching no pending request is a hard error rather
than a silent corruption. The run is prefill-only (`max_tokens=1`, no prefix
caching, no chunked prefill), so every request is prefilled exactly once,
whole, and `enforce_eager=True` is required both to avoid a 1-card
flashinfer-autotune memory blow-up and because `torch.compile` would bypass
the python-level `GatedResidual` patch the exporter depends on.

## Why vLLM instead of the HF `transformers` path

`scripts/data/prepare_target_cache_qwen38_flash_next.py` (the original HF
port) measured **~38 tok/s on 4x B200**: `Qwen4ExpTextQSAIndexer` is an eager
nested-Python loop, ~95% of forward time (one `Attention` module instance
took 3,148ms of a 3,976ms/batch forward; 12 QSA layers per forward). Not
viable for a 100k-row cache. vLLM ships a fused CUDA QSA indexer and --
decisively -- is the *exact* engine that generated the training corpus
(`nvidia/Qwen3.8-Flash-Next-NVFP4`), so its hidden states are what a drafter
deployed against that engine will actually see at serve time.

## Drafter architecture

`Qwen3DSparkModel` (DeepSpec's existing dense Qwen3-style draft
architecture, reused unchanged) with a real `Qwen3Config` built by
`deepspec/modeling/dspark/qwen4_exp/config.py::build_draft_config` -- **not**
a deepcopied `Qwen4ExpTextConfig` (that keeps Qwen4Exp's strict
`layer_types` validator, which rejects the plain `"full_attention"` the
draft needs and breaks `save_pretrained` at the first checkpoint; also
carries mRoPE-only `rope_parameters` fields that mean nothing to a dense
draft). Fields actually copied from the target: `vocab_size`, `hidden_size`,
`num_attention_heads`, `num_key_value_heads`, `head_dim`, `hidden_act`,
`max_position_embeddings`, `initializer_range`, `rms_norm_eps`,
`attention_bias`, `attention_dropout`, plus a clean `rope_theta`-only rope
spec. `intermediate_size = 3 * hidden_size = 7680` (no principled 1:1 mapping
from the target's MoE expert width exists, so this follows Qwen3-8B's own
dense hidden:intermediate ratio, `1:3`).

| | 5-layer draft, `block_size=7` |
|---|---|
| **DFlash** | 5 decoder layers, block 7, **498.1M** params (excl. stripped `embed_tokens`/`lm_head`) |
| **DSpark** | same backbone + `markov_head` (127.1M) + `confidence_head` (2.8K), **625.2M** params (excl. stripped `embed_tokens`/`lm_head`) |

(`draft_params_excl_embed_head` measured 625,249,537 for DSpark via a
build -> save -> reload round trip, `config/dspark/dspark_qwen38_flash_next.py`;
DFlash is the same backbone minus the markov/confidence heads.)

## How `embed_tokens`/`lm_head` bind at load time

**The published checkpoint does not contain `embed_tokens.weight` or
`lm_head.weight`.** Both are explicitly stripped before push (`push_hf` in
the launcher, not part of this repo -- see "What's not here"): shipping
`248320 x 2560` target-derived rows in a "draft layers + `W_c`" artifact
would be wasteful and a re-publication of the target's own weights.

At training/load time, `base_trainer.py`'s `build_models()` initializes the
draft's embedding and (if untied) head **directly from the target model's**
`embed_tokens.weight` / `lm_head.weight` -- copied once, then frozen/never
trained (the draft only ever learns the 5 transformer layers, the fusion
`fc` (`[2560, 12800]`, the concat-of-5-taps -> draft-hidden-size projection),
and `hidden_norm`). A consumer loading this checkpoint against the target
therefore **must** perform the same bind: load `embed_tokens.weight` and
`lm_head.weight` straight from `nvidia/Qwen3.8-Flash-Next-NVFP4` (or the
matching-vocab base `Qwen/Qwen3.8-Flash-Next`) into the draft model before
use -- the checkpoint alone is not runnable.

## Cache format

Per token, bf16 (2 bytes/elem): `(5 * 2560 + 2560) * 2 = 30,720 bytes/token`
-- 5 taps concatenated (12,800) + last hidden (2,560), same formula as the
original Qwen3-8B protocol, re-derived for `hidden_size=2560`. Measured at
scale: 120,776,745 tokens over 98,470 samples = **3.711 TB** for the full
99,457-row training cache (matched a 3.72 TB projection from a 5k-row
diagnostic to within 0.3%).

## Measured numbers

- **Exporter throughput: 31,109 tok/s on 1x B200** (S2 diagnostic cache,
  5,000 corpus rows, 6,020,936 tokens, 681s) -- ~310x the HF path's
  measured-at-scale 38 tok/s on 4x B200, and the number that made the
  100k-row full cache economically viable (~$18.45 actual for the full
  3.711 TB cache at 13,715 tok/s once warmed up).
- **S1 fidelity gate** (vLLM export vs. HF reference, 20 prompts, 9,194-9,214
  compared positions): per-tap mean cosine 0.9965 (L3) down to 0.9761 (L43,
  monotonic decay with depth -- the NVFP4-vs-BF16 quantization signature, not
  a wiring defect). `lm_head(last_hidden)` argmax reproduces the *engine's
  own* top-1 (internal to vLLM, cannot be affected by the HF-vs-vLLM
  comparison) at **0.99695** of positions. The worst per-tap cosine (0.9761)
  came in under the contract's `>= 0.98` PASS bar but above its `< 0.95` STOP
  bar; proceeding past that was a labelled, coordinator-accepted deviation
  (full reasoning in `docs/dflash-training-log.md`).
- **DFlash training, `block_size=7`, 5 draft layers, 10 epochs, full 99,457-row
  cache:** pos-1 top-1 agreement and expected-accepted-length `tau` rose every
  epoch, best at epoch 7 (step 1344): **tau = 3.079452** (epochs 8-10 bought
  only ~+0.01 tau over epoch 7 and were not worth the extra checkpoint eval
  cost as the pick).
- A 3-layer / 1-epoch diagnostic at equal training measured pos-1 = 0.0482
  against the 5-layer architecture's pos-1 = 0.3010 at the same training
  budget -- a 6x gap from **depth**, not epochs. Confirms a 3-layer draft is
  below the useful threshold for this target rather than a scaled-down proxy
  for the 5-layer one; do not use the 3-layer diagnostic depth as a cheap
  stand-in for the real protocol.

## What's not here

This repo carries the **source**: the exporter, eval scripts, configs, the
`qwen4_exp` draft-config package, and the trainer/parser patches. It does
**not** carry: `docs/dflash-training-log.md` (the full phase-by-phase
narrative, kept in the private ops clone), the Modal launcher
(`modal_dflash_p3.py`, same reason), trained checkpoints (pushed separately
to the private HF repo `PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash`), or the
regen corpus / target-hidden-state caches (live on the `qwen38-drafter` Modal
volume, `/vol/dflash_p3/...`).

## Key files

- `scripts/data/export_features_vllm.py` -- the vLLM feature exporter (this
  document's main subject).
- `scripts/data/export_features_hf_gate.py` -- HF-path reference dump for the
  S1 fidelity gate.
- `scripts/eval/compare_features.py` -- S1 gate: per-tap cosine + lm_head
  argmax comparison.
- `scripts/eval/agreement_curve.py` -- per-checkpoint held-out top-k
  agreement curve + `tau` estimate.
- `scripts/eval/check_cache_alignment.py` -- `argmax(lm_head @
  cached_last_hidden[i]) == input_ids[i+1]` alignment diagnostic (used to
  rule out a wiring bug when a 3-layer diagnostic underperformed).
- `config/dflash/dflash_qwen38_flash_next.py`, `config/dspark/dspark_qwen38_flash_next.py`.
- `deepspec/modeling/dspark/qwen4_exp/` -- `build_draft_config` +
  `get_qwen4_exp_text_config`, re-exports `Qwen3DSparkModel`.
- `deepspec/trainer/base_trainer.py`, `deepspec/trainer/dspark_trainer.py`,
  `deepspec/data/parser.py` -- `target_model_revision` threading,
  `Qwen4ExpDSparkTrainer`, the `"qwen38_flash_next"` chat template
  (`enable_thinking=False`, no `assistant_loss_prefix`).
