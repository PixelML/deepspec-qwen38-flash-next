# Serving a DeepSpec DFlash drafter for Qwen3.8-Flash-Next on vLLM

This directory carries everything the engine side needs. Stock vLLM cannot serve a
DeepSpec-trained DFlash drafter against `nvidia/Qwen3.8-Flash-Next-NVFP4`. It is not the
checkpoint's fault — field for field the checkpoint is a near-exact fit for vLLM's DFlash
loader. Every blocker is on the target side or in the runner.

Two things are needed, and they are not the same thing:

1. **Three control-flow patches** to vLLM. These make the engine stop refusing.
2. **A six-file adapter overlay.** This makes the target *emit contracted auxiliary hidden
   states at all* and teaches the proposer DeepSpec's anchor convention. **The patches alone
   serve nothing.** Anyone reproducing this needs the whole set.

Everything here is published as a diff against a pinned vLLM, not as a copy of vLLM source.
Verify with the `sha256` pairs in `adapter/overlay-manifest.json`.

## Pins

| | |
| --- | --- |
| Target | `nvidia/Qwen3.8-Flash-Next-NVFP4@fc694b54fb0174e0913e6adf86691ef85a4ead47` |
| Drafter | `PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash` |
| vLLM | source `e962733e08d10f7ca65dac4df99e116460b8b174`, image `vllm/vllm-openai@sha256:89dd8f442a3f4c08c6b3cd634c4f735cd709160651c296596673cf974ea6ee39` (arm64) |
| Hardware measured on | 2× NVIDIA DGX Spark (GB10, sm_121a, arm64), TP2 + expert parallel, RoCE |

Line numbers below are that vLLM commit. On another commit, apply by hand.

## The three patches — and the failure each one fixes

### `patches/01-model-runner-none-guard.diff`

Fixes:

```
TypeError: 'NoneType' object is not subscriptable
  vllm/v1/worker/gpu/model_runner.py:834
```

The V2 runner overrides the drafter's input hidden states with the target's pre-HC MTP
residual whenever the target exposes `get_mtp_target_hidden_states()`. `Qwen4Exp` **exposes
the accessor but returns `None`** when no native MTP head is loaded (`method=dflash`), and
the subscript is unguarded. The patch takes the engine's own absent-attribute fallback
(`spec_hidden_states = hidden_states`) instead. Two sites, `:832-834` and `:2016-2018`.

There is no V1 fallback. Under `VLLM_USE_V2_MODEL_RUNNER=0` this target does not boot **with
or without a drafter**:

```
RuntimeError: PLE inputs were not prepared
  vllm/models/qwen4_exp/nvidia/model.py:300   (raised during profile_run)
```

`query_start_loc` and `ngram_context` are produced only by
`Qwen4ExpModelState.prepare_inputs` / `prepare_dummy_inputs`
(`vllm/models/qwen4_exp/nvidia/model_state.py:93,112`), and those are called only from the V2
runner. Use V2.

### `patches/02-dflash-speculator-anchor-layout.diff`

Fixes:

```
ValueError: sample_from_anchor=True is not supported for DFlash. DFlash uses a
fixed 1+N query layout where the anchor is the bonus token.
  vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:69
```

DeepSpec trains **anchor-as-first-prediction**: K query slots, every position predicts.
vLLM's DFlash path assumes the speculators-format **1+N** layout, where position 0 is the
anchor and its output is discarded. `DSparkSpeculator` already selects our layout natively,
and the shared Triton `_prepare_dflash_inputs_kernel` already implements it behind a
`SAMPLE_FROM_ANCHOR` constexpr. The raise is the only thing in the way. The patch honours
`dflash_config.sample_from_anchor` and sets `num_query_per_req = num_speculative_steps`.

Credit where it is due: vLLM raises here. SGLang has the same semantic gap and instead
shifts every draft position by one silently, presenting only as a bad drafter.

**Not needed for DSpark.** `DSparkSpeculator` sets both attributes itself after
`super().__init__`, so leaving `sample_from_anchor` out of `dflash_config` avoids the stock
raise entirely.

### `patches/03-qwen4exp-spec-method-allowlist.diff`

Fixes:

```
NotImplementedError: Qwen4Exp speculative decoding supports only its native MTP
checkpoint and linear n-gram proposers
  vllm/model_executor/models/config.py
  Qwen4ExpForConditionalGenerationConfig.verify_and_update_config
```

`EngineArgs.create_engine_config` rejects the method **before weights load**. Upstream the
allowlist is `{mtp, ngram, ngram_gpu}`. The patch adds `dflash` and `dspark`.

The `dflash` line is also carried inside `adapter/adapter-overlay.patch`, because the overlay
touches the same file. If you apply the overlay, this patch adds only the `dspark` line.

## The six-file adapter overlay — `adapter/adapter-overlay.patch`

| file | what it adds |
| --- | --- |
| `vllm/models/qwen4_exp/nvidia/model.py` | captures `block_input` (the native-width 2560 contracted residual, tuple slot **[1]**) at boundary layers; exposes `set_dflash_aux_hidden_state_layers`; declares `supports_eagle3` / `has_own_embed_tokens=False` / `has_own_lm_head=False`; returns `(sample_hidden_states, aux_hidden_states)` and raises `RuntimeError("Missing epoch7 HC features in target forward")` if fewer than 5 are collected |
| `vllm/v1/spec_decode/dflash.py` | adds `query_zero_predicts_next`, making `num_query_per_req = K` (DeepSpec) instead of `K + 1` (legacy) |
| `vllm/v1/spec_decode/utils.py` | threads `QUERY_ZERO_PREDICTS_NEXT` into the Triton kernel: `sample_offset = 0` instead of `1`, so query 0's output is sampled rather than discarded |
| `vllm/v1/spec_decode/llm_base_proposer.py` | registers `Qwen4ExpForConditionalGeneration` in the target-sharing allowlist so the draft aliases the target's embedding and untied `lm_head` |
| `vllm/config/speculative.py` | returns `num_draft_tokens - 1` additional scheduler slots when `query_zero_predicts_next` is set |
| `vllm/model_executor/models/config.py` | the `dflash` allowlist entry |

**The vLLM aux boundary ids are the trained tap ids + 1.** Trained `target_layer_ids =
[3, 15, 23, 35, 43]` are captured at `[4, 16, 24, 36, 44]`, each at
`layers[i].attn_hyper_connection.mix/combine_and_mix(...)[1]`. There is no separate final
norm to add, and the learned HC contraction must never be replaced by a raw HC-stream mean.
Concatenation is `[T, 5 × 2560] = [T, 12800]`, matching the drafter's `fc` in_features
exactly. The last hidden state is the distillation target, **not** a sixth conditioning tap.

## Also mounted, and not redistributed here

The measured arms mounted two further files that are pinned-image workarounds rather than
drafter-related changes. They are vLLM-derived and we do not have a clean diff against the
image's own copies, so they are described rather than shipped:

- `vllm/models/qwen4_exp/common/qsa_cache.py` — widens the QSA raw-key rollback ring to a
  whole-group size that divides the attention block size (`qsa_ring_capacity`). Required for
  `num_speculative_tokens >= 5`; equivalent to upstream vLLM PR #54912. Without it, K=5 and
  K=7 fail a ring assertion.
- `vllm/model_executor/layers/quantization/modelopt.py` — block-FP8 weight loading. Needed by
  the **native MTP baseline arm only**, not by DFlash, which loads no MTP head. Equivalent to
  upstream vLLM PR #55513.

## The draft config override — `adapter/draft-config.json`

The published checkpoint's own `config.json` does not carry the `dflash_config` block the
engine needs. Mount `adapter/draft-config.json` over `/draft/config.json`. It adds:

```json
"dflash_config": {
  "target_layer_ids": [3, 15, 23, 35, 43],
  "mask_token_id": 248077,
  "use_aux_hidden_state": true,
  "causal": false,
  "query_zero_predicts_next": true
}
```

plus `is_neox_style` and `target_hidden_size`.

## The plugin — `plugin/`

`plugin/dflash_plugin.py` is a vLLM general-plugin entry point that registers
`DFlashQwen3DSparkModel`; `plugin/dflash_epoch7.py` is the model class, a thin subclass of
vLLM's own `DFlashQwen3ForCausalLM` that adds strict checkpoint validation and declares that
the draft owns neither the embedding nor the LM head. Install it so
`[vllm.general_plugins] pixelml_epoch7 = dflash_plugin:register` is discoverable, or just put
the directory on `PYTHONPATH` and register manually.

Note the class asserts TP2, PP1, eager mode and BF16 draft compute. Those are the conditions
this drafter was measured under, not proven requirements — relax them deliberately, and
re-measure if you do.

## Apply and launch

```bash
VLLM=/usr/local/lib/python3.12/dist-packages/vllm   # inside the image

# 1. the adapter overlay (6 files)
patch -p1 -d "$VLLM/.." < adapter/adapter-overlay.patch

# 2. the three control-flow patches
patch -p1 -d "$VLLM/.." < patches/01-model-runner-none-guard.diff
patch -p1 -d "$VLLM/.." < patches/02-dflash-speculator-anchor-layout.diff
patch -p1 -d "$VLLM/.." < patches/03-qwen4exp-spec-method-allowlist.diff   # dspark line only, if the overlay is applied

# 3. the QSA ring fix, for K >= 5 (see "Also mounted" above)
```

Then serve. This is the exact configuration behind every number in the model card, with local
addresses and interface names replaced by placeholders:

```bash
vllm serve /model \
  --served-model-name qwen3.8-flash-next \
  --trust-remote-code \
  --quantization modelopt \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --nnodes 2 --node-rank 0 \
  --master-addr <RANK0_ADDR> --master-port 25100 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 8192 \
  --max-num-seqs 16 \
  --enforce-eager \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --speculative-config '{"method":"dflash","model":"/draft","num_speculative_tokens":5,"draft_tensor_parallel_size":2,"quantization":null,"attention_backend":"FLASH_ATTN"}'
```

with `VLLM_USE_V2_MODEL_RUNNER=1`, `PYTHONPATH=/epoch7` (the plugin directory), `/model` the
target, `/draft` the drafter, and `adapter/draft-config.json` mounted at
`/draft/config.json`. Rank 1 adds `--headless`. NCCL/RoCE environment is site-specific.

`num_speculative_tokens` 4, 5 and 7 were all measured. **5 was fastest overall**, 7 gave the
highest accepted length and was slowest. K is the proposal count *and* the number of draft
query positions, so served K = 5 means up to 6 emitted tokens per fully accepted iteration.
K = 4 and 5 execute all five trained decoder layers with shorter bidirectional query blocks;
they are not shallower models and not a truncation of a block-7 result, so each K is a
separate configuration to benchmark, never an extrapolation.

## What is not established

Every number this configuration produced is **eager mode**, single boot per arm, c=1, greedy,
non-thinking, 256-token outputs, 8k context. Graph-mode behaviour is untested. Concurrency
and soak were never run. Losslessness is **not** certified. See the model card, which states
all of this at length.

## Upstream

- SGLang: [sgl-project/sglang#38589](https://github.com/sgl-project/sglang/issues/38589)
- vLLM: [vllm-project/vllm#56088](https://github.com/vllm-project/vllm/issues/56088)
- Detail: [`UPSTREAM.md`](UPSTREAM.md)

We would much rather these landed upstream than lived here.
