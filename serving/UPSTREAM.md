# Upstream

We would rather these landed in the engines than lived here.

## vLLM

**[vllm-project/vllm#56088](https://github.com/vllm-project/vllm/issues/56088)** —
*qwen4_exp (Qwen3.8-Flash-Next) cannot serve DeepSpec DFlash/DSpark drafters: five blockers,
patches available.*

Covers:

1. The V1 runner cannot prepare PLE inputs, so the target does not boot at all under it
   (`vllm/models/qwen4_exp/nvidia/model.py:300`).
2. The V2 runner dereferences a `None` MTP hidden-state getter
   (`vllm/v1/worker/gpu/model_runner.py:834`). Patch offered.
3. The DFlash speculator hard-rejects DeepSpec's anchor layout
   (`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:69`), even though `DSparkSpeculator`
   already selects that layout and the shared Triton kernel already implements it. Patch
   offered. Not needed for DSpark.
4. The `Qwen4Exp` spec-method allowlist omits `dflash` and `dspark`
   (`vllm/model_executor/models/config.py`), and rejects before weights load. Patch offered.
5. Adaptive verification is structurally incompatible with GDN backends
   (`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py:463`). This is the one with the
   widest blast radius, since it affects every Mamba-class hybrid target. No fix offered; we
   list the options we can see.

The six-file adapter overlay is raised in the same issue as a feature request rather than a
bug, since it is a capability the target does not currently have rather than something broken.

## SGLang

**[sgl-project/sglang#38589](https://github.com/sgl-project/sglang/issues/38589)** —
`qwen4_exp` cannot emit DFlash auxiliary hidden states at all, and the capture setup succeeds
silently, so the misconfiguration is invisible. The plumbing is severed in three places in
`models/qwen4_exp.py` and the return slot the DFlash worker reads is occupied by the
HC-flattened stream. `models/glm5_next.py` already carries the fix pattern for its own mHC
architecture (sglang #36708), so the shape of the port is known.

The same issue also reports the silent anchor mis-shift: where vLLM raises on the
`sample_from_anchor` layout mismatch, SGLang shifts every draft position by one with no error
and presents only as poor acceptance.

We filed and did not port. No PR was opened.
