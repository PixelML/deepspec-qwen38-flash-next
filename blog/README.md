# Our 25% inference speedup became 4.6% when we fixed the benchmark

We spent about $450 in cloud GPU time and four days training a speculative decoding drafter for Qwen3.8-Flash-Next.

When we compared its throughput with our earlier baseline, it looked like a 25% win.

Then we reran the baseline on the same benchmark. The win became 4.6%.

We're publishing the weights, serving patches, and corrected evaluator. But the measurement mistakes are probably more useful than the checkpoint.

## The baseline was already good

Speculative decoding lets a smaller model propose several tokens for the larger model to verify. The proposal has to be cheap enough, and enough tokens have to survive verification, to save time.

Qwen3.8-Flash-Next already ships with a multi-token prediction head. On our pair of DGX Sparks, that built-in head roughly doubled decoding throughput.

So beating ordinary token-by-token decoding was the easy comparison. We wanted to know whether a separately trained drafter could beat what the model already provided.

We followed DeepSpec's DFlash recipe: regenerate about 98,500 conversations with the exact NVFP4 target, cache hidden states from five layers, and train a five-layer drafter to predict a block of seven tokens in one parallel pass.

On the same 100-prompt fixture:

| Configuration | Aggregate tokens/sec |
|---|---:|
| Speculation off | 24.3 |
| Native MTP, k=3 | 49.9 |
| Our DFlash drafter, block 5 | 52.2 |

These are single-request, greedy, non-thinking results in eager mode, with 256-token outputs and an 8k context setting. All arms used the same patched vLLM engine, TP2 plus expert parallel, on two DGX Sparks (GB10). The fixture contains 33 code, 33 math, and 34 chat prompts from the held-out corpus split, with two 100-prompt runs per arm.

That is a 4.6% aggregate improvement over native MTP. The paired stratified prompt-bootstrap 95% confidence interval is +2.1% to +7.1%, but each configuration had only one engine boot. We have no estimate of boot-to-boot variability.

The earlier baseline was 41.9 tokens/sec, measured using eight short synthetic prompts on a different harness.

Comparing 52.2 against 41.9 gave us the attractive number. Comparing it against 49.9 gave us the relevant one.

## "Faster" depended on what we asked it to do

The aggregate result hid a much larger split.

With block 5, median throughput versus native MTP was:

| Workload | Change |
|---|---:|
| Math | +36.1% |
| Code | +12.0% |
| Chat | -13.5% |

A math-heavy workload and a chat application would have very different opinions of this checkpoint.

Our training mix was about 78% math and code. We then evaluated on a roughly balanced fixture. That is a plausible explanation for the split, but we haven't run the ablation to establish it.

We also trained at block size 7, which produced the highest accepted length. Block 5 still won on wall-clock throughput. These configurations need separate measurement: draft queries attend to each other bidirectionally, so a smaller block is not simply a truncated larger one.

Getting more tokens accepted does not automatically make the system faster. Drafting and verification have costs too.

## Our metric looked right because two errors cancelled out

This was the part that bothered us most.

Our offline evaluator estimated accepted length by multiplying per-position agreement rates.

But a speculative block stops contributing accepted draft tokens at its first rejection. Getting position five right does not help if position one was wrong.

Our calculation didn't enforce that. It averaged over valid positions rather than surviving prefixes. It also treated agreement across positions as independent, even though the predictions share context and are correlated.

The errors pulled in opposite directions.

The proxy said 3.08. Serving measured 2.99. Those numbers were close enough to make us think the evaluator worked.

After fixing the calculation to count surviving prefixes, the offline estimate became 4.34. The corrected evaluator reports E[L] = 1 + the sum of prefix survival probabilities, matching the server counter's convention. It ships with 13 unit tests.

The correct offline calculation was now further from serving reality than the broken one.

The correction also changed our comparison with DSpark, a variant with extra heads. The old evaluation fed its Markov head the ground-truth previous token, while serving uses its own sample. Conditioning on an accepted prefix removes that unfair advantage. We retired the old comparison; the DSpark variant is not part of this release.

## Fixing the calculation didn't fix the prediction

The gap grew with block size:

| Block | Served accepted length | Corrected offline estimate |
|---|---:|---:|
| 4 | 2.74 | 3.43 |
| 5 | 2.88 | 3.79 |
| 7 | 2.99 | 4.34 |

One hypothesis is that we were evaluating at the wrong starting positions. Offline, we sampled anchors uniformly. During serving, the next block starts where the previous block stopped, potentially concentrating attempts at harder positions.

We haven't tested that explanation.

We also haven't ruled out a hidden-state tap mismatch or differences between the offline and served data.

For now, the corrected evaluator measures offline prefix acceptance. We cannot use it as a reliable forecast of serving speed.

## Even the control needed a control

We initially required 98% token agreement between cloud and Spark execution.

We measured about 97.5%.

Then we checked the same engine against itself on a repeat run: about 97.3%.

Our threshold demanded better agreement across platforms than we had observed within one platform.

A relative comparison made more sense, but the sample was too small to establish a strong result. On about 4,600 positions, the standard error was around 0.24 percentage points, while the point estimates were only about 0.2 points apart. Changing the gate did not create stronger evidence.

We also tried comparing free-running generations token by token as a correctness check. Early numerical differences can send the continuations down different paths. That measures divergence, not just correctness. A stronger check needs common-prefix controls and explicit rejection and rollback coverage. We have not completed that certification.

## What we're releasing

The drafter is a research artifact. Stock vLLM cannot load it without the published control-flow patches and six-file adapter overlay. The checkpoint omits the target's embedding and LM head; consumers must bind those from the target at load time.

Graph-mode performance remains unmeasured. We haven't established concurrent-serving performance, soak stability, thinking-mode performance, sampling performance, or long-context performance. Losslessness is not certified. An earlier 25% aggregate release threshold was not met and was withdrawn as arbitrary; the model card preserves that decision.

If you're investigating math or code workloads on this setup, the checkpoint may be worth testing. If you're evaluating drafters more generally, start with the corrected evaluator and the failure notes.

Our first check on the next run will be boring: rerun the strongest baseline on exactly the same fixture before spending more GPU time.

## Artifacts and technical detail

- [Weights and full model card](https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash)
- [Serving patches, adapter overlay, exporter, and corrected evaluator](https://github.com/PixelML/deepspec-qwen38-flash-next)
- [Full technical write-up](technical-notes.md), including feature extraction failures, training pitfalls, engine integration, and the original measurement analysis.
- Upstream reports: [vLLM #56088](https://github.com/vllm-project/vllm/issues/56088) and [SGLang #38589](https://github.com/sgl-project/sglang/issues/38589).

Built with Qwen. The Qwen Community License 1.0 and NVIDIA Open Model License apply to the artifact; MIT applies to the DeepSpec and DFlash code. The training corpus is not published. Read the model card's licence section before use.
