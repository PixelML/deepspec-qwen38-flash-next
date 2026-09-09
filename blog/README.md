# Our 25% inference speedup became 3.9% after we fixed the benchmark twice

We spent about $450 in cloud GPU time and four days training a speculative decoding drafter for Qwen3.8-Flash-Next.

When we compared its throughput with our earlier baseline, it looked like a 25% win.

Then we reran the baseline on the same benchmark. The win became 4.6%.

Then we tuned the baseline properly, ran two boots of every arm instead of one, and the win became 3.9%.

We're publishing the weights, serving patches, and corrected evaluator. But the measurement mistakes are probably more useful than the checkpoint.

## The baseline was already good, and it wasn't even tuned

Speculative decoding lets a smaller model propose several tokens for the larger model to verify. The proposal has to be cheap enough, and enough tokens have to survive verification, to save time.

Qwen3.8-Flash-Next already ships with a multi-token prediction head. On our pair of DGX Sparks, that built-in head roughly doubled decoding throughput.

So beating ordinary token-by-token decoding was the easy comparison. We wanted to know whether a separately trained drafter could beat what the model already provided.

We followed DeepSpec's DFlash recipe: regenerate about 98,500 conversations with the exact NVFP4 target, cache hidden states from five layers, and train a five-layer drafter to predict a block of seven tokens in one parallel pass.

What we did not do, for a long time, was ask the built-in head to run at its best setting. We compared against k=3 because that's what the model card suggested. So we swept it.

| Configuration | Aggregate tokens/sec |
|---|---:|
| Speculation off | 24.85 |
| Native MTP, k=1 | 41.32 |
| Native MTP, k=3 | 50.07 |
| **Native MTP, k=4 (best baseline)** | **50.37** |
| Native MTP, k=6 | 46.99 |
| Our DFlash drafter, block 4 | 52.02 |
| **Our DFlash drafter, block 5** | **52.32** |
| Our DFlash drafter, block 7 | 50.54 |

k=4 beats k=3 by 0.60%, and that difference's confidence interval spans zero, so the honest way to say it is that the tuned baseline sits at about 50.2 and k=3 and k=4 are a tie. k=1 and k=6 are clearly worse. We quote everything against k=4 because it's the harder number to beat.

Against k=4, DFlash block 5 is **+3.87%, 95% CI [+2.10%, +5.77%]**. Block 4 is +3.26% [+1.84%, +4.79%]. Both intervals exclude zero, so the gain is real. It is just smaller than the 4.6% we published, and much smaller than the 25% we nearly published. Against k=3 the same block-5 arm is +4.50% [+2.35%, +6.80%], which is the number that lets you trace what moved.

These are single-request, greedy, non-thinking results in eager mode, with 256-token outputs and an 8k context setting. All arms used the same patched vLLM engine, TP2 plus expert parallel, on two DGX Sparks (GB10). The fixture contains 33 code, 33 math, and 34 chat prompts from the held-out corpus split. Every arm is two engine boots, two runs of 100 prompts per boot, except native MTP k=1 (one boot) and DFlash block 4 (three runs rather than four). Blocks 2 and 3 needed a one-line widening of our own K allowlist in the serving plugin, so those two arms are labelled patched.

**Here is the part that should worry you more than the number.** The confidence intervals above are paired bootstraps over prompts. They contain no boot-to-boot variance component, because two boots cannot estimate one. And the boot-to-boot movement we did see is the same size as the effect we're claiming: spec-off moved +3.4% between its two boots, native MTP k=4 moved -2.0%, DFlash block 4 moved +1.0%. So a +3.9% aggregate gain with a CI of [+2.1%, +5.8%] is a gain measured under a source of variation the interval doesn't cover. We're stating that next to the number rather than in a footnote, because it's the weakest part of the result.

The earlier baseline was 41.9 tokens/sec, measured using eight short synthetic prompts on a different harness. Comparing 52.3 against 41.9 gave us the attractive number. Comparing it against 50.4 gives us the relevant one.

## "Faster" depended on what we asked it to do

The aggregate result hid a much larger split.

With block 5, aggregate throughput versus native MTP k=4 was:

| Workload | Change | 95% CI |
|---|---:|---|
| Math | +28.0% | [+22.8%, +33.4%] |
| Code | +0.4% | [-3.2%, +4.2%] |
| Chat | -5.9% | [-8.0%, -3.9%] |

So it's a math win, a code wash, and a chat regression. Against the softer k=3 baseline the same arm reads math +32.8%, code +3.3% with a CI spanning zero, chat -9.0%.

### The numbers we posted first, and why they're bigger

Those aren't the figures in our first public post about this drafter. That post gave per-workload *medians* against native MTP *k=3*, from the single-boot run that came before we tuned the baseline and reran everything:

| Workload | Native MTP k=3, median tok/s | DFlash block 5, median tok/s | Change |
|---|---:|---:|---:|
| Math | 57.58 | 78.36 | +36.1% |
| Code | 52.50 | 58.79 | +12.0% |
| Chat | 43.73 | 37.83 | -13.5% |

We posted those as 78.3, 58.7 and 37.8 tokens/sec, which is the same numbers truncated to one decimal. They all reproduce from the per-request records, so we're leaving them up and printing them here rather than quietly replacing them.

Both sets are real measurements of different things. The table before this one is aggregate throughput, meaning total output tokens over total wall time, against the tuned k=4 baseline, over two boots per arm. This one is the median of per-request throughput against the untuned k=3 baseline on one boot. Medians drop the slow tail that aggregate throughput has to pay for, and k=3 is the easier baseline, so the same runs read higher here. Math and chat shift a bit between the two framings. Code is the one that moves materially: +12.0% by medians against k=3, and +0.4% with a confidence interval spanning zero against the tuned k=4 baseline. On the better-controlled comparison we can't tell the code gain apart from zero.

Our training mix was about 78% math and code. We then evaluated on a roughly balanced fixture. That is a plausible explanation for the split, but we haven't run the ablation to establish it.

We also trained at block size 7, which produced the highest accepted length. Block 5 still won on wall-clock throughput. These configurations need separate measurement: draft queries attend to each other bidirectionally, so a smaller block is not simply a truncated larger one.

Getting more tokens accepted does not automatically make the system faster. Drafting and verification have costs too. On chat the drafter accepts 2.14 tokens per pass against the native head's 2.76, while paying a cheaper pass (54 ms against 66 ms), and it still loses.

### We hoped a shorter block would save chat. We tested it. It doesn't.

The obvious fix for a chat regression is a smaller block: draft fewer tokens, waste less on rejection. When we first published this we hadn't run blocks 2 and 3, so we said so and left it open. Now we've run them, against the tuned baseline:

| Block | Chat vs native MTP k=4 | 95% CI |
|---|---:|---|
| 2 | -6.13% | [-8.10%, -4.13%] |
| 3 | -3.87% | [-5.74%, -2.04%] |
| 4 | -3.90% | [-5.85%, -1.93%] |
| 5 | -5.90% | [-7.96%, -3.89%] |
| 7 | -11.32% | [-13.17%, -9.41%] |

Every interval excludes zero. And chat doesn't keep improving as the block shrinks. It bottoms out around blocks 3 and 4 at about -3.9%, then gets worse again at block 2, which also gives up most of the math gain (-7.3%). Chat throughput never approaches native MTP k=3's 43.3 tokens/sec on chat; the drafter tops out near 40.

The narrow release we were holding open, where you serve the drafter only for math and code, now fails on a measurement instead of on an assumption. That's a better place to fail.

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
| 4 | 2.76 | 3.43 |
| 5 | 2.88 | 3.79 |
| 7 | 2.98 | 4.34 |

One hypothesis is that we were evaluating at the wrong starting positions. Offline, we sampled anchors uniformly. During serving, the next block starts where the previous block stopped, potentially concentrating attempts at harder positions.

We haven't tested that explanation.

We also haven't ruled out a hidden-state tap mismatch or differences between the offline and served data.

For now, the corrected evaluator measures offline prefix acceptance. We cannot use it as a reliable forecast of serving speed.

## The engine doesn't agree with itself, and that's why our gates could never pass

We initially required 98% token agreement between cloud and Spark execution.

We measured about 97.5%.

Then we checked the same engine against itself on a repeat run: about 97.3%.

Our threshold demanded better agreement across platforms than we had observed within one platform.

We knew that was embarrassing. We didn't realise how deep it went until we measured it properly. At batch size one, temperature 0, greedy, on the same server, with the same prompts, two runs do not produce the same tokens. We measured the per-step argmax flip hazard on a common prefix, which is the rate at which two runs first disagree given they've agreed so far:

| Arm | Per-step flip hazard | Runs that diverge at all |
|---|---:|---:|
| Spec off | 2.61% | 99% |
| Native MTP k=3 | 2.08% | 98.5% |
| Native MTP k=4 | 0.97% | 81% |
| Native MTP k=6 | 1.06% | 80.5% |
| DFlash, blocks 2 through 7 | 2.15% to 2.51% | 98% to 100% |

At roughly 2% per step, almost every 256-token greedy generation differs somewhere between two runs of the same server. Eight of our eleven arms diverge on 98% or more of requests. The three best-behaved arms still diverge on about 81%.

This isn't a drafter property. It's the engine, this model, this hardware, at concurrency one with sampling turned off. And it means any gate written on free-running output identity, including our harness's own `token_agreement >= 0.99`, cannot pass and never could. We spent time treating a failed 0.98 gate as a fidelity problem when it was a resolution problem.

If you're building acceptance gates for speculative decoding, measure your engine against itself first. Your floor is probably not where you think it is.

## Losslessness, done properly this time

The first version of this check was not a certification and we said so. It compared every emitted token, including the correction and bonus tokens the accept path doesn't own, it replayed different prefixes across arms, and it had a blind spot of about 2.5% from its own control's mismatch rate. Its recorded maximum mismatch margins exceeded 12 log units, which is not what "all mismatches are near-ties" looks like.

We redid it with the controls that were missing. We captured one continuation from the DFlash server at block 5 and recovered the engine-step block boundaries from the streaming chunk boundaries, so we know which emitted positions were accepted drafts and which were the block-final correction slot. Then we scored that same continuation under teacher forcing on the DFlash server and twice on a spec-off server, so every arm sees an identical prefix, and the two spec-off replays give the pure nondeterminism floor at those same positions.

| | All emitted tokens | Accepted-draft positions only |
|---|---:|---:|
| Positions checked | 25,600 | **16,711** |
| Violation rate | 2.316% | **0.784%** |
| Nondeterminism floor, identical prefix | 2.215% | **0.844%** |

On the positions the accept path is actually responsible for, the drafter disagrees with the spec-off target less often than the target disagrees with itself. The standard error is about 0.07 percentage points, so this version of the check resolves down to roughly 0.2 points, where the old one was blind below about 2.5%.

We also broke the mismatch rate out by position within the block, because a real accept-path bug concentrates at one offset and noise doesn't:

| Offset in block | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|---:|
| Positions checked | 8,889 | 6,252 | 4,199 | 2,805 | 1,991 | 1,464 |
| Violation | 2.70% | 2.51% | 2.17% | 1.93% | 1.71% | 1.16% |
| Floor | 2.80% | 2.42% | 1.79% | 1.85% | 1.51% | 0.68% |

No concentration anywhere. Violation tracks the floor at every offset.

And the 12-log-unit mismatches turned out to be explainable. There are 49 of them above 4 log units, with a maximum of 12.0, and all 49 sit at the block-final position, which is the bonus and correction slot. None are at an accepted-draft position. The same tail shows up when the DFlash server scores its own capture, 53 cases with a maximum of 12.9, so it's a property of teacher-forced replay at the correction slot, not of the accept path.

So we'll say it: **the accept path is certified lossless at block 5**, greedy, against a measured nondeterminism floor. That is a stronger claim than the one we published, and we were underselling ourselves.

It is not an absolute claim. It's certified against a floor we measured on this stack, at one block size, greedy, under teacher forcing. It says nothing about sampling, other blocks, or long context.

## Graph mode: we were wrong about which way it cuts

Every serving number here is still eager mode, and we previously wrote that graph capture had never been shown to fit under our 0.75 memory policy, and that eager might be flattering the drafter, because one wide five-layer draft pass and three narrow MTP passes have comparable kernel launch counts.

Both halves of that are now wrong.

Graph memory does fit. We booted native MTP k=4 twice with `{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}` at utilization 0.75 and served four full runs. That's the first time we've demonstrated it. The earlier attempt died inside `profile_cudagraph_memory` before it ever got to capture.

And graphs make the baseline slower, not faster. Native MTP k=4 in graph mode is 48.37 tokens/sec against 50.37 eager, which is **-3.97% [-4.68%, -3.26%]**, consistent across both boots and all three workloads. So the eager-only protocol wasn't flattering the drafter. If anything it was costing us margin.

We can't finish the comparison, and the reason is ours. Our own serving adapter refuses to load the drafter unless eager mode is on. It's a one-line assertion in the plugin we published, `serving/plugin/dflash_epoch7.py:80`, raising `ValueError("HC attachment requires enforce_eager")` during the draft model's `__init__`. It's a guard we wrote for correctness of the hidden-state tap, not a vLLM limitation and not a memory failure. So the graph A/B is one-sided: baseline only. Fixing that guard is on us.

## What we're releasing

The drafter is a research artifact. Stock vLLM cannot load it without the published control-flow patches and six-file adapter overlay. The checkpoint omits the target's embedding and LM head; consumers must bind those from the target at load time.

What's now established: a real but small aggregate gain over a tuned baseline, a large math gain, a code wash, a chat regression at every block we can serve, a certified accept path at block 5, and graph memory fit for the baseline.

What isn't: graph-mode throughput for the drafter, because of our own adapter guard. Concurrent serving, soak stability, thinking mode, sampling, and long context. Boot-to-boot variance, which two boots can't estimate and which moves as much as the effect we're reporting. And the math gain in the mode someone chasing a math accelerator on this hardware would actually run, which is thinking mode with sampling, not greedy non-thinking 256-token decodes.

An earlier 25% aggregate release threshold was not met and was withdrawn as arbitrary; the model card preserves that decision.

If you're investigating math or code workloads on this setup, the checkpoint may be worth testing. If you're evaluating drafters more generally, start with the corrected evaluator, the nondeterminism floor, and the failure notes.

Our first check on the next run will be boring: rerun the strongest baseline, tuned, on exactly the same fixture, and boot it more than twice, before spending more GPU time.

## Artifacts and technical detail

- [Weights and full model card](https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash)
- [Serving patches, adapter overlay, exporter, and corrected evaluator](https://github.com/PixelML/deepspec-qwen38-flash-next)
- [Full technical write-up](technical-notes.md), including feature extraction failures, training pitfalls, engine integration, and the original measurement analysis.
- Upstream reports: [vLLM #56088](https://github.com/vllm-project/vllm/issues/56088) and [SGLang #38589](https://github.com/sgl-project/sglang/issues/38589).

Built with Qwen. The Qwen Community License 1.0 and NVIDIA Open Model License apply to the artifact; MIT applies to the DeepSpec and DFlash code. The training corpus is not published. Read the model card's licence section before use.
