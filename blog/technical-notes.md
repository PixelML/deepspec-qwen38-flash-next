# We trained a speculative decoding drafter and it barely won. Here is everything that went wrong.

We spent about $450 of cloud GPU and four days training a DFlash drafter for
Qwen3.8-Flash-Next, then served it on a pair of DGX Sparks. It beat the model's own
speculative decoding head by 3.9% in eager mode, once we bothered to tune that head.

That's not much of a headline. The interesting part is the four measurement mistakes we made
along the way. Two we caught ourselves, and the two that mattered most we only caught because
we asked two other models to tear the work apart. If you're evaluating drafters, there's a
decent chance one of these is in your pipeline right now.

Weights are public. So are the engine patches you need to run them, and the fixed evaluator,
which is probably the most useful thing we're shipping.

## What we built

Qwen3.8-Flash-Next is a mixture of experts model, 352 GB in BF16 with roughly 6B parameters
active per token, and it ships with its own multi-token prediction head. That head is good.
On two Sparks it takes decoding from 24 tokens per second to 50. So we weren't trying to beat
plain autoregressive decoding, we were trying to beat a tuned speculative baseline. Much
harder target, and we think it's the only comparison worth making.

We followed DeepSpec's recipe fairly closely. Regenerate about 100,000 conversations with the
exact model you will serve, cache the contracted hidden states from five of its attention
layers, train a five layer drafter to predict a block of seven tokens in one parallel pass.
The idea is that one wide pass beats the built in head's three sequential ones.

Final numbers, same 100 prompt fixture for every arm, greedy, batch size one, eager mode,
256 token outputs, two boots per arm and two runs of 100 prompts per boot:

| | tokens/sec |
|---|---|
| no speculation | 24.85 |
| built in MTP, 1 token | 41.32 |
| built in MTP, 3 tokens | 50.07 |
| built in MTP, 4 tokens (best baseline) | 50.37 |
| built in MTP, 6 tokens | 46.99 |
| our drafter, block 4 | 52.02 |
| our drafter, block 5 | 52.32 |
| our drafter, block 7 | 50.54 |

Plus 3.87% overall against the best built in setting, with a 95% confidence interval of 2.10
to 5.77 percent from a paired stratified bootstrap over prompts. Block 4 is plus 3.26%
[1.84, 4.79]. Against the k=3 baseline we used to quote, block 5 is plus 4.50% [2.35, 6.80].
k=3 and k=4 are statistically a tie at about 50.2, so the honest description of the baseline
is "about 50.2", and we quote against k=4 because it is the harder number.

Split by workload against k=4 it is more interesting: math 28% faster, code a wash at plus
0.4% with a confidence interval spanning zero, chat 5.9% slower.

So: a maths drafter that costs you chat. Not nothing. Not a product either.

Those confidence intervals are paired over prompts and contain no boot to boot variance
component, because two boots cannot estimate one. Between its two boots, spec off moved 3.4%,
built in MTP k=4 moved 2.0%, our block 4 arm moved 1.0%. That is the same size as the effect
we are reporting, and it is the weakest part of this result. Two of the arms are also thinner
than the rest: built in MTP k=1 is a single boot, and our block 4 arm has three runs rather
than four. Blocks 2 and 3 required a one line widening of our own K allowlist in the serving
plugin, so those arms are labelled patched.

Worth noting because it surprised us. We trained at block 7, and block 7 does give the
highest accepted length, but block 5 is faster on the wall clock. Each block size is a
separate configuration you have to measure, not a truncation of a bigger one, because the
draft queries attend to each other bidirectionally.

## Mistake one: we compared across harnesses

Our first baseline was 41.9 tokens per second, measured with eight short synthetic prompts on
a different harness. Our drafter hit 52.3 on the real evaluation set. That's a 25% win, and
we nearly wrote it down.

Then someone re-ran the baseline on the same fixture as the drafter. It was 50.1, not 41.9.

Then we made the same class of error a second time, more quietly. We had been comparing
against the built in head at k=3, because that is the setting the model card suggests, and we
never swept it. When we did, k=4 was faster. Not by much, and the difference between k=3 and
k=4 has a confidence interval spanning zero, but it moved our headline from 4.6% to 3.9%. A
baseline you inherited is not a tuned baseline. Sweep the thing you are trying to beat.

We never decomposed how much of that was prompt length and how much was the rest of the
harness, so we can't tell you which it was. What we can tell you is the size of the error.
The old baseline was 16% below the true one, which would have inflated our ratio by about
19%. We would have published a 25% win that was really a 5% win.

Now nothing here quotes a number measured on a different harness than the thing it's compared
against, which sounds obvious written down and wasn't obvious at the time.

## Mistake two: we set gates that nothing could pass

We wanted to check that the features we extracted in the cloud matched what the Spark
produces. We set the bar at 98% token agreement.

It came back 97.5%. Fail.

Then someone ran the control we hadn't thought to run: same engine, same prompts, same
everything, twice. It agreed with itself 97.3% of the time.

Mixture of experts routing, batched reductions, atomics. The engine isn't deterministic, and
our threshold sat below its own noise floor. Nothing could have passed it. We replaced the
absolute bar with a relative one: cross platform disagreement must not exceed same platform
disagreement.

By that measure it passes, but we want to be honest about how weakly. On about 4,600 compared
positions the standard error is around 0.24 percentage points, and the two numbers are 0.2
points apart. That is a coin flip on point estimates, not a demonstration. The gate has the
right shape now. It still doesn't have the power to decide much.

We made the same mistake twice, actually. Our losslessness check compared free running
generations token by token and demanded 99% agreement. A single numerical coin flip early in
a sequence sends the two generations somewhere completely different, so that check was
measuring divergence, not correctness. Our own same configuration control agreed with itself
20% of the time.

We later measured that floor properly, with a common prefix estimator that asks how often two
runs first disagree given they have agreed so far. The per step argmax flip hazard is 0.97%
to 2.61% depending on the arm. At roughly 2% per step, almost every 256 token greedy
generation differs somewhere between two runs of the same server: eight of our eleven arms
diverge on 98% or more of requests, and the best behaved three still diverge on about 81%.
Batch size one, temperature zero, same server, same prompts.

So no gate written on free running output identity can pass on this stack, at any threshold
above about 0.99, for any configuration, drafter or not. Ours included. The right test
constrains both runs to the same prefix and asks whether the emitted tokens match what the
target would have chosen, which is what we eventually ran.

The fixed version is in the model card. It certifies the accept path at block 5 against a
measured nondeterminism floor.

## Mistake three: our acceptance metric was the wrong quantity

This one is why we're writing this up at all.

Everyone reports accepted length for drafters. It's the number that determines your speedup.
We computed it the way the evaluation script we inherited did: measure the fraction of correct
predictions at each position in the block, then multiply.

Two things are wrong with that.

The per position rates were not conditioned on survival. The mask those rates were averaged
over is a validity mask, covering sequence truncation and padding, not an acceptance mask. So
a draft that got position one wrong stayed in the denominator for position five and could
score correct there, even though in a real decode that block was already dead.

And multiplying marginal rates assumes the positions are independent. They are strongly
correlated. Positions inside one block share a context, so a drafter that is wrong at position
one is much more likely to be wrong at position two, and the product understates the joint.

The two errors point in opposite directions, which is exactly why nobody noticed. Our proxy
said 3.08, the real served accepted length was 2.99, and we took that closeness as validation.
It was two mistakes cancelling out.

We fixed it properly: walk each block in order, count position k as accepted only if every
earlier position in that same block was also accepted, and report the expected accepted length
as one plus the sum of those prefix survival probabilities, which is what the server's own
counter measures. On the same checkpoint the corrected expected accepted length is 4.34, not
3.08. The fix ships with thirteen unit tests that pin it against hand computed answers,
including the case where acceptance really is independent and the corrected number must equal
the old proxy exactly.

That's a big correction, and it changed something we'd already written up. We had written up
DSpark, a variant with extra heads, as roughly tied with DFlash. Under the corrected metric
the gap is 12.7 times wider and DSpark is clearly behind.

The reason is a second bug the broken metric was hiding. DSpark's Markov head takes the
previous token as input. Offline we were feeding it the ground truth previous token. At
serving time it consumes its own sample, which is often wrong. The old metric handed it a free
correct token exactly where reality hands it a wrong one, and the comparison was never
architecture fair. The joint metric is immune, because once you condition on the prefix being
accepted, the teacher forced token is the self generated token.

## Mistake four, which we have not fixed: offline evaluation does not predict serving

After fixing the metric, we checked it against ground truth. The corrected number is worse at
predicting served accepted length than the broken one was. Not slightly worse. About fourteen
times further off at block seven, and it gets worse as the block gets longer.

| block | served | corrected offline |
|---|---|---|
| 4 | 2.76 | 3.43 |
| 5 | 2.88 | 3.79 |
| 7 | 2.99 | 4.34 |

Offline, the hazard rate is nearly flat across positions. Position seven is about as easy as
position one, once the prefix has survived. In serving, acceptance collapses with depth. That
is a disagreement about shape, not a level offset, so no rescaling closes it.

We have a guess. We want to be clear that it is a guess, that we have not run the experiment,
and that we are publishing it as a hypothesis rather than a finding. Offline evaluation
samples anchor positions uniformly across the corpus. A real server doesn't. It re-anchors
precisely where the last block was rejected, which would mean it disproportionately starts
drafting at hard positions. If that's what's happening, it's structural to speculative
decoding rather than a bug in anyone's code, and it would suggest that offline drafter
evaluations built on uniform anchors are systematically optimistic. We don't know that. We
know our own numbers diverge and we know the divergence grows with depth.

There are at least two other candidates we haven't ruled out. The served hidden state tap may
not be the trained one, because our engine patch lets the runner fall back to a different
buffer. And the offline and served workloads are different data. Neither looks big enough to
us to explain a five times gap at position seven, but "looks big enough to us" is not a
measurement.

It's cheap to test and we intend to. Until then we wouldn't trust any offline acceptance
number, ours included, to predict serving throughput.

## Things that will cost you a day if nobody warns you

vLLM's `@support_torch_compile` installs its own `__call__` on the model class, so it never
runs `nn.Module._call_impl` and a `register_forward_hook` on that module never fires. It
doesn't error either. We processed twenty prompts, got zero captures, exited clean at 15,640
tokens per second. Any code that hooks vLLM internals needs an assertion that it actually
captured something.

Extracting features through HuggingFace transformers gave us 38 tokens per second on four
B200s, because this architecture's sparse attention indexer runs as an eager Python loop. One
attention module instance took 3.1 seconds of a 4.0 second forward. Through vLLM it was 31,000
tokens per second on a single B200. That is a comparison of two specific extraction paths on
this specific architecture, not a general claim about engines, and we spent real money finding
it out. Profile per module before you blame device placement.

DeepSpec's cache preparation spawns one process per GPU and loads a full copy of the target in
each. Fine for the 14B and smaller targets it was written for. Fatal for a 352 GB one, and it
OOMs at load on a node with ample aggregate memory, which reads like a memory fraction problem
and isn't.

Two trainers pointed at the same experiment name will resume through the same `step_latest`
symlink and silently mix two training trajectories into one curve. No crash.

At equal training, a three layer drafter scored 0.048 first position agreement and a five
layer drafter scored 0.301 on the same data. If you want a cheap diagnostic run, cut epochs,
not depth. Three layers wasn't a scaled down version of the real thing, it was below the
useful threshold entirely.

## You cannot actually serve this without patching your engine

The drafter doesn't load into stock vLLM. Four separate code paths block it, and none of them
are about our checkpoint being weird. Field for field the checkpoint is a near exact fit for
vLLM's own DFlash loader.

The older runner can't prepare the per layer embedding inputs this model needs, so the target
doesn't boot under it at all, drafter or no drafter. The newer one boots the target but
crashes handing off to the drafter, dereferencing a hidden state getter that this model
implements and returns None from. Its DFlash path hard codes a block layout different from the
one DeepSpec trains, and refuses ours with an explicit error. And a config allowlist rejects
the whole method before any weights load.

That third one deserves credit rather than complaint. vLLM raises. SGLang has the same
semantic gap on the same layout and shifts every draft position by one silently, so it would
just have looked like a bad drafter. If you're porting a DeepSpec checkpoint anywhere, check
the anchor convention before you believe a low acceptance rate.

We should be equally clear about what our fix is. Three of the changes are control flow only
and touch no numerics. But they sit on top of a six file adapter overlay that makes the target
emit contracted auxiliary hidden states at all and teaches the proposer DeepSpec's anchor
convention. The patches alone serve nothing. Everything is published, as diffs against a
pinned vLLM with base and patched hashes, and we'd much rather it landed upstream than lived
in our repo.

SGLang cannot serve this class of drafter on this model at all. The capture setup succeeds
without error, then the plumbing is severed in three places and the return slot the drafter
needs is occupied by something else. GLM's model file already has the fix pattern for its own
architecture, so the shape of the port is known. We filed it upstream rather than writing it.

## Would we do it again

Partly.

The corpus and the feature exporter were worth building, and they'd make a second run much
cheaper. The engine findings are probably more useful to other people than the weights. The
measurement corrections cost us the most and taught us the most, which is annoying but seems
to be how it goes.

The drafter itself is a marginal win on a target whose built in head is already good. That's
the real lesson, we think: a drafter is worth roughly whatever the vendor's own speculative
head left on the table. Qwen left very little. On a model with a weak built in head, the same
work would presumably buy more, though we haven't run that experiment either.

We would also pick our training prompts differently. We copied DeepSpec's mix, which is about
78% maths and code, then evaluated on a balanced set and were surprised that chat regressed.
And we generated with reasoning disabled, then realised we had no evidence about what the
people who'd want a maths accelerator on this hardware actually run. Our guess is that a lot
of them run reasoning on and sample rather than decode greedily, in which case none of our
numbers describe their setup. Match your training and evaluation distribution to what you'll
actually serve. We didn't, and we didn't check.

## What we are not claiming

Every number above is eager mode, two boots per arm, batch size one, greedy, non-thinking,
256 token outputs, 8k context, and the intervals carry no boot to boot variance component.

Graph mode is what anyone would actually deploy. We now know the memory fits under our 0.75
policy, because we booted the built in MTP baseline in graph mode twice and served four full
runs on it, and we know graphs make that baseline 3.97% slower rather than faster. What we
cannot report is graph mode for the drafter, because our own serving plugin asserts
`enforce_eager` at `serving/plugin/dflash_epoch7.py:80` and refuses to load without it. That
is our guard, not a vLLM limit and not a memory failure, so the graph comparison is one sided
and the fix is ours to make.

Concurrency and soak were never run. Neither was thinking mode or sampling, which is what
someone chasing a maths accelerator on this hardware would actually run.

Losslessness is certified for the accept path at block 5, greedy, against a measured
nondeterminism floor, and not more than that. The model card lists all of this in more detail
than this post does, and it is the document to read before using any of it.

## Links

- Weights: [PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash](https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash)
- Patches, adapter overlay, exporter, and the corrected evaluator with its tests: [PixelML/deepspec-qwen38-flash-next](https://github.com/PixelML/deepspec-qwen38-flash-next)
- SGLang upstream issue: [sgl-project/sglang#38589](https://github.com/sgl-project/sglang/issues/38589)
- vLLM upstream issue: [vllm-project/vllm#56088](https://github.com/vllm-project/vllm/issues/56088)

The training corpus is not published, because it consists of the target model's own outputs
and carries that model's licence terms.

If you work out whether the anchor sampling hypothesis holds, either way, we'd like to hear
about it.
