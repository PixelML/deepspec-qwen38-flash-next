"""Accepted-prefix evaluation for a trained DFlash/DSpark draft checkpoint.

Reuses Qwen3DSparkModel.forward() (the same forward used in training, see
deepspec/trainer/dspark_trainer.py Qwen3DSparkTrainer.run_batch) in eval mode
against a held-out target-hidden-state cache. No live target-model forward is
needed: target_ids for the agreement check are gathered directly from the
cached input_ids by the model's own forward (see modeling.py).

Two DIFFERENT quantities are reported, side by side:

1. ``agreement_curve`` -- the historical MARGINAL per-position top-1 rate,
       p_k = mean( argmax(draft_logits_k) == target_ids_k )  over eval_mask_k
   averaged across ALL anchors valid at position k, *including anchors whose
   earlier positions already diverged*. This is what this script used to
   report, and the number the old tau estimate was built from as
   tau_proxy = 1 + sum_k prod_{j<=k} p_j.

   That product is NOT the probability of an accepted prefix. Speculative
   decoding accepts a token only if every earlier token in the same block was
   accepted, so the quantity that matters is a JOINT over one anchor. The
   product of marginals mis-conditions (p_k averages over anchors that were
   already dead) and it assumes independence across positions, which is false:
   positions inside one anchor are strongly positively correlated. Both errors
   are one-directional, so tau_proxy is a biased proxy, not an estimator.

2. ``accepted_prefix_curve`` -- the CORRECT joint quantity. For each anchor we
   walk positions 1..K in order and count position k as accepted only if every
   earlier position in that same anchor was also accepted:

       P(prefix of length k accepted) = mean_anchors( AND_{j<=k} correct_j )
       E[L] = 1 + sum_{k=1..K} P(prefix >= k)

   The leading 1 is the token the target emits for free at the anchor, which
   is how the served accepted-length counter is defined too
   (accepted_len = 1 + accepted_tokens / spec_passes).

   Two cohorts are reported because the denominator choice is a real decision:

     * ``variable_cohort`` -- at each k, the denominator is every anchor whose
       eval_mask is valid at k. Cohort membership shrinks with k for reasons
       unrelated to acceptance (sequence truncation / loss-mask padding), so
       the curve mixes populations across k.
     * ``full_depth_cohort`` -- a single fixed cohort of anchors valid at ALL
       K positions. The denominator is constant, so P(prefix >= k) is a proper
       survival function over one homogeneous population and E[L] is a real
       expectation. THIS IS THE HEADLINE.

   ``hazard`` reports P(correct at k | prefix k-1 accepted), the conditional
   that the marginal p_k is so often mistaken for. hazard_k >= p_k whenever
   positions are positively correlated, which is the whole defect in one line.

   ``expected_length_truncated_at_k`` gives E[L] for a drafter served with a
   block of only k tokens (partial sums of the survival curve), so an offline
   checkpoint can be compared against served arms run at several block sizes.

Usage:
    python scripts/eval/agreement_curve.py \
        --checkpoint-dir /vol/ckpt/dflash_block7_qwen38_flash_next/step_latest \
        --cache-dir /vol/cache/heldout_500 \
        --batch-size 8 \
        --out /vol/agreement_curve.json
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader

from deepspec.data.target_cache_dataset import CacheDataset, CacheCollator
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel


class PrefixAccumulator:
    """Streaming accumulator for both the marginal and the joint curves.

    Everything it needs per batch is ``correct`` and ``eval_mask``, both
    [bsz, num_blocks, block_size]; dim 1 indexes independent anchors sampled
    from the sequence and dim 2 indexes the block_size consecutive positions
    predicted from that one anchor. ``eval_mask`` is already a prefix mask
    (build_eval_mask cumprods it along dim -1), so validity at k implies
    validity at every j < k -- validity truncation and acceptance are
    therefore cleanly separable.
    """

    def __init__(self, block_size: int):
        self.k = block_size
        z = lambda: torch.zeros(block_size, dtype=torch.long)  # noqa: E731
        self.marginal_correct = z()
        self.valid = z()            # anchors valid at position k
        self.surv = z()             # valid at k AND accepted through k
        self.surv_prev_valid = z()  # valid at k AND accepted through k-1
        self.full_surv = z()        # full-depth anchors accepted through k
        self.full_total = 0         # anchors valid at ALL K positions
        self.anchors = 0            # anchors valid at position 1

    def update(self, correct: torch.Tensor, eval_mask: torch.Tensor) -> None:
        correct = correct.bool()
        eval_mask = eval_mask.bool()
        acc = correct & eval_mask
        # accepted-through-k: cumulative AND along the position axis, i.e. the
        # anchor survives to k only if it was correct at every j <= k.
        surv = acc.to(torch.int32).cumprod(dim=-1).bool()
        # accepted-through-(k-1), shifted so index k holds the prefix state
        # BEFORE position k; position 0 has an empty prefix, always alive.
        surv_prev = torch.cat(
            [torch.ones_like(surv[..., :1]), surv[..., :-1]], dim=-1
        )
        full = eval_mask[..., -1]  # prefix mask => valid at K implies valid all

        self.marginal_correct += acc.sum(dim=(0, 1)).cpu()
        self.valid += eval_mask.sum(dim=(0, 1)).cpu()
        self.surv += surv.sum(dim=(0, 1)).cpu()
        self.surv_prev_valid += (surv_prev & eval_mask).sum(dim=(0, 1)).cpu()
        self.full_surv += (surv & full.unsqueeze(-1)).sum(dim=(0, 1)).cpu()
        self.full_total += int(full.sum().item())
        self.anchors += int(eval_mask[..., 0].sum().item())

    def result(self) -> dict:
        def div(a, b):
            return (a / b) if b > 0 else None

        marginal, joint_var, joint_full, hazard = [], [], [], []
        for pos in range(self.k):
            valid = int(self.valid[pos])
            marginal.append({
                "position": pos + 1,
                "correct": int(self.marginal_correct[pos]),
                "total": valid,
                "top1_agreement": div(int(self.marginal_correct[pos]), valid),
            })
            joint_var.append({
                "position": pos + 1,
                "accepted_prefix": int(self.surv[pos]),
                "total": valid,
                "p_prefix_accepted": div(int(self.surv[pos]), valid),
            })
            joint_full.append({
                "position": pos + 1,
                "accepted_prefix": int(self.full_surv[pos]),
                "total": self.full_total,
                "p_prefix_accepted": div(int(self.full_surv[pos]), self.full_total),
            })
            denom = int(self.surv_prev_valid[pos])
            hazard.append({
                "position": pos + 1,
                "correct": int(self.surv[pos]),
                "total": denom,
                "p_correct_given_prefix": div(int(self.surv[pos]), denom),
            })

        def elen(curve):
            return 1.0 + sum((e["p_prefix_accepted"] or 0.0) for e in curve)

        def elen_trunc(curve):
            out, run = [], 1.0
            for e in curve:
                run += (e["p_prefix_accepted"] or 0.0)
                out.append(run)
            return out

        return {
            "block_size": self.k,
            "n_anchors": self.anchors,
            "n_full_depth_anchors": self.full_total,
            # historical marginal curve, kept verbatim for comparability
            "agreement_curve": marginal,
            "tau_proxy_product_of_marginals": _tau_proxy(marginal),
            "accepted_prefix_curve": {
                "full_depth_cohort": joint_full,
                "variable_cohort": joint_var,
            },
            "hazard_curve": hazard,
            "expected_accepted_length": {
                "full_depth_cohort": elen(joint_full),
                "variable_cohort": elen(joint_var),
            },
            "expected_length_truncated_at_k": {
                "full_depth_cohort": elen_trunc(joint_full),
                "variable_cohort": elen_trunc(joint_var),
            },
        }


def _tau_proxy(marginal_curve):
    """The OLD, WRONG estimate: tau = 1 + sum_k prod_{j<=k} p_j.

    Retained only so every rescored checkpoint carries the number it used to
    be judged by, next to the corrected one.
    """
    tau, running = 1.0, 1.0
    for entry in marginal_curve:
        running *= (entry.get("top1_agreement") or 0.0)
        tau += running
    return tau


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Must match training: Qwen3DSparkModel._forward_backbone builds a
    # flex_attention BlockMask for its block-causal draft mask, so loading with
    # "sdpa" fails inside attention with
    #   TypeError: scaled_dot_product_attention(): argument 'attn_mask' must be
    #   Tensor, not BlockMask
    # (the draft config itself pins _attn_implementation="flex_attention").
    model = Qwen3DSparkModel.from_pretrained(
        args.checkpoint_dir,
        dtype=torch.bfloat16,
        attn_implementation="flex_attention",
    ).to(device=device, dtype=torch.bfloat16).eval()

    dataset = CacheDataset(cache_dir=args.cache_dir)
    collator = CacheCollator()
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collator,
        shuffle=False,
        drop_last=False,
    )

    block_size = int(model.config.block_size) if hasattr(model.config, "block_size") else 7
    acc = PrefixAccumulator(block_size)
    n_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
                if k != "attention_mask"
            }
            # The cache stores input_ids as int32 (see TARGET_CACHE_TOKEN_DTYPE)
            # and training reaches the model through CUDAPrefetcher, which casts
            # to int64 on GPU. This script builds its own loader, so it must do
            # the same cast -- otherwise create_noise_embed's index_put_ raises
            # "Index put requires the source and destination dtypes match,
            # got Long for the destination and Int for the source".
            if batch["input_ids"].dtype != torch.long:
                batch["input_ids"] = batch["input_ids"].to(torch.long)
            outputs = model(
                input_ids=batch["input_ids"],
                target_hidden_states=batch["target_hidden_states"],
                loss_mask=batch["loss_mask"],
                target_last_hidden_states=batch["target_last_hidden_states"],
            )
            draft_logits = outputs.draft_logits  # [bsz, num_blocks, block_size, vocab]
            target_ids = outputs.target_ids  # [bsz, num_blocks, block_size]
            eval_mask = outputs.eval_mask.bool()  # [bsz, num_blocks, block_size]

            pred = draft_logits.argmax(dim=-1)
            acc.update(pred == target_ids, eval_mask)
            n_samples += draft_logits.shape[0]

    result = acc.result()
    curve = result["agreement_curve"]
    result.update({
        "checkpoint_dir": args.checkpoint_dir,
        "cache_dir": args.cache_dir,
        "n_samples": n_samples,
        "metric_note": (
            "agreement_curve is the MARGINAL per-position rate and "
            "tau_proxy_product_of_marginals is the old, biased estimate built "
            "from it. expected_accepted_length.full_depth_cohort is the "
            "corrected joint accepted-prefix expectation and is the number to "
            "report."
        ),
        "gate": {
            "position_1_agreement": curve[0]["top1_agreement"],
            "pass_threshold": 0.5,
            "stop_threshold": 0.3,
            "outcome": (
                "PASS" if (curve[0]["top1_agreement"] or 0) >= 0.5
                else ("STOP" if (curve[0]["top1_agreement"] or 0) < 0.3 else "AMBIGUOUS")
            ),
        },
    })
    print(json.dumps(result, indent=2))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
