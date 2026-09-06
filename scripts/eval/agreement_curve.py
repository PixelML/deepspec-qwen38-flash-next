"""7-position top-1 agreement curve for a trained DFlash draft checkpoint.

Reuses Qwen3DSparkModel.forward() (the same forward used in training, see
deepspec/trainer/dspark_trainer.py Qwen3DSparkTrainer.run_batch) in eval mode
against a held-out target-hidden-state cache. No live target-model forward is
needed: target_ids for the agreement check are gathered directly from the
cached input_ids by the model's own forward (see modeling.py).

For each of the block_size (7) label-offset positions, computes:
    mean( (draft_logits.argmax(-1) == target_ids) & eval_mask ) over eval_mask

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Qwen3DSparkModel.from_pretrained(
        args.checkpoint_dir,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
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
    correct_per_pos = torch.zeros(block_size, dtype=torch.long)
    total_per_pos = torch.zeros(block_size, dtype=torch.long)
    n_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch.items()
                if k != "attention_mask"
            }
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
            correct = (pred == target_ids) & eval_mask

            # Aggregate per block-size position (dim=2), summed over bsz/num_blocks.
            correct_per_pos += correct.sum(dim=(0, 1)).cpu()
            total_per_pos += eval_mask.sum(dim=(0, 1)).cpu()
            n_samples += draft_logits.shape[0]

    curve = []
    for pos in range(block_size):
        total = int(total_per_pos[pos].item())
        correct_n = int(correct_per_pos[pos].item())
        agreement = correct_n / total if total > 0 else None
        curve.append({
            "position": pos + 1,
            "correct": correct_n,
            "total": total,
            "top1_agreement": agreement,
        })

    result = {
        "checkpoint_dir": args.checkpoint_dir,
        "cache_dir": args.cache_dir,
        "n_samples": n_samples,
        "block_size": block_size,
        "agreement_curve": curve,
        "gate": {
            "position_1_agreement": curve[0]["top1_agreement"],
            "pass_threshold": 0.5,
            "stop_threshold": 0.3,
            "outcome": (
                "PASS" if (curve[0]["top1_agreement"] or 0) >= 0.5
                else ("STOP" if (curve[0]["top1_agreement"] or 0) < 0.3 else "AMBIGUOUS")
            ),
        },
    }
    print(json.dumps(result, indent=2))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
