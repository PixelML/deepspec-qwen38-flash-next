"""Does the written cache actually align hidden states with tokens?

The S1 gate established that `lm_head(last_hidden)` reproduces the engine's own
argmax on 99.7% of positions. So if the cache is aligned the way DeepSpec
expects -- `target_last_hidden_states[i]` being the state *after* consuming
`input_ids[i]`, hence the state that predicts `input_ids[i+1]` -- then
    argmax(lm_head @ cache.target_last_hidden_states[i]) == cache.input_ids[i+1]
must hold on most positions (it is the teacher's own greedy continuation, and
on assistant spans the corpus IS that continuation).

If instead it matches `input_ids[i]`, the cache is off by one and every
downstream agreement number is meaningless. This distinguishes "the drafter is
undertrained" from "the features are misaligned" for the price of reading two
tensors and a handful of cache rows.
"""

import argparse
import json
import os

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--target-path", required=True,
                   help="Local snapshot dir of the BF16 target (for lm_head).")
    p.add_argument("--num-samples", type=int, default=16)
    p.add_argument("--out", required=True)
    cli = p.parse_args()

    from deepspec.data.target_cache_dataset import CacheDataset
    from deepspec.trainer.base_trainer import load_target_embeddings_lazily

    shims = load_target_embeddings_lazily(
        model_name_or_path=cli.target_path, revision=None, dtype=torch.bfloat16
    )
    assert shims is not None, "could not lazily read target embed/lm_head"
    _embed, lm_head = shims
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight = lm_head.weight.to(device)

    dataset = CacheDataset(cache_dir=cli.cache_dir)
    manifest = dataset.manifest
    hs = int(manifest["hidden_size"])
    taps = list(manifest["target_layer_ids"])

    stats = {
        "next_token": [0, 0],   # [correct, total]  aligned:   hidden[i] -> ids[i+1]
        "same_token": [0, 0],   # [correct, total]  off-by-one: hidden[i] -> ids[i]
        "next_token_loss_span": [0, 0],
    }
    per_sample = []
    for idx in range(min(cli.num_samples, len(dataset))):
        item = dataset[idx]
        ids = item["input_ids"].to(torch.long)
        last = item["target_last_hidden_states"].to(device=device, dtype=weight.dtype)
        loss_mask = item["loss_mask"].to(torch.long)
        n = ids.shape[0]
        pred = torch.nn.functional.linear(last, weight).argmax(-1).cpu()

        nxt = (pred[: n - 1] == ids[1:]).sum().item()
        same = (pred == ids).sum().item()
        stats["next_token"][0] += nxt
        stats["next_token"][1] += n - 1
        stats["same_token"][0] += same
        stats["same_token"][1] += n
        span = loss_mask[1:].bool()
        if span.any():
            stats["next_token_loss_span"][0] += (
                (pred[: n - 1] == ids[1:]) & span
            ).sum().item()
            stats["next_token_loss_span"][1] += int(span.sum())
        per_sample.append({
            "idx": idx, "seq_len": n,
            "next_token_rate": nxt / max(n - 1, 1),
            "same_token_rate": same / max(n, 1),
        })

    result = {
        "cache_dir": cli.cache_dir,
        "hidden_size": hs,
        "target_layer_ids": taps,
        "num_samples_checked": len(per_sample),
        "next_token_agreement": stats["next_token"][0] / max(stats["next_token"][1], 1),
        "same_token_agreement": stats["same_token"][0] / max(stats["same_token"][1], 1),
        "next_token_agreement_on_loss_span":
            stats["next_token_loss_span"][0] / max(stats["next_token_loss_span"][1], 1),
        "raw": stats,
        "per_sample": per_sample[:8],
    }
    result["verdict"] = (
        "ALIGNED" if result["next_token_agreement"] > result["same_token_agreement"] * 2
        else ("OFF_BY_ONE" if result["same_token_agreement"] > result["next_token_agreement"] * 2
              else "UNCLEAR")
    )
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    with open(cli.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
