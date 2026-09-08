"""S1 gate: per-tap cosine similarity between vLLM-exported and HF-exported
target features. CPU only.

Gate (coordinator contract): every tap >= 0.98 pass; < 0.95 stop and report.
"""

import argparse
import json

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vllm", required=True)
    p.add_argument("--hf", required=True)
    p.add_argument("--hidden-size", type=int, default=2560)
    p.add_argument("--out", required=True)
    cli = p.parse_args()

    a = torch.load(cli.vllm, map_location="cpu", weights_only=False)
    b = torch.load(cli.hf, map_location="cpu", weights_only=False)
    taps = [int(t) for t in a["target_layer_ids"]]
    assert taps == [int(t) for t in b["target_layer_ids"]], "tap sets differ"
    hs = int(cli.hidden_size)

    rows = sorted(set(a["features"]) & set(b["features"]))
    per_tap = {str(t): [] for t in taps}
    per_tap["last"] = []
    token_mismatch = []
    for row in rows:
        fa, fb = a["features"][row], b["features"][row]
        ia, ib = fa["input_ids"], fb["input_ids"]
        n = min(len(ia), len(ib))
        if not torch.equal(ia[:n], ib[:n]):
            token_mismatch.append(row)
            continue
        ha = fa["target_hidden_states"][:n].float()
        hb = fb["target_hidden_states"][:n].float()
        for k, tap in enumerate(taps):
            sa = ha[:, k * hs:(k + 1) * hs]
            sb = hb[:, k * hs:(k + 1) * hs]
            per_tap[str(tap)].append(
                torch.nn.functional.cosine_similarity(sa, sb, dim=-1)
            )
        per_tap["last"].append(
            torch.nn.functional.cosine_similarity(
                fa["target_last_hidden_states"][:n].float(),
                fb["target_last_hidden_states"][:n].float(),
                dim=-1,
            )
        )

    table = {}
    worst = 1.0
    for key, chunks in per_tap.items():
        if not chunks:
            continue
        values = torch.cat(chunks)
        table[key] = {
            "mean": float(values.mean()),
            "p05": float(values.quantile(0.05)),
            "min": float(values.min()),
            "n_tokens": int(values.numel()),
        }
        worst = min(worst, table[key]["mean"])

    # Guard against a hook-wiring bug that would capture the same tensor for
    # several taps: within EACH dump, adjacent taps must be clearly distinct.
    def _within(dump, label):
        stats = {}
        for row in rows[:5]:
            hh = dump["features"][row]["target_hidden_states"].float()
            for i in range(len(taps)):
                for j in range(i + 1, len(taps)):
                    key = f"{taps[i]}~{taps[j]}"
                    c = torch.nn.functional.cosine_similarity(
                        hh[:, i * hs:(i + 1) * hs], hh[:, j * hs:(j + 1) * hs], dim=-1
                    ).mean()
                    stats.setdefault(key, []).append(float(c))
            for i, tap in enumerate(taps):
                stats.setdefault(f"std_{tap}", []).append(
                    float(hh[:, i * hs:(i + 1) * hs].std())
                )
        return {label: {k: round(sum(v) / len(v), 5) for k, v in stats.items()}}

    within = {}
    within.update(_within(a, "vllm"))
    within.update(_within(b, "hf"))

    result = {
        "within_dump_cross_tap": within,
        "per_tap_cosine": table,
        "worst_mean_cosine": worst,
        "num_rows_compared": len(rows) - len(token_mismatch),
        "token_mismatch_rows": token_mismatch,
        "lm_head_argmax": a.get("lm_head_argmax"),
        "vllm_tok_s": a.get("tok_s"),
        "hf_tok_s": b.get("tok_s"),
        "gate_cosine_pass": worst >= 0.98,
        "gate_cosine_stop": worst < 0.95,
    }
    argmax = a.get("lm_head_argmax") or {}
    result["gate_argmax_pass"] = float(argmax.get("rate", 0.0)) >= 0.98
    result["status"] = "ok"
    with open(cli.out, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
