"""S1 gate reference: dump the SAME taps via the HF `transformers` path.

Only used to validate `export_features_vllm.py`. Reuses the P2 script's
hook logic verbatim (`run_target_forward_with_hooks`) so the reference is the
exact tensor Phase 2 would have cached, and writes a .pt with the same layout
as the vLLM `--gate-dump` payload. ~38 tok/s is fine for 20 short prompts.
"""

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepspec.data.parser import preprocess_record
from deepspec.utils import load_config, parse_opts_to_config
from prepare_target_cache_qwen38_flash_next import run_target_forward_with_hooks

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--opts", action="append", default=[])
    p.add_argument("--train-data-path", required=True)
    p.add_argument("--start-row", type=int, default=0)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--min-loss-tokens", type=int, default=14)
    p.add_argument("--max-prompt-tokens", type=int, default=1024,
                   help="Skip long rows: the HF path runs at ~38 tok/s.")
    p.add_argument("--out", required=True)
    cli = p.parse_args()
    config = parse_opts_to_config(cli.opts, load_config(cli.config))
    target_layer_ids = [int(v) for v in config.model.target_layer_ids]
    revision = config.model.get("target_model_revision")

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.target_model_name_or_path, revision=revision
    )

    rows = []
    with open(cli.train_data_path, "r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx < cli.start_row:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                processed = preprocess_record(
                    record=record, tokenizer=tokenizer,
                    chat_template=config.data.chat_template,
                    max_length=int(config.data.max_length),
                )
            except AssertionError:
                continue
            if int(processed["loss_mask"].sum().item()) < cli.min_loss_tokens:
                continue
            if int(processed["input_ids"].shape[0]) > cli.max_prompt_tokens:
                continue
            rows.append((idx, processed))
            if len(rows) >= cli.limit:
                break
    print(f"[hf-gate] {len(rows)} rows selected", flush=True)

    num_gpus = torch.cuda.device_count()
    t0 = time.time()
    model = AutoModel.from_pretrained(
        config.model.target_model_name_or_path,
        revision=revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
        max_memory={i: "165GiB" for i in range(num_gpus)},
    ).eval()
    print(f"[hf-gate] model loaded in {time.time() - t0:.0f}s", flush=True)

    store = {}
    total_tokens = 0
    t1 = time.time()
    for row_idx, processed in rows:
        input_ids = processed["input_ids"].unsqueeze(0).to("cuda:0")
        attention_mask = processed["attention_mask"].unsqueeze(0).to("cuda:0")
        result = run_target_forward_with_hooks(
            target_model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_layer_ids=target_layer_ids,
        )
        seq_len = int(attention_mask.sum().item())
        store[int(row_idx)] = {
            "input_ids": processed["input_ids"][:seq_len].cpu(),
            "target_hidden_states": result.target_hidden_states[0, :seq_len].to(torch.bfloat16).cpu(),
            "target_last_hidden_states": result.target_last_hidden_states[0, :seq_len].to(torch.bfloat16).cpu(),
        }
        total_tokens += seq_len
        print(f"[hf-gate] row {row_idx}: {seq_len} tok "
              f"({total_tokens / max(time.time() - t1, 1e-9):.1f} tok/s)", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    torch.save({
        "features": store,
        "target_layer_ids": target_layer_ids,
        "tok_s": total_tokens / max(time.time() - t1, 1e-9),
        "total_tokens": total_tokens,
    }, cli.out)
    print(f"[hf-gate] wrote {cli.out}", flush=True)


if __name__ == "__main__":
    main()
