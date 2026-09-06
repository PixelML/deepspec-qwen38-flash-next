"""DFlash P2 diagnostic probe (2026-09-06): why is prep_cache doing ~38 tok/s
on 4x B200 instead of the expected 1-3k tok/s?

Standalone, no caching, no repo config plumbing. Loads the target model the
same way prepare_target_cache_qwen38_flash_next.py now does
(device_map="auto", max_memory=165GiB/gpu, bf16, sdpa), then:

  1. Prints device placement for every parameter group > 1GB -- specifically
     the PLE/ngram embedding (suspected to be excluded from device_map
     balancing via _no_placement_params and landing on CPU), embed_tokens,
     lm_head, and each decoder layer.
  2. Times one forward of batch=8 x seq=1024 synthetic tokens, with
     torch.cuda.synchronize() across all visible GPUs bracketing the timed
     region.
  3. Wraps one PLE module, one GatedDeltaNet (linear-attn) layer, one
     Attention (+its QSAIndexer) layer, and one SparseMoeBlock with
     forward hooks (each synced across all GPUs) to get per-module ms.
  4. Prints torch.cuda.memory_allocated() per GPU.

Exact class names confirmed by reading the real qwen4_exp modeling source
(transformers 5.16.1): Qwen4ExpTextPLELayer -> Qwen4ExpTextNGramEmbedding
(~95GB ngram_embedding.weight, excluded from device_map placement),
Qwen4ExpTextGatedDeltaNet, Qwen4ExpTextAttention -> Qwen4ExpTextQSAIndexer
(nested Python for batch_idx / for query_idx loop -- independent of
attn_implementation), Qwen4ExpTextSparseMoeBlock -> Qwen4ExpTextExperts.
"""
import argparse
import json
import time

import torch
from transformers import AutoConfig, AutoModel

GB = 1024 ** 3


def sync_all():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def param_group_device(module):
    """Best-effort single device for a module's parameters (first param
    found); reports 'mixed' if parameters span multiple devices."""
    devices = {str(p.device) for p in module.parameters(recurse=False)}
    if not devices:
        devices = {str(p.device) for p in module.parameters(recurse=True)}
    if not devices:
        return "no-params"
    if len(devices) > 1:
        return f"mixed:{sorted(devices)}"
    return next(iter(devices))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    args = parser.parse_args()

    report = {"status": "error"}
    try:
        num_gpus = torch.cuda.device_count()
        print(f"visible GPUs: {num_gpus}", flush=True)

        t_load0 = time.time()
        cfg = AutoConfig.from_pretrained(args.model, revision=args.revision)
        vocab_size = getattr(cfg, "vocab_size", None)
        text_cfg = getattr(cfg, "text_config", None)
        if vocab_size is None and text_cfg is not None:
            vocab_size = getattr(text_cfg, "vocab_size", None)
        if vocab_size is None:
            vocab_size = 248320  # fallback, per training log
        print(f"vocab_size resolved to {vocab_size}", flush=True)

        model = AutoModel.from_pretrained(
            args.model,
            revision=args.revision,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
            max_memory={i: "165GiB" for i in range(num_gpus)},
        ).eval()
        load_dt = time.time() - t_load0
        print(f"model load took {load_dt:.1f}s", flush=True)

        # ---- (1) device placement for params > 1GB, plus named specials ----
        placement = {"large_params": [], "special": {}, "per_layer": []}
        seen_special = {"ple": None, "ngram_embedding": None, "embed_tokens": None, "lm_head": None}
        for name, p in model.named_parameters():
            nbytes = p.numel() * p.element_size()
            if nbytes > 1 * GB:
                placement["large_params"].append({
                    "name": name, "shape": list(p.shape), "dtype": str(p.dtype),
                    "device": str(p.device), "gib": round(nbytes / GB, 2),
                })
            lname = name.lower()
            if "ngram_embedding" in lname and seen_special["ngram_embedding"] is None:
                seen_special["ngram_embedding"] = {"name": name, "device": str(p.device), "gib": round(nbytes / GB, 2)}
            if ".ple." in lname and seen_special["ple"] is None:
                seen_special["ple"] = {"name": name, "device": str(p.device), "gib": round(nbytes / GB, 2)}
            if "embed_tokens" in lname and seen_special["embed_tokens"] is None:
                seen_special["embed_tokens"] = {"name": name, "device": str(p.device), "gib": round(nbytes / GB, 2)}
            if "lm_head" in lname and seen_special["lm_head"] is None:
                seen_special["lm_head"] = {"name": name, "device": str(p.device), "gib": round(nbytes / GB, 2)}
        placement["special"] = seen_special

        backbone = getattr(model, "language_model", model)
        layers = getattr(backbone, "layers", None)
        if layers is not None:
            for idx, layer in enumerate(layers):
                placement["per_layer"].append({"layer_idx": idx, "device": param_group_device(layer)})

        print("=== DEVICE PLACEMENT ===", flush=True)
        print(json.dumps(placement, indent=2), flush=True)

        # ---- hook targets: one instance each of the suspect module classes ----
        target_classes = [
            "Qwen4ExpTextPLELayer",
            "Qwen4ExpTextNGramEmbedding",
            "Qwen4ExpTextGatedDeltaNet",
            "Qwen4ExpTextAttention",
            "Qwen4ExpTextQSAIndexer",
            "Qwen4ExpTextSparseMoeBlock",
            "Qwen4ExpTextExperts",
        ]
        found = {c: None for c in target_classes}
        for name, module in model.named_modules():
            cls_name = type(module).__name__
            if cls_name in found and found[cls_name] is None:
                found[cls_name] = name
        print("=== HOOK TARGETS FOUND ===", flush=True)
        print(json.dumps(found, indent=2), flush=True)

        timings = {}
        starts = {}

        def make_pre(tag):
            def _pre(_module, _args):
                sync_all()
                starts[tag] = time.perf_counter()
            return _pre

        def make_post(tag):
            def _post(_module, _args, _output):
                sync_all()
                timings.setdefault(tag, []).append((time.perf_counter() - starts[tag]) * 1000.0)
            return _post

        handles = []
        for cls_name, mod_name in found.items():
            if mod_name is None:
                continue
            module = dict(model.named_modules())[mod_name]
            handles.append(module.register_forward_pre_hook(make_pre(cls_name)))
            handles.append(module.register_forward_hook(make_post(cls_name)))

        # ---- (2) synthetic forward timing ----
        embed_device = None
        try:
            embed_device = dict(model.named_parameters())[
                [n for n in dict(model.named_parameters()) if "embed_tokens.weight" in n][0]
            ].device
        except Exception:
            embed_device = torch.device("cuda", 0)

        input_ids = torch.randint(0, int(vocab_size), (args.batch_size, args.seq_len), device=embed_device)
        attention_mask = torch.ones_like(input_ids)

        print("running warmup forward...", flush=True)
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        sync_all()
        timings.clear()  # discard warmup hook timings

        print("running timed forward...", flush=True)
        sync_all()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        sync_all()
        t1 = time.perf_counter()
        forward_s = t1 - t0
        total_tokens = args.batch_size * args.seq_len
        tok_per_s = total_tokens / forward_s

        for h in handles:
            h.remove()

        module_ms = {tag: sum(vals) for tag, vals in timings.items()}

        mem_per_gpu = {
            f"cuda:{i}": round(torch.cuda.memory_allocated(i) / GB, 2)
            for i in range(num_gpus)
        }

        report = {
            "status": "ok",
            "load_s": load_dt,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "forward_s": forward_s,
            "tok_per_s": tok_per_s,
            "module_ms": module_ms,
            "module_pct_of_forward": {
                tag: round(ms / (forward_s * 1000.0) * 100, 1) for tag, ms in module_ms.items()
            },
            "device_placement": placement,
            "hook_targets_found": found,
            "memory_allocated_gib_per_gpu": mem_per_gpu,
        }
        print("=== PROBE RESULT ===", flush=True)
        print(json.dumps(report, indent=2), flush=True)

    except Exception as e:
        report = {"status": "error", "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(report, indent=2), flush=True)

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
