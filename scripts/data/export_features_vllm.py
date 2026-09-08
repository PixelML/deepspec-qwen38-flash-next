"""Target-hidden-state export for Qwen3.8-Flash-Next (qwen4_exp) via vLLM.

WHY THIS EXISTS (see docs/dflash-training-log.md, Phase 2 -> Phase 3):
scripts/data/prepare_target_cache_qwen38_flash_next.py drives the HF
`transformers` implementation, which runs at ~38 tok/s on 4x B200 because
`Qwen4ExpTextQSAIndexer` is an eager nested-Python loop (measured: ~95% of
forward time). That is unusable for a 100k-row cache. vLLM ships a fused CUDA
QSA indexer, and -- decisively -- the vLLM NVFP4 engine is the *exact* engine
that generated the regen corpus, so its hidden states are the ones a drafter
deployed against that engine will actually see.

WHAT IT CAPTURES

DeepSpec's cache format wants, per token:
  * `target_hidden_states`      -- concat of the native-width (2560) residual
                                   "after layer L" for each L in
                                   `model.target_layer_ids`  -> 5*2560 = 12800
  * `target_last_hidden_states` -- the model's final contracted hidden -> 2560

Qwen4Exp's inter-layer residual is HyperConnection-wide
(`hc_count * hidden_size` = 4*2560 = 10240). The native 2560-wide stream only
exists as the *block input* produced by a `GatedResidual` mix. Mapping between
the two implementations (both read from real source, not assumed):

  HF   `Qwen4ExpTextGatedResidual.forward(...)` (use_combine=True)
         -> (mixed_input[2560], hyper_input, injection_weights)
         i.e. the tap is return value **[0]**.
  vLLM `.../qwen4_exp/nvidia/hyperconnection.py::GatedResidual`
         `.mix(hidden_states)` and
         `.combine_and_mix(hidden_states, prev_block_output, prev_injection)`
         -> (hidden_states[10240], block_input[2560], injection)
         i.e. the tap is return value **[1]**.

`block_input` is the same quantity as HF's `mixed_input`: grouped GemmaRMSNorm
over the HC streams, low-rank silu/sigmoid gate, gated mean over hc_count.
The NVIDIA variant merely *defers* each combine to the next mix boundary
(`combine_and_mix` fuses the pending residual add into its input RMSNorm), so
the value at layer L+1's `attn_hyper_connection` is still exactly "the fully
combined state after layer L, contracted" -- DeepSpec's own
`extract_context_feature` convention (`hidden_states[layer_id + 1]`).

Two structural consequences, both handled below:
  1. vLLM's `GatedResidual` has **no `forward()`** -- the decoder layer calls
     `.mix()` / `.combine_and_mix()` directly -- so `register_forward_hook`
     would never fire. We patch those two methods at class level and dispatch
     on a per-instance `_ds_tap` tag we attach to just the tapped modules.
  2. `Qwen4ExpModel.forward` returns `final_mixer.combine_and_mix(...)[1]`,
     which IS the HF `last_hidden_state` (HF applies no separate final norm
     after `hyper_connection_mixer`). So the final mixer is tapped the same
     way, tagged `_ds_tap = "last"`.

PLE: `ple_layer_ids` is 1-based; layer i owns a PLE iff `(i+1) in
ple_layer_ids`. For the NVFP4 teacher that is `[2]` -> only layer_idx 1. The
tap layers here are `layer_id + 1` in {4,16,24,36,44}, none of which has a
PLE, so every tap goes through `combine_and_mix`. Both methods are patched
anyway so a different tap set stays correct.

BATCH SLICING
vLLM V1 flattens the whole scheduled batch into one `[num_tokens, ...]`
sequence. We recover per-request spans from `query_start_loc` (passed straight
into the model forward for the PLE n-gram path) and cross-check against
`positions == 0`. Each span is then *identified* by its exact token ids
(prefill-only, `max_tokens=1`, no prefix caching, no chunked prefill => every
request is prefilled exactly once, whole). A span that matches no pending
request is a hard error rather than a silent corruption.
"""

import argparse
import gc
import json
import os
import time
from collections import defaultdict, deque

import torch

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from deepspec.data.parser import preprocess_record  # noqa: E402
from deepspec.data.target_cache_dataset import (  # noqa: E402
    AsyncTargetCacheWriter,
    LocalCacheWriteSummary,
    atomic_json_dump,
    build_global_target_cache_shard_map,
    build_target_cache_manifest,
    cleanup_target_cache_tmp_dir,
    finalize_target_cache_index,
    load_local_cache_write_summary,
    rename_local_target_cache_shards,
    write_target_cache_manifest,
)
from deepspec.utils import load_config, parse_opts_to_config  # noqa: E402


# --------------------------------------------------------------------------
# capture state (module-level: the patched GatedResidual methods are class
# level, so per-instance dispatch happens via the `_ds_tap` tag)
# --------------------------------------------------------------------------
_CAPTURED: dict = {}
_CAPTURE_ENABLED = False


def _patch_gated_residual(gated_residual_cls):
    """Patch `mix` / `combine_and_mix` to record return value [1] (block_input,
    the native-width contracted residual) for tagged instances only."""
    if getattr(gated_residual_cls, "_ds_patched", False):
        return
    orig_mix = gated_residual_cls.mix
    orig_cm = gated_residual_cls.combine_and_mix

    def _record(self, out):
        if not _CAPTURE_ENABLED:
            return out
        tap = getattr(self, "_ds_tap", None)
        if tap is not None:
            # out[1] == block_input == HF's `mixed_input`, [num_tokens, hidden]
            _CAPTURED[tap] = out[1]
        return out

    def mix(self, hidden_states):
        return _record(self, orig_mix(self, hidden_states))

    def combine_and_mix(self, hidden_states, prev_block_output, prev_injection):
        return _record(
            self, orig_cm(self, hidden_states, prev_block_output, prev_injection)
        )

    gated_residual_cls.mix = mix
    gated_residual_cls.combine_and_mix = combine_and_mix
    gated_residual_cls._ds_patched = True


def _unwrap_causal_lm(module):
    """The registered top-level module is the VLM wrapper
    `Qwen4ExpForConditionalGeneration`, whose `.language_model` is the
    `Qwen4ExpForCausalLM` that owns `.model` (Qwen4ExpModel) and `.lm_head`.
    Note the wrapper calls `self.language_model.model(...)` DIRECTLY, so the
    only module whose forward hooks see every batch is Qwen4ExpModel."""
    name = type(module).__name__
    if name == "Qwen4ExpForCausalLM":
        return module
    inner = getattr(module, "language_model", None)
    if inner is not None and type(inner).__name__ == "Qwen4ExpForCausalLM":
        return inner
    raise RuntimeError(f"unexpected top-level model class {name!r}")


def _find_model(llm):
    """Locate the live Qwen4Exp top-level module inside the in-process engine.

    `VLLM_ENABLE_V1_MULTIPROCESSING=0` keeps the engine core and its worker in
    this process, so the module object we tag/hook is the one that actually
    runs. Try the public `apply_model` API first, then a bounded BFS over the
    object graph (attribute paths churn between vLLM releases; this survives
    that)."""
    found = []
    try:
        llm.apply_model(lambda m: found.append(m))
        if found:
            return found[0]
    except Exception as exc:  # pragma: no cover - version dependent
        print(f"[export] apply_model unavailable ({exc}); falling back to BFS", flush=True)

    import torch.nn as nn

    seen = set()
    frontier = [llm]
    for _depth in range(12):
        nxt = []
        for obj in frontier:
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            if isinstance(obj, nn.Module) and type(obj).__name__ in (
                "Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"
            ):
                return obj
            for value in list(getattr(obj, "__dict__", {}).values()):
                if isinstance(value, (list, tuple)):
                    nxt.extend(v for v in value if hasattr(v, "__dict__"))
                elif hasattr(value, "__dict__"):
                    nxt.append(value)
        frontier = nxt
    raise RuntimeError("could not locate the Qwen4Exp model inside the vLLM engine")


def _segment_boundaries(num_tokens, positions, query_start_loc):
    """Per-request [start, end) spans within the flattened batch."""
    from_qsl = None
    if isinstance(query_start_loc, torch.Tensor) and query_start_loc.numel() >= 2:
        qsl = query_start_loc.tolist()
        # The runner may pad query_start_loc; trailing entries repeat num_tokens.
        cut = [qsl[0]]
        for value in qsl[1:]:
            if value > cut[-1]:
                cut.append(value)
            if value >= num_tokens:
                break
        if cut[-1] == num_tokens and len(cut) >= 2:
            from_qsl = [(cut[i], cut[i + 1]) for i in range(len(cut) - 1)]
    # Independent derivation: prefill-only + no prefix caching => every request's
    # positions run 0..L-1, so `positions == 0` marks each sequence start.
    # vLLM may hand the model an mrope-shaped or otherwise degenerate
    # `positions` tensor (observed: all zeros), so this is a best-effort
    # cross-check only -- `query_start_loc` is authoritative, and every span
    # is independently confirmed by exact token-id identity below.
    if positions is None or positions.dim() != 1 or int((positions != 0).sum()) == 0:
        return (from_qsl, None) if from_qsl is not None else (None, None)
    starts = (positions == 0).nonzero(as_tuple=True)[0].tolist()
    from_pos = None
    if starts and starts[0] == 0:
        bounds = starts + [num_tokens]
        from_pos = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    if from_qsl is not None:
        return from_qsl, from_pos
    if from_pos is not None:
        return from_pos, None
    raise RuntimeError("cannot segment batch: no query_start_loc and no positions==0")


class Exporter:
    def __init__(self, *, taps, writer, pending, hidden_size, on_sample=None):
        self.taps = list(taps)
        self.writer = writer
        self.pending = pending  # tuple(ids) -> deque of (row_idx, loss_mask)
        self.hidden_size = int(hidden_size)
        self.on_sample = on_sample
        self.source_row_ids = []
        self.num_written = 0
        self.num_tokens = 0
        self.segment_mismatches = 0

    def handle_forward(self, input_ids, positions, query_start_loc):
        num_tokens = int(input_ids.shape[0])
        spans, cross = _segment_boundaries(num_tokens, positions, query_start_loc)
        if cross is not None and cross != spans:
            self.segment_mismatches += 1
            print(
                f"[export] WARNING query_start_loc/positions disagree: "
                f"{spans[:4]} vs {cross[:4]}",
                flush=True,
            )
        missing = [t for t in self.taps + ["last"] if t not in _CAPTURED]
        if missing:
            raise RuntimeError(f"taps not captured this forward: {missing}")
        ids_cpu = input_ids.tolist()
        for start, end in spans:
            key = tuple(ids_cpu[start:end])
            bucket = self.pending.get(key)
            if not bucket:
                raise RuntimeError(
                    "batch span matched no pending request (len="
                    f"{end - start}); chunked prefill or prefix caching is on?"
                )
            row_idx, loss_mask = bucket.popleft()
            if not bucket:
                self.pending.pop(key, None)
            hidden = torch.cat(
                [_CAPTURED[t][start:end] for t in self.taps], dim=-1
            ).to(torch.bfloat16).cpu()
            last = _CAPTURED["last"][start:end].to(torch.bfloat16).cpu()
            seq_len = end - start
            assert hidden.shape == (seq_len, self.hidden_size * len(self.taps)), (
                f"tap concat shape {tuple(hidden.shape)} != "
                f"({seq_len}, {self.hidden_size * len(self.taps)})"
            )
            assert last.shape == (seq_len, self.hidden_size)
            token_ids = torch.tensor(ids_cpu[start:end], dtype=torch.long)
            if self.writer is not None:
                self.writer.write_sample(
                    input_ids=token_ids,
                    attention_mask=torch.ones(seq_len, dtype=torch.long),
                    loss_mask=loss_mask[:seq_len],
                    target_hidden_states=hidden,
                    target_last_hidden_states=last,
                )
                self.source_row_ids.append(int(row_idx))
            if self.on_sample is not None:
                self.on_sample(row_idx, token_ids, hidden, last, loss_mask[:seq_len])
            self.num_written += 1
            self.num_tokens += seq_len
        _CAPTURED.clear()


# --------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--opts", action="append", default=[])
    p.add_argument("--train-data-path", action="append", required=True)
    p.add_argument("--output-dir", default=None,
                   help="DeepSpec cache dir. Omit in --gate-dump mode.")
    p.add_argument("--model-path", required=True,
                   help="Local snapshot dir (or repo id) of the NVFP4 teacher.")
    p.add_argument("--tokenizer-path", default=None)
    p.add_argument("--start-row", type=int, default=0)
    p.add_argument("--end-row", type=int, default=-1, help="-1 = to end of file")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--finalize", action="store_true",
                   help="Assemble shards/index/manifest from existing _tmp/rank_* dirs and exit.")
    p.add_argument("--min-loss-tokens", type=int, default=14)
    p.add_argument("--max-shard-bytes", type=int, default=64 * 1024 ** 3)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=4160)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--max-num-batched-tokens", type=int, default=16384)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--gate-dump", default=None,
                   help="Write raw features + prompt-logprob argmax check to this .pt "
                        "instead of a cache (S1 gate mode).")
    p.add_argument("--gate-limit", type=int, default=20)
    p.add_argument("--max-prompt-tokens", type=int, default=-1,
                   help="Skip rows longer than this. Used in gate mode to select "
                        "exactly the same rows the (38 tok/s) HF reference can afford.")
    p.add_argument("--progress-every", type=int, default=200)
    p.add_argument("--commit-every", type=int, default=2000)
    cli = p.parse_args()
    config = parse_opts_to_config(cli.opts, load_config(cli.config))
    return cli, config


def _read_rows(paths, start, end):
    rows = []
    idx = 0
    for path in sorted(paths):
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if end >= 0 and idx >= end:
                    break
                if idx >= start:
                    line = line.strip()
                    if line:
                        try:
                            rows.append((idx, json.loads(line)))
                        except json.JSONDecodeError:
                            pass
                idx += 1
    return rows


def _finalize(output_dir, *, config, cli, target_layer_ids, hidden_size, world_size):
    summaries = [
        load_local_cache_write_summary(os.path.join(output_dir, "_tmp", f"rank_{r}"))
        for r in range(world_size)
    ]
    shard_map, shards = build_global_target_cache_shard_map(summaries)
    for summary in summaries:
        rank_dir = os.path.join(output_dir, "_tmp", f"rank_{int(summary['global_rank'])}")
        rename_local_target_cache_shards(
            output_dir=output_dir, rank_dir=rank_dir, summary=summary, shard_map=shard_map
        )
    num_samples = finalize_target_cache_index(
        output_dir=output_dir, summaries=summaries, shard_map=shard_map
    )
    row_ids = []
    for summary in sorted(summaries, key=lambda s: int(s["source_sample_start"])):
        rank_dir = os.path.join(output_dir, "_tmp", f"rank_{int(summary['global_rank'])}")
        path = os.path.join(rank_dir, "source_row_ids.json")
        if os.path.exists(path):
            with open(path) as handle:
                row_ids.extend(json.load(handle))
    manifest = build_target_cache_manifest(
        num_samples=num_samples,
        shards=shards,
        target_layer_ids=target_layer_ids,
        hidden_size=hidden_size,
        extra_fields={
            "target_model_name_or_path": str(config.model.target_model_name_or_path),
            "feature_source": "vllm",
            "vllm_model_path": str(cli.model_path),
            "source_jsonl_paths": [str(p) for p in cli.train_data_path],
            "source_row_range": [int(cli.start_row), int(cli.end_row)],
            "chat_template": str(config.data.chat_template),
            "max_length": int(config.data.max_length),
            "min_loss_tokens": int(cli.min_loss_tokens),
            "project_name": str(config.get("project_name")),
            "exp_name": str(config.get("exp_name")),
        },
    )
    write_target_cache_manifest(output_dir=output_dir, manifest=manifest)
    atomic_json_dump(row_ids, os.path.join(output_dir, "source_row_ids.json"))
    cleanup_target_cache_tmp_dir(output_dir)
    print(json.dumps({"status": "ok", "num_samples": num_samples,
                      "num_shards": len(shards)}, indent=2), flush=True)
    return num_samples


def main():
    cli, config = parse_args()
    target_layer_ids = [int(v) for v in config.model.target_layer_ids]
    hidden_size = 2560
    output_dir = os.path.abspath(cli.output_dir) if cli.output_dir else None

    if cli.finalize:
        _finalize(output_dir, config=config, cli=cli,
                  target_layer_ids=target_layer_ids, hidden_size=hidden_size,
                  world_size=cli.world_size)
        return

    from transformers import AutoTokenizer
    tok_path = cli.tokenizer_path or cli.model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    hidden_size = int(getattr(config.model, "hidden_size", 0) or 2560)

    t_tok = time.time()
    rows = _read_rows(cli.train_data_path, cli.start_row, cli.end_row)
    if cli.world_size > 1:
        rows = rows[cli.rank::cli.world_size]
    print(f"[export] {len(rows)} rows to tokenize", flush=True)

    pending = defaultdict(deque)
    prompts = []
    num_skipped = 0
    max_len = int(config.data.max_length)
    for row_idx, record in rows:
        try:
            processed = preprocess_record(
                record=record,
                tokenizer=tokenizer,
                chat_template=config.data.chat_template,
                max_length=max_len,
            )
        except AssertionError:
            num_skipped += 1
            continue
        if int(processed["loss_mask"].sum().item()) < cli.min_loss_tokens:
            num_skipped += 1
            continue
        if 0 < cli.max_prompt_tokens < int(processed["input_ids"].shape[0]):
            num_skipped += 1
            continue
        ids = processed["input_ids"].tolist()
        pending[tuple(ids)].append((row_idx, processed["loss_mask"]))
        prompts.append(ids)
        if cli.gate_dump and len(prompts) >= cli.gate_limit:
            break
    print(f"[export] tokenized {len(prompts)} kept / {num_skipped} skipped "
          f"in {time.time() - t_tok:.0f}s", flush=True)

    # ---- engine -----------------------------------------------------------
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    t_boot = time.time()
    llm = LLM(
        model=cli.model_path,
        tokenizer=tok_path,
        quantization="modelopt",
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=cli.tensor_parallel_size,
        max_model_len=cli.max_model_len,
        max_num_seqs=cli.max_num_seqs,
        max_num_batched_tokens=max(cli.max_num_batched_tokens, cli.max_model_len),
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        gpu_memory_utilization=cli.gpu_memory_utilization,
        # torch.compile/CUDA-graph capture both (a) blew up 1-card memory during
        # flashinfer autotune in P2 and (b) would bypass the python-level
        # GatedResidual patch this exporter depends on.
        enforce_eager=True,
    )
    boot_s = time.time() - t_boot
    print(f"[export] engine up in {boot_s:.0f}s", flush=True)

    top_module = _find_model(llm)
    model = _unwrap_causal_lm(top_module)
    backbone = model.model
    from vllm.models.qwen4_exp.nvidia.hyperconnection import GatedResidual
    _patch_gated_residual(GatedResidual)

    layers = backbone.layers
    for layer_id in target_layer_ids:
        tap_idx = layer_id + 1
        if tap_idx < len(layers) and getattr(layers[tap_idx], "attn_hyper_connection", None) is not None:
            layers[tap_idx].attn_hyper_connection._ds_tap = layer_id
        else:
            # Last-layer boundary: same contraction, done by the final mixer.
            backbone.hyper_connection_mixer._ds_tap = layer_id
    backbone.hyper_connection_mixer._ds_tap = "last"
    tagged = [layer_id for layer_id in target_layer_ids]
    print(f"[export] tagged taps {tagged} (+ final mixer as 'last')", flush=True)

    # ---- writer -----------------------------------------------------------
    writer = None
    rank_dir = None
    if output_dir is not None:
        os.makedirs(os.path.join(output_dir, "_tmp"), exist_ok=True)
        rank_dir = os.path.join(output_dir, "_tmp", f"rank_{cli.rank}")
        os.makedirs(rank_dir, exist_ok=True)
        writer = AsyncTargetCacheWriter(
            rank_dir=rank_dir, max_shard_bytes=int(cli.max_shard_bytes), max_queue_size=64
        )

    gate_store = {} if cli.gate_dump else None

    def _on_sample(row_idx, token_ids, hidden, last, loss_mask):
        if gate_store is not None:
            gate_store[int(row_idx)] = {
                "input_ids": token_ids,
                "loss_mask": loss_mask.clone(),
                "target_hidden_states": hidden,
                "target_last_hidden_states": last,
            }

    exporter = Exporter(
        taps=target_layer_ids, writer=writer, pending=pending,
        hidden_size=hidden_size, on_sample=_on_sample if gate_store is not None else None,
    )

    state = {"last": None, "t0": time.time(), "printed": 0}

    def _pre_hook(_module, args, kwargs):
        state["last"] = (
            args[0] if len(args) > 0 else kwargs.get("input_ids"),
            args[1] if len(args) > 1 else kwargs.get("positions"),
            kwargs.get("query_start_loc"),
        )
        return None

    def _post_hook(_module, _args, _kwargs, _output):
        input_ids, positions, qsl = state["last"]
        if input_ids is None or not _CAPTURE_ENABLED:
            _CAPTURED.clear()
            return None
        exporter.handle_forward(input_ids, positions, qsl)
        if exporter.num_written - state["printed"] >= cli.progress_every:
            state["printed"] = exporter.num_written
            dt = time.time() - state["t0"]
            print(f"[export] {exporter.num_written}/{len(prompts)} samples, "
                  f"{exporter.num_tokens} tokens, {exporter.num_tokens / max(dt, 1e-9):.0f} tok/s",
                  flush=True)
        return None

    # Hooks go on the TOP-LEVEL module the runner calls, never on
    # `Qwen4ExpModel`: that class carries `@support_torch_compile`, which
    # installs its own `__call__` on the class and therefore never runs
    # nn.Module._call_impl -- forward hooks registered there silently never
    # fire (observed: 20/20 requests processed, 0 captures). The top-level
    # wrapper is undecorated and receives the same
    # input_ids / positions / query_start_loc.
    h1 = top_module.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    h2 = top_module.register_forward_hook(_post_hook, with_kwargs=True)

    global _CAPTURE_ENABLED
    _CAPTURE_ENABLED = True

    sp_kwargs = dict(max_tokens=1, temperature=0.0, detokenize=False)
    if cli.gate_dump:
        # prompt_logprobs=1 (NOT 0). With 0, vLLM returns only the ACTUAL
        # prompt token's logprob, so "top-1" would really be "the corpus
        # token" and the check would measure teacher-vs-corpus agreement
        # (~0.83 measured) instead of the intended lm_head reconstruction.
        # With 1 the dict carries the engine's own argmax token as well.
        sp_kwargs["prompt_logprobs"] = 1
    sampling = SamplingParams(**sp_kwargs)

    t_run = time.time()
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in prompts],
        sampling,
    )
    run_s = time.time() - t_run
    _CAPTURE_ENABLED = False
    h1.remove()
    h2.remove()

    tok_s = exporter.num_tokens / max(run_s, 1e-9)
    print(f"[export] DONE {exporter.num_written} samples / {exporter.num_tokens} tokens "
          f"in {run_s:.0f}s = {tok_s:.0f} tok/s", flush=True)
    if pending:
        print(f"[export] WARNING {sum(len(v) for v in pending.values())} requests never matched",
              flush=True)

    # ---- gate mode: lm_head argmax vs vLLM prompt_logprobs -----------------
    if cli.gate_dump:
        lm_weight = model.lm_head.weight.detach()
        argmax_total = 0
        argmax_match = 0
        per_prompt = []
        by_ids = {}
        for out in outputs:
            by_ids[tuple(out.prompt_token_ids)] = out
        for row_idx, blob in gate_store.items():
            key = tuple(blob["input_ids"].tolist())
            out = by_ids.get(key)
            last = blob["target_last_hidden_states"].to(lm_weight.device, lm_weight.dtype)
            logits = torch.nn.functional.linear(last, lm_weight)
            mine = logits.argmax(dim=-1).tolist()
            # Full per-position reference top-1 (argmax(lm_head(last_hidden)))
            # for every token of this row -- independent of whether vLLM's own
            # prompt_logprobs happened to be available for the match check
            # below. This is the fixture's ground-truth comparison target.
            blob["lm_head_argmax_top1"] = torch.tensor(mine, dtype=torch.long)
            if out is None or not out.prompt_logprobs:
                continue
            ref = []
            keep = []
            for pos, entry in enumerate(out.prompt_logprobs):
                if entry is None:  # position 0 has no logprobs
                    continue
                top = min(entry.items(), key=lambda kv: -kv[1].logprob)[0]
                ref.append(int(top))
                keep.append(pos)
            # vLLM's prompt_logprobs[i] is the distribution that PRODUCED token
            # i, i.e. it is conditioned on hidden state i-1.
            match = sum(1 for j, pos in enumerate(keep) if mine[pos - 1] == ref[j])
            argmax_match += match
            argmax_total += len(keep)
            per_prompt.append({"row": int(row_idx), "n": len(keep),
                               "match": match, "rate": match / max(len(keep), 1)})
            blob["vllm_prompt_top1"] = torch.tensor(ref, dtype=torch.long)
            blob["vllm_prompt_top1_positions"] = torch.tensor(keep, dtype=torch.long)
        payload = {
            "features": {int(k): v for k, v in gate_store.items()},
            "target_layer_ids": target_layer_ids,
            "hidden_size": hidden_size,
            "lm_head_argmax": {
                "total_positions": argmax_total,
                "matched": argmax_match,
                "rate": argmax_match / max(argmax_total, 1),
                "per_prompt": per_prompt,
            },
            "tok_s": tok_s,
            "boot_s": boot_s,
        }
        os.makedirs(os.path.dirname(os.path.abspath(cli.gate_dump)), exist_ok=True)
        torch.save(payload, cli.gate_dump)
        print("[export] lm_head argmax vs vLLM prompt_logprobs: "
              + json.dumps(payload["lm_head_argmax"], indent=2)[:2000], flush=True)

    # ---- close ------------------------------------------------------------
    if writer is not None:
        writer.close()
        atomic_json_dump(exporter.source_row_ids,
                         os.path.join(rank_dir, "source_row_ids.json"))
        local_start, local_end = cli.start_row, cli.end_row
        summary = LocalCacheWriteSummary(
            global_rank=int(cli.rank),
            source_sample_start=int(local_start) + int(cli.rank),
            source_sample_end=int(local_end),
            num_local_samples=writer.num_local_samples,
            num_local_shards=len(writer.local_shard_files),
            local_shard_files=list(writer.local_shard_files),
        )
        atomic_json_dump(summary.to_json(), os.path.join(rank_dir, "summary.json"))

    receipt = {
        "status": "ok",
        "num_samples": exporter.num_written,
        "num_tokens": exporter.num_tokens,
        "num_skipped": num_skipped,
        "tok_s": tok_s,
        "boot_s": boot_s,
        "run_s": run_s,
        "segment_mismatches": exporter.segment_mismatches,
        "unmatched_requests": sum(len(v) for v in pending.values()),
        "rank": int(cli.rank),
        "world_size": int(cli.world_size),
    }
    if output_dir is not None:
        atomic_json_dump(receipt, os.path.join(output_dir, f"export_receipt_rank{cli.rank}.json"))
    print("[export] receipt " + json.dumps(receipt), flush=True)
    del llm
    gc.collect()


if __name__ == "__main__":
    main()
