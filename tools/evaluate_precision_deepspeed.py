"""Within-engine numerical-precision sweep for the HF-transformers + DeepSpeed stack.

This is the DeepSpeed-side counterpart to `tools/evaluate_precision.py` (which measures the
same thing inside Fast-LLM). It loads a HF checkpoint, runs one forward + backward per precision
variant through a DeepSpeed engine, and reports two quantities against the fp32 reference, using
the same metrics (`CompareConfig._compute_diff`: RMS / bias / correlation / slope / residual):

  * chosen-token log-probability per position (the RL importance-ratio input);
  * parameter gradients, aggregated by category (embedding/head, linear, norm, bias).

The point is to check whether Fast-LLM's bf16 loses precision the *same way* DeepSpeed's bf16
does — each measured against its own fp32 reference.

The log-π computation and the fp32 LM-head mechanism mirror PipelineRL's DeepSpeed trainer
(`pipelinerl/finetune/rl/__init__.py` and `pipelinerl/finetune/checkpoints.py`) so the numbers
reflect the proven baseline rather than a bespoke path. `param.grad` is populated and already
unscaled after `engine.backward` (verified for both bf16 and fp16), so gradients are read directly.

Run where transformers + deepspeed are installed (e.g. the PipelineRL stack image):

    python -m tools.evaluate_precision_deepspeed --model Qwen/Qwen2.5-0.5B --sequence-length 2048
"""

import argparse
import functools
import logging
import os
import pathlib
import statistics
import typing

import torch

logger = logging.getLogger(__name__)

_REFERENCE_NAME = "fp32"
# (name, compute dtype, fp32 lm head). Reference is fp32 + fp32 head. `*_head_<dtype>` variants
# turn the fp32 head OFF (head runs in compute dtype) to reproduce the within-engine
# "fp32 lm head has ~no effect" finding on the DeepSpeed side.
_VARIANTS: list[tuple[str, torch.dtype, bool]] = [
    (_REFERENCE_NAME, torch.float32, True),
    ("bf16", torch.bfloat16, True),
    ("bf16_head_bf16", torch.bfloat16, False),
    ("fp16", torch.float16, True),
    ("fp16_head_fp16", torch.float16, False),
]

_FIXED_TEXT = (
    "The numerical precision of large language model training depends on the dtype used for "
    "matrix multiplications, the accumulation precision of the hardware, and whether the output "
    "projection is kept in full precision. In reinforcement learning from human feedback, the "
    "importance ratio between the new and old policy is the exponential of the difference of "
    "log-probabilities, so even small per-token errors in the log-probability can compound. "
    "We compute the chosen-token log-probability as the log-softmax of the logits evaluated at "
    "the next token, and we compare bfloat16 and float16 against a float32 reference. "
)


def apply_fp32_lm_head(model: torch.nn.Module, layer_prefix: str = "lm_head") -> torch.nn.Module:
    """Cast the LM head to fp32 at compute time. Mirrors PipelineRL `apply_fp32_lm_head`.

    For tied embeddings (e.g. Qwen2.5-0.5B) the weight storage stays in the model dtype and is
    upcast only for the head matmul; for untied heads the storage itself is moved to fp32.
    """
    lm_head = model.get_output_embeddings()
    if lm_head is None or not isinstance(lm_head, torch.nn.Linear):
        raise RuntimeError(f"Could not find an nn.Linear LM head via get_output_embeddings(): {lm_head!r}")
    tied = False
    inp_emb = model.get_input_embeddings()
    if inp_emb is not None and hasattr(inp_emb, "weight"):
        tied = lm_head.weight is inp_emb.weight
    if not tied and lm_head.weight.dtype != torch.float32:
        lm_head.to(dtype=torch.float32)
    original_forward = lm_head.forward

    @functools.wraps(original_forward)
    def fp32_forward(x: torch.Tensor) -> torch.Tensor:
        x32 = x if x.dtype == torch.float32 else x.float()
        w = lm_head.weight
        w32 = w if w.dtype == torch.float32 else w.float()
        b = lm_head.bias
        b32 = b.float() if (b is not None and b.dtype != torch.float32) else b
        return torch.nn.functional.linear(x32, w32, b32)

    lm_head.forward = fp32_forward
    logger.info(f"Applied fp32 lm head (tied={tied})")
    return model


def chosen_logprob(logits: torch.Tensor, input_ids: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """log π(next token) per position. Mirrors PipelineRL `rl/__init__.py:203-208`."""
    logits = logits[:, :-1, :].float() / temperature
    next_ids = input_ids[:, 1:].unsqueeze(-1)
    selected = torch.gather(logits, 2, next_ids).squeeze(-1)
    log_norm = torch.logsumexp(logits, dim=-1)
    return (selected - log_norm).reshape(-1)


def build_input_ids(tokenizer, vocab_size: int, sequence_length: int, device: torch.device, mode: str) -> torch.Tensor:
    if mode == "random":
        # Match Fast-LLM's random dataset (uniform token ids over the model vocab) so both engines
        # see the same input distribution. The relative metrics depend strongly on it: on random
        # tokens the model is maximally surprised (|log π| large), on realistic text |log π| ≈ 0,
        # which shifts the relative RMS by several-fold even at identical absolute precision.
        generator = torch.Generator().manual_seed(0)
        ids = torch.randint(0, vocab_size, (sequence_length,), generator=generator)
    else:
        ids = tokenizer(_FIXED_TEXT, return_tensors="pt").input_ids[0]
        repeats = (sequence_length + ids.numel() - 1) // ids.numel()
        ids = ids.repeat(repeats)[:sequence_length]
    return ids.unsqueeze(0).to(device)


def _ds_config(dtype: torch.dtype, forward_only: bool = False) -> dict[str, typing.Any]:
    config: dict[str, typing.Any] = {"train_micro_batch_size_per_gpu": 1}
    if not forward_only:
        # No optimizer for forward-only: avoids the fp32 master copy + Adam state that would OOM a 7B run.
        config["optimizer"] = {"type": "Adam", "params": {"lr": 1e-6}}
    if dtype == torch.bfloat16:
        config["bf16"] = {"enabled": True}
    elif dtype == torch.float16:
        config["fp16"] = {"enabled": True, "initial_scale_power": 16}
    return config


def grad_category(name: str) -> str:
    if name.endswith(".bias"):
        return "bias"
    if "layernorm" in name or name.endswith("norm.weight"):
        return "norm"
    if "embed_tokens" in name or "lm_head" in name:
        return "embed_head"
    return "linear"


def capture_variant(
    model_id: str,
    dtype: torch.dtype,
    fp32_head: bool,
    input_ids: torch.Tensor,
    attn_implementation: str,
    random_init: bool = False,
    forward_only: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Capture one variant. Returns (chosen_logprob, {param_name: gradient}), both on CPU in fp32.

    Default: forward + backward through a DeepSpeed engine. `forward_only=True` initializes the same
    DeepSpeed engine but without an optimizer (no fp32 master copy / Adam state, which would OOM a 7B
    run) and runs a single `eval()` + `no_grad` forward (returns empty gradients). The engine is kept —
    DeepSpeed's bf16/fp16 forward is not bit-identical to a plain HF forward in the same dtype, so
    bypassing it would shift the measured log π."""
    import deepspeed
    import transformers

    if random_init:
        model = transformers.AutoModelForCausalLM.from_config(
            transformers.AutoConfig.from_pretrained(model_id), dtype=dtype, attn_implementation=attn_implementation
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, attn_implementation=attn_implementation
        )
    if fp32_head:
        apply_fp32_lm_head(model)
    if forward_only:
        engine, *_ = deepspeed.initialize(model=model, config=_ds_config(dtype, forward_only=True))
        engine.eval()
        with torch.no_grad():
            logprob = chosen_logprob(engine(input_ids).logits, input_ids).detach().float().cpu()
        del engine, model
        torch.cuda.empty_cache()
        return logprob, {}

    engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=_ds_config(dtype))
    outputs = engine(input_ids)
    logprob = chosen_logprob(outputs.logits, input_ids)
    # fp16's narrow exponent range underflows small gradients; scale the loss up before backward and
    # divide it back out (loss scaling, as in fp16 training). bf16/fp32 have fp32 range, no scaling.
    # engine.backward leaves param.grad unscaled, so dividing by our own loss_scale recovers the true
    # gradient computed with extra headroom against underflow.
    loss_scale = 256.0 if dtype == torch.float16 else 1.0
    engine.backward(-logprob.mean() * loss_scale)
    grads = {
        name: (p.grad.detach().float() / loss_scale).cpu()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    logprob = logprob.detach().float().cpu()
    del engine, model, outputs
    torch.cuda.empty_cache()
    return logprob, grads


def capture_variant_multi(
    model_id: str,
    dtype: torch.dtype,
    fp32_head: bool,
    sequences: list[torch.Tensor],
    attn_implementation: str,
    random_init: bool = False,
) -> list[torch.Tensor]:
    """Forward-only over many independent sequences on one resident DeepSpeed engine (no optimizer, so no
    fp32 master copy / Adam state). Returns one per-token chosen_logprob vector per sequence, on CPU in
    fp32. The engine is kept (not bypassed for plain HF) because DeepSpeed's bf16/fp16 forward is not
    bit-identical to a plain HF forward in the same dtype."""
    import deepspeed
    import transformers

    if random_init:
        model = transformers.AutoModelForCausalLM.from_config(
            transformers.AutoConfig.from_pretrained(model_id), dtype=dtype, attn_implementation=attn_implementation
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, attn_implementation=attn_implementation
        )
    if fp32_head:
        apply_fp32_lm_head(model)
    engine, *_ = deepspeed.initialize(model=model, config=_ds_config(dtype, forward_only=True))
    engine.eval()
    vectors: list[torch.Tensor] = []
    with torch.no_grad():
        for sequence in sequences:
            input_ids = sequence.unsqueeze(0)
            vectors.append(chosen_logprob(engine(input_ids).logits, input_ids).detach().float().cpu())
    del engine, model
    torch.cuda.empty_cache()
    return vectors


def _entry(tensor: torch.Tensor) -> dict[str, typing.Any]:
    return {"shape": list(tensor.shape), "step": 1, "samples": tensor}


def _print_logprob_summary(metrics_by_variant: dict[str, dict[str, typing.Any]]) -> None:
    cols = [
        ("RMS rel", lambda m: f"{m['rms_rel'] * 100:.4f}%"),
        ("Bias rel", lambda m: f"{m['bias_rel'] * 100:+.4f}%"),
        ("Resid rel", lambda m: f"{m['residual_rms_rel'] * 100:.4f}%"),
        ("Corr", lambda m: f"{m['correlation']:.5f}"),
        ("Slope", lambda m: f"{m['slope']:+.5f}"),
        ("Max abs", lambda m: f"{m['max_abs']:.4g}"),
        ("Scale", lambda m: f"{m['ref_scale']:.4g}"),
    ]
    _print_table("chosen_logprob (per-token) vs fp32 reference", metrics_by_variant, cols)


def _print_grad_summary(grad_metrics_by_variant: dict[str, dict[str, list[float]]]) -> None:
    # Per-category aggregation of gradient RMS-rel, mirroring tools/evaluate_precision.py's grad table.
    def med(values: list[float]) -> str:
        return f"{statistics.median(values) * 100:.4f}%" if values else "n/a"

    def mx(values: list[float]) -> str:
        return f"{max(values) * 100:.4f}%" if values else "n/a"

    cols = [
        ("embed_head", lambda c: med(c.get("embed_head", []))),
        ("linear med", lambda c: med(c.get("linear", []))),
        ("linear max", lambda c: mx(c.get("linear", []))),
        ("norm med", lambda c: med(c.get("norm", []))),
        ("norm max", lambda c: mx(c.get("norm", []))),
        ("bias med", lambda c: med(c.get("bias", []))),
        ("bias max", lambda c: mx(c.get("bias", []))),
    ]
    _print_table("gradient RMS-rel by category vs fp32 reference", grad_metrics_by_variant, cols)


def _print_table(title: str, by_variant: dict, cols: list[tuple[str, typing.Callable]]) -> None:
    name_width = max((len(n) for n in by_variant), default=7) + 1
    widths = [max(len(label), max((len(fn(v)) for v in by_variant.values()), default=0)) for label, fn in cols]
    print(f"\n=== DeepSpeed/HF: {title} ===")
    header = f"{'Variant':<{name_width}}" + " ".join(
        f"{label:<{w}}" for (label, _), w in zip(cols, widths, strict=True)
    )
    print(header)
    print("-" * len(header))
    for name, value in by_variant.items():
        cells = [fn(value) for _, fn in cols]
        print(f"{name:<{name_width}}" + " ".join(f"{c:<{w}}" for c, w in zip(cells, widths, strict=True)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--input-mode", choices=["random", "text"], default="random")
    parser.add_argument(
        "--input-file",
        default=None,
        help="Path to an input_ids.pt saved by tools/evaluate_precision.py. When set, feeds that exact"
        " model input (so Fast-LLM and DeepSpeed see byte-identical tokens); --input-mode is ignored.",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Build the model from config with random weights instead of loading the pretrained"
        " checkpoint (contrast with the pretrained run; weights won't match Fast-LLM's random init).",
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Initialize the DeepSpeed engine without an optimizer and run a single eval()+no_grad"
        " forward (no optimizer state, no gradients). Fits large models (e.g. 7B fp32) where"
        " forward+backward+Adam would OOM. The gradient table is then empty.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="If set, save each variant's full per-token log π vector to"
        " `<output-dir>/logprobs_<variant>.pt` (plain fp32 CPU tensor, aligned 1:1 with vLLM's"
        " `prompt_logprobs[1:]`) for the cross-engine comparison.",
    )
    parser.add_argument(
        "--inputs-file",
        default=None,
        help="Path to a multi-sequence inputs.pt saved by tools/evaluate_precision.py (a dict with"
        " 'input_ids' = list of per-sequence token tensors). Runs one forward-only pass per sequence and"
        " saves each variant's per-sequence log π list to `<output-dir>/logprobs_<variant>.pt`. Requires"
        " --output-dir; --input-file / --input-mode / gradient table are ignored.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for key, value in (
        ("RANK", "0"),
        ("LOCAL_RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", "29555"),
    ):
        os.environ.setdefault(key, value)

    import transformers

    from fast_llm.engine.config_utils.compare_tensor_logs import CompareConfig

    device = torch.device("cuda:0")
    if args.inputs_file is not None:
        assert args.output_dir is not None, "--inputs-file requires --output-dir"
        data = torch.load(args.inputs_file)
        sequences = [tensor.to(device=device, dtype=torch.int64) for tensor in data["input_ids"]]
        output_dir = pathlib.Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Multi-sequence: {len(sequences)} sequences from {args.inputs_file}")
        for name, dtype, fp32_head in _VARIANTS:
            logger.info(f"=== variant {name} (dtype={dtype}, fp32_head={fp32_head}) ===")
            vectors = capture_variant_multi(
                args.model, dtype, fp32_head, sequences, args.attn_implementation, args.random_init
            )
            torch.save(vectors, output_dir / f"logprobs_{name}.pt")
            logger.info(f"variant {name}: saved {len(vectors)} per-sequence log π vectors")
        return

    if args.input_file is not None:
        input_ids = torch.load(args.input_file).to(device=device, dtype=torch.int64)
        logger.info(f"Loaded shared model input {tuple(input_ids.shape)} from {args.input_file}")
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        vocab_size = transformers.AutoConfig.from_pretrained(args.model).vocab_size
        input_ids = build_input_ids(tokenizer, vocab_size, args.sequence_length, device, args.input_mode)
    logger.info(f"input_ids shape {tuple(input_ids.shape)}")

    compare = CompareConfig()
    ref_logprob: torch.Tensor | None = None
    ref_grads: dict[str, torch.Tensor] = {}
    logprob_metrics: dict[str, dict[str, typing.Any]] = {}
    grad_metrics: dict[str, dict[str, list[float]]] = {}
    for name, dtype, fp32_head in _VARIANTS:
        logger.info(f"=== variant {name} (dtype={dtype}, fp32_head={fp32_head}) ===")
        logprob, grads = capture_variant(
            args.model, dtype, fp32_head, input_ids, args.attn_implementation, args.random_init, args.forward_only
        )
        if args.output_dir is not None:
            output_dir = pathlib.Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(logprob, output_dir / f"logprobs_{name}.pt")
        if name == _REFERENCE_NAME:
            ref_logprob, ref_grads = logprob, grads
        logprob_metrics[name] = compare._compute_diff(_entry(ref_logprob), _entry(logprob), "step", "chosen_logprob")
        by_category: dict[str, list[float]] = {}
        for param_name, grad in grads.items():
            if param_name not in ref_grads:
                continue
            metrics = compare._compute_diff(_entry(ref_grads[param_name]), _entry(grad), "step", param_name)
            by_category.setdefault(grad_category(param_name), []).append(metrics["rms_rel"])
        grad_metrics[name] = by_category

    _print_logprob_summary(logprob_metrics)
    _print_grad_summary(grad_metrics)


if __name__ == "__main__":
    main()
