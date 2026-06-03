"""Within-engine numerical-precision sweep for the vLLM inference stack.

This is the vLLM-side counterpart to `tools/evaluate_precision.py` (Fast-LLM) and
`tools/evaluate_precision_deepspeed.py` (HF + DeepSpeed). It loads a checkpoint once per
precision variant, feeds a fixed prompt, and reports the per-token chosen-token
log-probability against the vLLM fp32 reference, using the same metrics
(`CompareConfig._compute_diff`: RMS / bias / correlation / slope / residual).

vLLM is inference-only here, so there is no backward pass: only the forward log-π is measured.
For a prompt of tokens (t_0 .. t_{n-1}), vLLM's `prompt_logprobs[i]` is log P(t_i | t_0..t_{i-1});
slicing `[1:]` gives the n-1 next-token log-probs that align 1:1 with the trainers'
`chosen_logprob` (log-softmax of the logits at the next token).

The variants mirror the PipelineRL inference settings:

  * `bf16_fp32_head` is the production setting — `--quantization bf16_last_layer_fp32` (bf16 body, fp32
    LM head with fp32 logits), the vLLM analog of the trainers' fp32 LM head. The quant binds its fp32
    head by layer-name suffix; for tied-embedding models vLLM names the head `embed_tokens` rather than
    `lm_head`, so `--fp32-head-prefix auto` targets the right layer (otherwise the head silently no-ops
    on tied models, running in bf16);
  * `bf16` turns that off (head runs in bf16) to isolate the head's contribution.

Each variant runs in its own subprocess because vLLM's global CUDA/engine state does not tear
down cleanly in-process; a fresh process per variant guarantees isolation. Each worker writes
`output_dir/logprobs_<variant>.pt`; the parent loads them and prints the comparison table.

Run where vLLM (and, for the fp32-head variants, pipelinerl) is installed, e.g. the PipelineRL
stack image:

    python -m tools.evaluate_precision_vllm --model Qwen/Qwen2.5-0.5B --input-file <dir>/input_ids.pt
"""

import argparse
import logging
import pathlib
import subprocess
import sys
import typing

import torch

logger = logging.getLogger(__name__)

_REFERENCE_NAME = "fp32"
# (name, vLLM dtype, quantization). Reference is full fp32. `*_fp32_head` variants add the
# `bf16_last_layer_fp32` quantization (bf16/fp16 body, fp32 LM head + fp32 logits) — the
# production PipelineRL inference setting; the plain `bf16`/`fp16` variants run the head in the
# body dtype, isolating the fp32-head contribution.
# The bf16_last_layer_fp32 quantization only supports bf16/fp32 bodies (it rejects fp16), so there is
# no fp16_fp32_head variant.
_QUANTIZATION = "bf16_last_layer_fp32"
_VARIANTS: list[tuple[str, str, str | None]] = [
    (_REFERENCE_NAME, "float32", None),
    ("bf16", "bfloat16", None),
    ("bf16_fp32_head", "bfloat16", _QUANTIZATION),
    ("fp16", "float16", None),
]


def build_input_ids(model: str, sequence_length: int, text_file: str | None) -> torch.Tensor:
    """Build the fixed prompt, mirroring Fast-LLM's `tools/evaluate_precision.py:_prepare_input_ids`
    (same tokenizer + truncation for text, same seed-0 uniform-random tokens otherwise) so a standalone
    run is byte-identical to the trainers' and cross-engine-ready."""
    import transformers

    if text_file is not None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model)
        ids = tokenizer(pathlib.Path(text_file).read_text(), return_tensors="pt").input_ids[0]
        if ids.numel() < sequence_length:
            ids = ids.repeat((sequence_length + ids.numel() - 1) // ids.numel())
        return ids[:sequence_length].to(torch.int64)
    vocab_size = transformers.AutoConfig.from_pretrained(model).vocab_size
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, vocab_size, (sequence_length,), generator=generator, dtype=torch.int64)


def resolve_fp32_head_prefix(model: str, prefix: str) -> str:
    """The bf16_last_layer_fp32 quant binds its fp32 head by matching the layer-name suffix. For tied
    embeddings vLLM sets `lm_head = model.embed_tokens`, so the head is named `embed_tokens`, not
    `lm_head` — the production `lm_head` prefix then silently no-ops. Pick the suffix that matches the
    actual output head so the fp32 head genuinely binds on tied models too."""
    if prefix != "auto":
        return prefix
    import transformers

    tied = transformers.AutoConfig.from_pretrained(model).tie_word_embeddings
    return "embed_tokens" if tied else "lm_head"


def _next_token_logprobs(prompt_logprobs: list, token_ids: list[int]) -> torch.Tensor:
    """prompt_logprobs[i] is log P(t_i | t_0..t_{i-1}); [0] is None. Take the actual token at each position
    from i=1 on -> n-1 next-token log-probs aligned with the trainers' chosen_logprob."""
    return torch.tensor(
        [prompt_logprobs[i][token_ids[i]].logprob for i in range(1, len(token_ids))], dtype=torch.float32
    )


def run_worker(
    model: str,
    variant: str,
    dtype: str,
    quantization: str | None,
    attention_backend: str | None,
    fp32_head_prefix: str,
    input_file: str | None,
    output_dir: str,
    inputs_file: str | None = None,
) -> None:
    """Load the model at one precision variant, feed the fixed prompt(s), save per-token chosen log-π.

    Single-sequence (`input_file`): one prompt, saved as a flat log-π tensor. Multi-sequence (`inputs_file`,
    a dict with 'input_ids' = list of per-sequence token tensors): all sequences fed in one batched
    `generate` call, saved as a list of per-sequence log-π tensors."""
    import os

    # Force a single attention backend across all variants so the fp32-vs-bf16 diff reflects precision,
    # not a backend switch (vLLM otherwise picks flash-attn for bf16/fp16 but a Triton/flex backend for
    # fp32). TRITON_ATTN supports all three dtypes. Mirrors forcing sdpa on the DeepSpeed side.
    if attention_backend is not None:
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", attention_backend)
    if quantization is not None:
        os.environ["PIPELINERL_FP32_LAYER_PREFIX"] = fp32_head_prefix
        import pipelinerl.vllm_quantization  # noqa: F401  registers the bf16_last_layer_fp32 config

    import vllm

    if inputs_file is not None:
        token_id_lists = [tensor.flatten().to(torch.int64).tolist() for tensor in torch.load(inputs_file)["input_ids"]]
    else:
        token_id_lists = [torch.load(input_file).flatten().to(torch.int64).tolist()]

    llm = vllm.LLM(
        model=model,
        dtype=dtype,
        quantization=quantization,
        seed=0,
        enforce_eager=True,
        gpu_memory_utilization=0.9,
        max_model_len=max(len(token_ids) for token_ids in token_id_lists) + 16,
        enable_prefix_caching=False,
        logprobs_mode="processed_logprobs",
    )
    sampling_params = vllm.SamplingParams(temperature=1.0, max_tokens=1, prompt_logprobs=0)
    # vLLM returns outputs in input order, so zip pairs each output with its source token ids.
    outputs = llm.generate(
        prompts=[{"prompt_token_ids": token_ids} for token_ids in token_id_lists], sampling_params=sampling_params
    )
    vectors = [
        _next_token_logprobs(output.prompt_logprobs, token_ids)
        for output, token_ids in zip(outputs, token_id_lists, strict=True)
    ]
    path = pathlib.Path(output_dir) / f"logprobs_{variant}.pt"
    if inputs_file is not None:
        torch.save(vectors, path)
        logger.info(f"variant {variant}: saved {len(vectors)} per-sequence log π vectors -> {path}")
    else:
        torch.save(vectors[0], path)
        logger.info(
            f"variant {variant}: {vectors[0].numel()} tokens, scale {vectors[0].square().mean().sqrt():.4g} -> {path}"
        )


def _entry(tensor: torch.Tensor) -> dict[str, typing.Any]:
    return {"shape": list(tensor.shape), "step": 1, "samples": tensor}


def _print_table(title: str, by_variant: dict, cols: list[tuple[str, typing.Callable]]) -> None:
    name_width = max((len(n) for n in by_variant), default=7) + 1
    widths = [max(len(label), max((len(fn(v)) for v in by_variant.values()), default=0)) for label, fn in cols]
    print(f"\n=== vLLM: {title} ===")
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
    parser.add_argument(
        "--input-file",
        default=None,
        help="Path to an input_ids.pt saved by tools/evaluate_precision.py. When set, feeds that exact"
        " model input (byte-identical to the trainers); otherwise the input is built from"
        " --input-text-file or seed-0 random tokens.",
    )
    parser.add_argument(
        "--input-text-file",
        default=None,
        help="Tokenize this text file (same tokenizer + truncation as Fast-LLM) for a realistic-text"
        " prompt instead of random tokens. Ignored when --input-file is set.",
    )
    parser.add_argument(
        "--inputs-file",
        default=None,
        help="Path to a multi-sequence inputs.pt saved by tools/evaluate_precision.py (a dict with"
        " 'input_ids' = list of per-sequence token tensors). Feeds all sequences in one batched generate"
        " call and saves each variant's per-sequence log π list to `<output-dir>/logprobs_<variant>.pt`"
        " (the within-engine table is skipped; use the cross-engine tool). Overrides --input-file.",
    )
    parser.add_argument("--output-dir", default="/tmp/fast_llm_tests/evaluate_precision/vllm")
    parser.add_argument(
        "--attention-backend",
        default="TRITON_ATTN",
        help="vLLM attention backend forced for every variant (isolates precision from the kernel)."
        " Pass 'auto' to let vLLM pick per dtype (the production path: flash-attn for bf16/fp16).",
    )
    parser.add_argument(
        "--fp32-head-prefix",
        default="auto",
        help="Layer-name suffix the bf16_last_layer_fp32 quant matches for its fp32 head. 'auto' picks"
        " embed_tokens for tied-embedding models / lm_head otherwise so the fp32 head actually binds."
        " Pass 'lm_head' for the literal production setting (a no-op on tied models).",
    )
    # Internal: a single-variant worker invocation (one vLLM engine per process).
    parser.add_argument("--worker-variant", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-dtype", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-quantization", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attention_backend = None if args.attention_backend == "auto" else args.attention_backend
    fp32_head_prefix = resolve_fp32_head_prefix(args.model, args.fp32_head_prefix)
    if args.worker_variant is not None:
        quantization = args.worker_quantization or None
        run_worker(
            args.model,
            args.worker_variant,
            args.worker_dtype,
            quantization,
            attention_backend,
            fp32_head_prefix,
            args.input_file,
            args.output_dir,
            args.inputs_file,
        )
        return

    if args.inputs_file is not None:
        logger.info(f"Multi-sequence: using shared sequence set {args.inputs_file}")
        input_file = None
    elif args.input_file is not None:
        input_file = args.input_file
        logger.info(f"Using shared model input {input_file}")
    else:
        input_ids = build_input_ids(args.model, args.sequence_length, args.input_text_file).unsqueeze(0)
        input_file = str(output_dir / "input_ids.pt")
        torch.save(input_ids, input_file)
        kind = "text" if args.input_text_file is not None else "seed-0 random"
        logger.info(f"Generated {kind} input {tuple(input_ids.shape)} -> {input_file}")

    for name, dtype, quantization in _VARIANTS:
        logger.info(f"=== variant {name} (dtype={dtype}, quantization={quantization}) ===")
        cmd = [
            sys.executable,
            "-m",
            "tools.evaluate_precision_vllm",
            "--model",
            args.model,
            "--output-dir",
            args.output_dir,
            "--attention-backend",
            args.attention_backend,
            "--fp32-head-prefix",
            fp32_head_prefix,
            "--worker-variant",
            name,
            "--worker-dtype",
            dtype,
        ]
        cmd += ["--inputs-file", args.inputs_file] if args.inputs_file is not None else ["--input-file", input_file]
        if quantization is not None:
            cmd += ["--worker-quantization", quantization]
        subprocess.run(cmd, check=True)

    if args.inputs_file is not None:
        # Multi-sequence: per-sequence log π lists are saved; the cross-engine tool does the analysis.
        logger.info("Multi-sequence vLLM run complete; compare with tools/evaluate_precision_cross_engine.py.")
        return

    from fast_llm.engine.config_utils.compare_tensor_logs import CompareConfig

    compare = CompareConfig()
    ref = torch.load(output_dir / f"logprobs_{_REFERENCE_NAME}.pt")
    metrics: dict[str, dict[str, typing.Any]] = {}
    for name, _, _ in _VARIANTS:
        logprob = torch.load(output_dir / f"logprobs_{name}.pt")
        metrics[name] = compare._compute_diff(_entry(ref), _entry(logprob), "step", "chosen_logprob")

    cols = [
        ("RMS rel", lambda m: f"{m['rms_rel'] * 100:.4f}%"),
        ("Bias rel", lambda m: f"{m['bias_rel'] * 100:+.4f}%"),
        ("Resid rel", lambda m: f"{m['residual_rms_rel'] * 100:.4f}%"),
        ("Corr", lambda m: f"{m['correlation']:.5f}"),
        ("Slope", lambda m: f"{m['slope']:+.5f}"),
        ("Max abs", lambda m: f"{m['max_abs']:.4g}"),
        ("Scale", lambda m: f"{m['ref_scale']:.4g}"),
    ]
    _print_table("chosen_logprob (per-token) vs fp32 reference", metrics, cols)


if __name__ == "__main__":
    main()
