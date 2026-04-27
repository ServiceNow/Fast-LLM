#!/usr/bin/env python3
"""Tiny Gemma4 HF-vs-Fast-LLM forward parity smoke test."""

import argparse
import pathlib
import shutil
import sys
import warnings

import torch
from transformers import Gemma4ForCausalLM, Gemma4TextConfig

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fast_llm.engine.checkpoint.config import CheckpointLoadConfig
from fast_llm.engine.multi_stage.config import StageMode
from fast_llm.models.gpt.conversion.config import Gemma4CheckpointFormat
from fast_llm.models.gpt.huggingface import HuggingfaceGPTModelForCausalLM


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-path",
        type=pathlib.Path,
        default=None,
        help="Existing HF Gemma4 text checkpoint to compare. If omitted, a tiny checkpoint is generated.",
    )
    parser.add_argument(
        "--work-dir",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/fastllm_gemma4_forward_parity"),
        help="Temporary directory for the generated HF checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Compute dtype for both HF and Fast-LLM.",
    )
    parser.add_argument(
        "--input-ids",
        default="1,5,7,9,11,13,17,19",
        help="Comma-separated token ids for the parity prompt.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for the parity run. 'auto' uses CUDA when available.",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep the generated checkpoint directory after the run.",
    )
    parser.add_argument(
        "--sequential-load",
        action="store_true",
        help="Run HF first, free it, then load Fast-LLM. Saves GPU memory for large checkpoints.",
    )
    parser.add_argument(
        "--mlp-implementation",
        choices=("auto", "dropless", "looped"),
        default=None,
        help="Override Gemma4 MoE implementation. Defaults to looped for generated tiny models and auto otherwise.",
    )
    parser.add_argument(
        "--fast-llm-mode",
        choices=("training", "inference"),
        default=None,
        help="Fast-LLM stage mode. Defaults to training for generated tiny models and inference otherwise.",
    )
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(name)


def _make_tiny_config() -> Gemma4TextConfig:
    config = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=24,
        moe_intermediate_size=8,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=8,
        global_head_dim=16,
        num_experts=4,
        top_k_experts=2,
        sliding_window=4,
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
        tie_word_embeddings=True,
        attention_k_eq_v=True,
        enable_moe_block=True,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        final_logit_softcapping=30.0,
        pad_token_id=0,
        attention_dropout=0.0,
        use_bidirectional_attention="vision",
        hidden_size_per_layer_input=0,
    )
    config.architectures = ["Gemma4ForCausalLM"]
    return config


def _load_fast_llm_model(
    source_path: pathlib.Path,
    device: torch.device,
    compute_dtype: str,
    mlp_implementation: str | None,
    mode: StageMode,
) -> HuggingfaceGPTModelForCausalLM:
    updates = {
        ("distributed", "use_cuda"): device.type == "cuda",
        ("distributed", "compute_dtype"): compute_dtype,
    }
    if mlp_implementation is not None:
        updates.update(
            {
                ("base_model", "decoder", "blocks", "sliding", "mlp", "implementation"): mlp_implementation,
                ("base_model", "decoder", "blocks", "full", "mlp", "implementation"): mlp_implementation,
            }
        )
    return HuggingfaceGPTModelForCausalLM.from_pretrained(
        CheckpointLoadConfig(path=source_path, format=Gemma4CheckpointFormat, model_weights=True),
        updates,
        mode=mode,
    ).eval()


def main() -> None:
    args = _parse_args()
    device = _device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    torch.manual_seed(args.seed)
    if args.source_path is None:
        if args.work_dir.exists():
            shutil.rmtree(args.work_dir)
        source_path = args.work_dir / "source"
        hf_model = Gemma4ForCausalLM(_make_tiny_config()).eval()
        hf_model.save_pretrained(source_path)
    else:
        source_path = args.source_path
        hf_model = Gemma4ForCausalLM.from_pretrained(source_path, torch_dtype=dtype).eval()

    mlp_implementation = args.mlp_implementation
    if mlp_implementation is None:
        # Tiny dimensions are not always compatible with dropless Triton tile constraints.
        mlp_implementation = "looped" if args.source_path is None else "auto"
    mode = StageMode(args.fast_llm_mode) if args.fast_llm_mode is not None else (
        StageMode.training if args.source_path is None else StageMode.inference
    )

    hf_model = hf_model.to(device=device, dtype=dtype)

    input_ids = torch.tensor([[int(token_id) for token_id in args.input_ids.split(",")]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        if args.sequential_load:
            hf_logits = hf_logits.cpu()
            del hf_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        fast_llm_model = _load_fast_llm_model(source_path, device, args.dtype, mlp_implementation, mode)
        fast_llm_logits = fast_llm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        if args.sequential_load:
            fast_llm_logits = fast_llm_logits.cpu()

    diff = (hf_logits - fast_llm_logits).float()
    max_abs = diff.abs().max().item()
    rms = torch.sqrt(torch.mean(diff.square())).item()
    passed = torch.allclose(hf_logits, fast_llm_logits, atol=args.atol, rtol=args.rtol)

    print(f"device: {device}")
    print(f"logits shape: {tuple(hf_logits.shape)}")
    print(f"max_abs_diff: {max_abs:.8g}")
    print(f"rms_diff: {rms:.8g}")
    print(f"tolerances: atol={args.atol:g}, rtol={args.rtol:g}")
    print("PASS" if passed else "FAIL")

    if args.source_path is None and not args.keep_work_dir:
        shutil.rmtree(args.work_dir)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*GenerationMixin.*")
        warnings.filterwarnings("ignore", message=".*use_return_dict.*")
        main()
