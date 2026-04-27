#!/usr/bin/env python3
"""Tiny Gemma4 HF-vs-Fast-LLM decoder-layer forward/backward parity check."""

import argparse
import pathlib
import sys
import warnings
from dataclasses import dataclass

import torch
from transformers import Gemma4ForCausalLM, Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import create_causal_mask, create_sliding_window_causal_mask

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fast_llm.data.document.config import LanguageModelBatchPreprocessingConfig
from fast_llm.data.document.language_model import LanguageModelBatch
from fast_llm.engine.base_model.config import set_model_names
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import StageConfig
from fast_llm.engine.multi_stage.stage import Stage
from fast_llm.functional.config import TritonConfig
from fast_llm.layers.decoder.block import DecoderBlock
from fast_llm.layers.decoder.config import DecoderBlockConfig
from fast_llm.models.gpt.conversion.gemma4 import Gemma4BlockConverter


@dataclass(frozen=True)
class Diff:
    max_abs: float
    rms: float
    passed: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Device for the parity run. CUDA is the default because this is a training smoke test.",
    )
    parser.add_argument(
        "--layers",
        choices=("both", "sliding", "full"),
        default="both",
        help="Which representative decoder layers to check.",
    )
    parser.add_argument(
        "--disable-triton",
        action="store_true",
        help="Disable optional Triton kernels for an easier-to-debug PyTorch reference path.",
    )
    return parser.parse_args()


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
    config._attn_implementation = "eager"
    return config


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(name)


def _setup_stage(block: DecoderBlock, distributed: Distributed) -> Stage:
    if not block._is_setup:
        block.setup(distributed)
    set_model_names(torch.nn.ModuleList([block]))
    stage = Stage(
        config=StageConfig(),
        layers=[block],
        distributed_config=distributed.config,
        index=0,
        tied_parameter_duplicates=(),
    )
    stage.setup(distributed=distributed, tied_parameter_duplicate_buffers=None)
    stage.initialize_weights()
    stage.restore_parameters()
    stage.reset_gradients()
    return stage


def _make_fast_block(config: Gemma4TextConfig, layer_type: str, device: torch.device) -> DecoderBlock:
    config_dict = Gemma4BlockConverter.import_config(
        config.to_dict(),
        full_attention=layer_type == "full_attention",
    )
    config_dict["mixer"]["implementation"] = "backup"
    config_dict["mlp"]["implementation"] = "looped"
    block_config = DecoderBlockConfig._from_dict(config_dict)
    distributed_config = DistributedConfig(compute_dtype="float32", use_cuda=device.type == "cuda")
    block = block_config.get_layer(
        distributed_config,
        TensorDim("hidden_size", config.hidden_size),
        lr_scale=None,
        peft=None,
    )
    _setup_stage(block, Distributed(distributed_config))
    return block.train()


def _copy_hf_layer_to_fast_block(hf_layer: torch.nn.Module, fast_block: DecoderBlock) -> list:
    hf_state = hf_layer.state_dict()
    fast_params = dict(fast_block.named_parameters())
    converters = Gemma4BlockConverter.get_converters(fast_block._config, "fast", "hf")

    with torch.no_grad():
        for converter in converters:
            if not converter.fast_llm_name:
                continue
            hf_names = [name.removeprefix("hf.") for name in converter.export_name]
            fast_names = [name.removeprefix("fast.") for name in converter.fast_llm_name]
            imported = converter.import_weight(tuple(hf_state[name] for name in hf_names))
            for fast_name, tensor in zip(fast_names, imported, strict=True):
                fast_params[fast_name].copy_(tensor.to(device=fast_params[fast_name].device))

    return converters


def _fast_kwargs(seq_len: int, batch_size: int, distributed_config: DistributedConfig, device: torch.device) -> dict:
    lengths = [seq_len] * batch_size
    tokens = torch.empty(seq_len * batch_size, dtype=torch.int64, device=device)
    (model_input,) = LanguageModelBatch(tokens=tokens, lengths=lengths).get_model_inputs(
        LanguageModelBatchPreprocessingConfig(
            distributed=distributed_config,
            predicted_tokens=0,
            return_document_index=True,
        )
    )
    kwargs = model_input.to_kwargs()
    return kwargs


def _hf_attention_mask(
    config: Gemma4TextConfig,
    inputs: torch.Tensor,
    position_ids: torch.Tensor,
    layer_type: str,
) -> torch.Tensor | None:
    mask_kwargs = {
        "config": config,
        "inputs_embeds": inputs,
        "attention_mask": torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device),
        "past_key_values": None,
        "position_ids": position_ids,
    }
    if layer_type == "full_attention":
        return create_causal_mask(**mask_kwargs)
    return create_sliding_window_causal_mask(**mask_kwargs)


def _grad(parameter: torch.nn.Parameter) -> torch.Tensor | None:
    if not getattr(parameter, "param_grad_is_zero", True):
        return parameter.grad_buffer
    if parameter.grad is not None:
        return parameter.grad
    return getattr(parameter, "grad_buffer", None)


def _diff(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> Diff:
    delta = (actual - expected).float()
    return Diff(
        max_abs=delta.abs().max().item(),
        rms=torch.sqrt(torch.mean(delta.square())).item(),
        passed=torch.allclose(actual, expected, atol=atol, rtol=rtol),
    )


def _print_diff(name: str, diff: Diff) -> None:
    status = "PASS" if diff.passed else "FAIL"
    print(f"  {name}: {status} max_abs={diff.max_abs:.8g} rms={diff.rms:.8g}")


def _compare_parameter_grads(
    converters: list,
    hf_layer: torch.nn.Module,
    fast_block: DecoderBlock,
    atol: float,
    rtol: float,
) -> tuple[int, list[tuple[str, Diff]]]:
    hf_grads = {name: parameter.grad for name, parameter in hf_layer.named_parameters()}
    fast_grads = {name: _grad(parameter) for name, parameter in fast_block.named_parameters()}
    failures: list[tuple[str, Diff]] = []
    checked = 0

    for converter in converters:
        if not converter.fast_llm_name:
            continue
        fast_names = [name.removeprefix("fast.") for name in converter.fast_llm_name]
        hf_names = [name.removeprefix("hf.") for name in converter.export_name]
        if any(fast_grads[name] is None for name in fast_names):
            raise RuntimeError(f"Missing Fast-LLM gradient for {converter.fast_llm_name}")
        exported = converter.export_weight(tuple(fast_grads[name] for name in fast_names))
        for hf_name, fast_grad in zip(hf_names, exported, strict=True):
            hf_grad = hf_grads.get(hf_name)
            if hf_grad is None:
                raise RuntimeError(f"Missing HF gradient for {hf_name}")
            grad_diff = _diff(fast_grad.to(hf_grad.device), hf_grad, atol, rtol)
            checked += 1
            if not grad_diff.passed:
                failures.append((hf_name, grad_diff))

    return checked, failures


def _run_layer(
    hf_model: Gemma4ForCausalLM,
    layer_index: int,
    layer_type: str,
    args: argparse.Namespace,
    device: torch.device,
) -> bool:
    config = hf_model.config
    hf_layer = hf_model.model.layers[layer_index].train()
    fast_block = _make_fast_block(config, layer_type, device)
    converters = _copy_hf_layer_to_fast_block(hf_layer, fast_block)
    fast_block.preprocess(kwargs := _fast_kwargs(args.seq_len, args.batch_size, fast_block._distributed_config, device))

    total_tokens = args.batch_size * args.seq_len
    base = torch.randn(total_tokens, config.hidden_size, dtype=torch.float32, device=device)
    hf_input = base.view(args.batch_size, args.seq_len, config.hidden_size).detach().clone().requires_grad_(True)
    fast_input = base.detach().clone().requires_grad_(True)
    position_ids = torch.arange(args.seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(args.batch_size, -1)
    position_embeddings = hf_model.model.rotary_emb(hf_input, position_ids, layer_type)
    attention_mask = _hf_attention_mask(config, hf_input, position_ids, layer_type)

    hf_output = hf_layer(
        hf_input,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=None,
    )
    fast_output = fast_block(fast_input, kwargs)

    output_diff = _diff(fast_output, hf_output.reshape(total_tokens, config.hidden_size), args.atol, args.rtol)
    output_grad = torch.randn_like(hf_output)
    hf_output.backward(output_grad)
    fast_output.backward(output_grad.reshape(total_tokens, config.hidden_size))

    input_grad_diff = _diff(fast_input.grad, hf_input.grad.reshape(total_tokens, config.hidden_size), args.atol, args.rtol)
    checked_grads, grad_failures = _compare_parameter_grads(converters, hf_layer, fast_block, args.atol, args.rtol)

    print(f"layer {layer_index} ({layer_type})")
    _print_diff("output", output_diff)
    _print_diff("input_grad", input_grad_diff)
    print(f"  parameter_grads: {'PASS' if not grad_failures else 'FAIL'} checked={checked_grads}")
    for name, grad_diff in grad_failures[:10]:
        _print_diff(f"grad {name}", grad_diff)

    return output_diff.passed and input_grad_diff.passed and not grad_failures


def main() -> None:
    args = _parse_args()
    if args.disable_triton:
        TritonConfig.TRITON_ENABLED = False
    device = _device(args.device)
    torch.manual_seed(args.seed)

    hf_model = Gemma4ForCausalLM(_make_tiny_config()).to(device)
    hf_model.train()

    layer_plan = []
    if args.layers in {"both", "sliding"}:
        layer_plan.append((0, "sliding_attention"))
    if args.layers in {"both", "full"}:
        layer_plan.append((5, "full_attention"))

    print(f"device: {device}")
    print(f"shape: batch={args.batch_size} seq={args.seq_len} hidden={hf_model.config.hidden_size}")
    print(f"tolerances: atol={args.atol:g}, rtol={args.rtol:g}")

    passed = True
    for layer_index, layer_type in layer_plan:
        passed = _run_layer(hf_model, layer_index, layer_type, args, device) and passed

    print("PASS" if passed else "FAIL")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*GenerationMixin.*")
        main()
