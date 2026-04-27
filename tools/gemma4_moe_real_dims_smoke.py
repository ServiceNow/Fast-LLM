#!/usr/bin/env python3
"""Smoke test Gemma4 MoE dropless kernels at real 26B A4B dimensions."""

import argparse
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fast_llm.engine.base_model.config import set_model_names
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import StageConfig
from fast_llm.engine.multi_stage.stage import Stage
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.decoder.mlp.config import Gemma4MoEMLPConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--implementation", choices=("dropless", "auto", "looped"), default="dropless")
    parser.add_argument("--forward-only", action="store_true", help="Skip backward.")
    parser.add_argument("--dynamic-shape", action="store_true", help="Use dynamic dropless sparse-map shape.")
    parser.add_argument("--hidden-size", type=int, default=2816)
    parser.add_argument("--intermediate-size", type=int, default=2112)
    parser.add_argument("--moe-intermediate-size", type=int, default=704)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(name)


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _setup_stage(layer: torch.nn.Module, distributed: Distributed) -> Stage:
    if not layer._is_setup:
        layer.setup(distributed)
    set_model_names(torch.nn.ModuleList([layer]))
    stage = Stage(
        config=StageConfig(),
        layers=[layer],
        distributed_config=distributed.config,
        index=0,
        tied_parameter_duplicates=(),
    )
    stage.setup(distributed=distributed, tied_parameter_duplicate_buffers=None)
    stage.initialize_weights()
    stage.restore_parameters()
    stage.reset_gradients()
    return stage


def _make_config(args: argparse.Namespace) -> Gemma4MoEMLPConfig:
    return Gemma4MoEMLPConfig._from_dict(
        {
            "type": "gemma4_moe",
            "intermediate_size": args.intermediate_size,
            "moe_intermediate_size": args.moe_intermediate_size,
            "experts": args.experts,
            "experts_per_token": args.top_k,
            "shared_experts": 0,
            "gated": True,
            "activation": "gelu",
            "add_linear_biases": False,
            "implementation": args.implementation,
            "dropless_dynamic_shape": args.dynamic_shape,
            "router_norm_eps": 1e-6,
            "layer_1": {},
            "layer_2": {},
            "expert_layer_1": {},
            "expert_layer_2": {},
            "router": {},
            "router_scale": {},
            "per_expert_scale": {},
            "post_feedforward_norm_1": {"type": "rms_norm"},
            "pre_feedforward_norm_2": {"type": "rms_norm"},
            "post_feedforward_norm_2": {"type": "rms_norm"},
        }
    )


def _memory(prefix: str, device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device) / 2**30
    reserved = torch.cuda.memory_reserved(device) / 2**30
    max_allocated = torch.cuda.max_memory_allocated(device) / 2**30
    print(f"{prefix} memory: allocated={allocated:.2f} GiB reserved={reserved:.2f} GiB max={max_allocated:.2f} GiB")


def _grad(parameter: torch.nn.Parameter) -> torch.Tensor | None:
    if not getattr(parameter, "param_grad_is_zero", True):
        return parameter.grad_buffer
    if parameter.grad is not None:
        return parameter.grad
    return getattr(parameter, "grad_buffer", None)


def main() -> None:
    args = _parse_args()
    device = _device(args.device)
    dtype = _dtype(args.dtype)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    distributed_config = DistributedConfig(compute_dtype=args.dtype, use_cuda=device.type == "cuda")
    layer = _make_config(args).get_layer(
        distributed_config,
        TensorDim("hidden_size", args.hidden_size),
        lr_scale=None,
        peft=None,
        return_bias=True,
    )
    _setup_stage(layer, Distributed(distributed_config))
    layer.train(not args.forward_only)
    _memory("after_setup", device)

    residual = torch.randn(args.tokens, args.hidden_size, dtype=dtype, device=device, requires_grad=not args.forward_only)
    # Approximate DecoderBlock.norm_2 output; this smoke focuses on Gemma4 MoE dropless dimensions/kernels.
    mlp_input = torch.rms_norm(residual, (args.hidden_size,), None, 1e-6)
    kwargs = {
        BlockKwargs.pre_mlp_residual: residual,
        BlockKwargs.hidden_token_dim: TensorDim("tokens", args.tokens),
    }

    start = time.perf_counter()
    output, bias = layer(mlp_input, kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - start
    assert bias is None
    assert output.shape == residual.shape
    _memory("after_forward", device)

    backward_seconds = None
    if not args.forward_only:
        start = time.perf_counter()
        output.float().square().mean().backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter() - start
        _memory("after_backward", device)

        for name, parameter in (
            ("router_scale", layer.router_scale),
            ("per_expert_scale", layer.per_expert_scale),
            ("expert_layer_1.weight", layer.expert_layer_1.weight),
            ("expert_layer_2.weight", layer.expert_layer_2.weight),
        ):
            grad = _grad(parameter)
            assert grad is not None, f"{name} gradient is None"
            assert torch.isfinite(grad).all(), f"{name} gradient has non-finite values"
            assert grad.abs().sum() > 0, f"{name} gradient is all-zero"

    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"implementation: {args.implementation}")
    print(
        "dims: "
        f"tokens={args.tokens} hidden={args.hidden_size} dense_intermediate={args.intermediate_size} "
        f"moe_intermediate={args.moe_intermediate_size} experts={args.experts} top_k={args.top_k}"
    )
    print(f"output shape: {tuple(output.shape)}")
    print(f"output finite: {torch.isfinite(output).all().item()}")
    print(f"forward_seconds: {forward_seconds:.4f}")
    if backward_seconds is not None:
        print(f"backward_seconds: {backward_seconds:.4f}")
    print("PASS")


if __name__ == "__main__":
    main()
