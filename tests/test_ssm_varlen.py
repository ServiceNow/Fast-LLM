import inspect
import itertools

import pytest
import torch

from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.ssm import kda as kda_module
from fast_llm.layers.ssm.config import KimiDeltaAttentionConfig, LinearAttentionKwargs

# from mamba2 import NemotronHMamba2


_mamba_varlen = False
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa

    _mamba_available = True
    sig = inspect.signature(selective_scan_fn)
    if "position_indices" in sig.parameters:
        _mamba_varlen = True
    else:
        _mamba_varlen = False
        # for training with packing install https://github.com/jxiw/varlen_mamba
        # see https://github.com/jxiw/M1/blob/main/HYBRID_PACK.md

except (ImportError, RuntimeError):
    _mamba_available = False


@pytest.fixture
def distributed_config():
    return DistributedConfig(
        tensor_parallel=1,
        pipeline_parallel=1,
        sequence_data_parallel=1,
        local_world_size=1,
        world_size=1,
    )


@pytest.fixture
def distributed(distributed_config):
    return Distributed(config=distributed_config)


def materialize_meta_tensors(model, tensor_space):
    # Materialize parameters that are on meta device
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            # Check if the parameter is a custom tensor type
            if hasattr(param, "tensor_name") and hasattr(param, "init_parameter"):
                param_data = param.new_empty(param.shape, device="cuda")
                # Initialize param_data
                param.init_parameter(param_data, tensor_space.distributed)
                # Replace the parameter in the module
                module_path, param_name = name.rsplit(".", 1) if "." in name else (None, name)
                module = model
                if module_path is not None:
                    for part in module_path.split("."):
                        module = getattr(module, part)
                param = torch.nn.Parameter(param_data, requires_grad=param.requires_grad)
                # TODO: add param_grad_is_zero etc., grad_buffer, etc., see test_mlp_recomputation
                param.grad = None
                param.grad_buffer = torch.empty_like(param)
                param.param_grad_is_zero = True
                module._parameters[param_name] = param
    return model


def unpack(packed_hidden_states, cu_seqlens):
    batch_size = packed_hidden_states.shape[0]
    package_num = cu_seqlens.shape[0] - 1
    seq_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
    hidden_dim = packed_hidden_states.shape[2]
    hidden_states = torch.zeros(
        package_num * batch_size,
        seq_len,
        hidden_dim,
        dtype=packed_hidden_states.dtype,
        device=packed_hidden_states.device,
    )
    for j in range(batch_size):
        for i in range(package_num):
            line = j * package_num + i
            hidden_states[line, : cu_seqlens[i + 1] - cu_seqlens[i], :] = packed_hidden_states[
                j, cu_seqlens[i] : cu_seqlens[i + 1], :
            ]
    return hidden_states


def pack(hidden_states, cu_seqlens, batch_size):
    package_num, seq_len, hidden_dim = hidden_states.shape
    seq_len_list = cu_seqlens[1:] - cu_seqlens[:-1]
    seq_len_list_3d = seq_len_list.unsqueeze(1).unsqueeze(2)
    indices_3d = (
        torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).unsqueeze(2).repeat(package_num, 1, hidden_dim)
    )
    mask_3d = indices_3d < seq_len_list_3d.repeat(batch_size, 1, 1)
    packed_hidden_states = hidden_states[mask_3d].view(batch_size, -1, hidden_dim)
    return packed_hidden_states


def generate_random_cu_seqlens(seq_len, packages_num=2):
    if packages_num < 1:
        raise ValueError("packages_num must be at least 1")

    # base size of each chunk, and how many get an extra token
    base, rem = divmod(seq_len, packages_num)
    # lengths: e.g. for seq_len=10, packages=3 → [4,3,3]
    lengths = [base + 1 if i < rem else base for i in range(packages_num)]

    # split points exclude the final cumulative (seq_len)
    split_points = list(itertools.accumulate(lengths))[:-1]

    cu_seqlens = [0] + split_points + [seq_len]
    # cu_seqlens = split_points  # + [seq_len]

    # index: for each chunk, we emit 0,1,...,length-1
    index = []
    for length in lengths:
        index.extend(range(length))

    # sanity check
    assert len(cu_seqlens) - 1 == packages_num
    assert sum(lengths) == seq_len
    assert len(index) == seq_len

    return cu_seqlens, index


def _materialize_kda_tensors(module: torch.nn.Module, distributed: Distributed, device: torch.device) -> None:
    """
    Materialize meta parameters on the requested device for KDA mixer layers.
    """
    for name, param in module.named_parameters():
        if param.device.type != "meta":
            continue
        param_data = torch.empty_like(param, device=device)
        param.init_parameter(param_data, distributed)
        module_path, param_name = name.rsplit(".", 1) if "." in name else (None, name)
        target = module
        if module_path is not None:
            for part in module_path.split("."):
                target = getattr(target, part)
        new_param = torch.nn.Parameter(param_data, requires_grad=param.requires_grad)
        new_param.grad = None
        new_param.grad_buffer = torch.zeros_like(param_data)
        new_param.param_grad_is_zero = True
        target._parameters[param_name] = new_param


def _param_grad(param: torch.nn.Parameter) -> torch.Tensor | None:
    # ParameterMeta stores grads in grad_buffer; fall back to .grad otherwise.
    return param.grad_buffer if hasattr(param, "grad_buffer") and param.grad_buffer is not None else param.grad


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="KDA varlen needs CUDA")
@pytest.mark.skipif(
    kda_module.chunk_kda is None or kda_module.fused_kda_gate is None,
    reason="KDA fused kernels not available",
)
def test_kda_varlen_stacking_equivalence(distributed_config, distributed):
    """
    Check that KDA forward/backward match with and without stacking using the real kernels.
    """
    device = torch.device("cuda")
    dtype = torch.float16
    heads, head_dim = 2, 16
    hidden_size = heads * head_dim

    config = KimiDeltaAttentionConfig(heads=heads, head_dim=head_dim)
    hidden_dim = TensorDim("hidden", hidden_size)
    kda_packed = config.get_layer(distributed_config, hidden_dim, lr_scale=None, peft=None, return_bias=False)
    kda_ref = config.get_layer(distributed_config, hidden_dim, lr_scale=None, peft=None, return_bias=False)
    kda_packed.setup(distributed)
    kda_ref.setup(distributed)
    _materialize_kda_tensors(kda_packed, distributed, device)
    _materialize_kda_tensors(kda_ref, distributed, device)
    kda_ref.load_state_dict(kda_packed.state_dict())
    kda_packed.to(device=device, dtype=dtype)
    kda_ref.to(device=device, dtype=dtype)

    batch_size = 2  # cu_seqlens path requires flattened batch
    seq_len = 15
    packages_num = torch.randint(2, 5, (1, batch_size))[0]  # randomize packages num between 2 and 4
    lengths = [
        torch.tensor(
            generate_random_cu_seqlens(seq_len, packages_num=packages_num[i].item())[0],
            device=device,
            dtype=torch.long,
        ).diff()
        for i in range(batch_size)
    ]

    # lengths = torch.tensor(cu_seqlens, device=device, dtype=torch.long)#.diff()
    # total_tokens = lengths.sum().item()
    packed = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=True)

    kwargs_packed = {
        LinearAttentionKwargs.sequence_lengths: lengths,
        BlockKwargs.sequence_first: False,
        BlockKwargs.hidden_dims: (hidden_dim,),
        # BlockKwargs.sequence_q_dim: TensorDim("sequence_q", lengths.sum().item()),
    }
    # Use the layer's preprocess to construct cu_seqlens/seq_idx the same way as the implementation.
    kda_packed.preprocess(packed, kwargs_packed)

    kwargs_ref = {
        BlockKwargs.sequence_first: False,
        BlockKwargs.hidden_dims: (hidden_dim,),
    }

    out_packed = kda_packed(packed, kwargs_packed)
    # Run reference path separately per sequence without varlen packing, then concatenate.
    ref_outs = []
    for b in range(batch_size):
        out_batch = []
        length = lengths[b]
        ref_seqs = torch.split(packed[b].unsqueeze(0), length.tolist(), dim=1)
        for seq in ref_seqs:
            kwargs_ref_seq = {
                **kwargs_ref,
                BlockKwargs.sequence_q_dim: TensorDim("sequence_q", seq.shape[1]),
            }
            out_batch.append(kda_ref(seq, kwargs_ref_seq))
        ref_outs.append(torch.cat(out_batch, dim=1))
    out_ref = torch.cat(ref_outs, dim=0)
    out_ref_packed = out_ref

    assert out_packed.shape == packed.shape
    assert out_ref_packed.shape == out_packed.shape
    assert torch.allclose(out_packed, out_ref_packed, atol=1e-3, rtol=1e-3)

    out_packed.sum().backward()
    out_ref_packed.sum().backward()

    assert _param_grad(kda_packed.q_proj.weight) is not None
    assert _param_grad(kda_ref.q_proj.weight) is not None
    assert torch.allclose(
        _param_grad(kda_packed.q_proj.weight), _param_grad(kda_ref.q_proj.weight), atol=1e-3, rtol=1e-3
    )
    assert torch.allclose(
        _param_grad(kda_packed.k_proj.weight), _param_grad(kda_ref.k_proj.weight), atol=1e-3, rtol=1e-3
    )
    assert torch.allclose(
        _param_grad(kda_packed.v_proj.weight), _param_grad(kda_ref.v_proj.weight), atol=1e-3, rtol=1e-3
    )
    assert torch.allclose(
        _param_grad(kda_packed.o_proj.weight), _param_grad(kda_ref.o_proj.weight), atol=1e-3, rtol=1e-3
    )


if __name__ == "__main__":
    pytest.main([__file__])
