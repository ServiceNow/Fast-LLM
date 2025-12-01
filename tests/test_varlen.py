import pytest
import torch

from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.decoder.config import MixerConfig
from fast_llm.layers.ssm import gdn as gdn_module
from fast_llm.layers.ssm.config import GatedDeltaNetConfig


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


def unpack_and_padd(packed_hidden_states, cu_seqlens, package_num):
    batch_size = packed_hidden_states.shape[0]
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


def generate_random_seq_len(seq_len, packages_num=2):
    if packages_num < 1:
        raise ValueError("packages_num must be at least 1")

    # base size of each chunk, and how many get an extra token
    base, rem = divmod(seq_len, packages_num)
    # lengths: e.g. for seq_len=10, packages=3 → [4,3,3]
    lengths = [base + 1 if i < rem else base for i in range(packages_num)]
    assert sum(lengths) == seq_len
    assert len(lengths) == packages_num
    return lengths


def _materialize_mixer_tensors(module: torch.nn.Module, distributed: Distributed, device: torch.device) -> None:
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
    return param.grad_buffer if hasattr(param, "grad_buffer") and param.grad_buffer is not None else param.grad


# TODO: include mamba varlen
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Varlen test needs CUDA")
@pytest.mark.skipif(
    gdn_module.chunk_gated_delta_rule is None,
    reason="Gated Delta Net fused kernels not available",
)
@pytest.mark.parametrize(
    "config, sequence_first",
    [
        pytest.param(GatedDeltaNetConfig(value_heads=4, key_heads=2, key_head_dim=16, value_head_dim=16), False),
        pytest.param(GatedDeltaNetConfig(value_heads=4, key_heads=2, key_head_dim=16, value_head_dim=16), True),
    ],
)
def test_mixer_varlen_stacking_equivalence(config: MixerConfig, sequence_first: bool, distributed_config, distributed):
    """
    Check that Gated Delta Net forward/backward match with and without packing.
    """
    device = torch.device("cuda")
    dtype = torch.float16
    hidden_size = 32
    hidden_dim = TensorDim("hidden", hidden_size)
    mixer_packed = config.get_layer(distributed_config, hidden_dim, lr_scale=None, peft=None, return_bias=False)
    mixer_ref = config.get_layer(distributed_config, hidden_dim, lr_scale=None, peft=None, return_bias=False)
    mixer_packed.setup(distributed)
    mixer_ref.setup(distributed)
    _materialize_mixer_tensors(mixer_packed, distributed, device)
    _materialize_mixer_tensors(mixer_ref, distributed, device)
    mixer_ref.load_state_dict(mixer_packed.state_dict())
    mixer_packed.to(device=device, dtype=dtype)
    mixer_ref.to(device=device, dtype=dtype)

    batch_size = 2  # cu_seqlens path requires flattened batch
    seq_len = 15
    packages_num = torch.tensor([2, 3], device=device, dtype=torch.long)
    sequence_lengths = [
        generate_random_seq_len(seq_len, packages_num=packages_num[i].item()) for i in range(batch_size)
    ]

    packed = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=True)
    if sequence_first:
        packed = packed.transpose(0, 1)

    kwargs_packed = {
        BlockKwargs.device: device,
        BlockKwargs.sequence_lengths: sequence_lengths,
        BlockKwargs.sequence_first: sequence_first,
        BlockKwargs.hidden_dims: (hidden_dim,),
    }
    mixer_packed.preprocess(kwargs_packed)

    kwargs_ref = {
        BlockKwargs.device: device,
        BlockKwargs.sequence_first: False,
        BlockKwargs.hidden_dims: (hidden_dim,),
    }

    out_packed = mixer_packed(packed, kwargs_packed)
    if sequence_first:
        out_packed = out_packed.transpose(0, 1)
    # Run reference path separately per sequence without varlen packing, then concatenate.
    ref_outs = []
    for b in range(batch_size):
        out_batch = []
        length = sequence_lengths[b]
        if sequence_first:
            ref_seqs = torch.split(packed[:, b].unsqueeze(0), length, dim=1)
        else:
            ref_seqs = torch.split(packed[b].unsqueeze(0), length, dim=1)
        for seq in ref_seqs:
            kwargs_ref_seq = {
                **kwargs_ref,
                BlockKwargs.sequence_lengths: [seq.shape[1]],
            }
            out_batch.append(mixer_ref(seq, kwargs_ref_seq))
        ref_outs.append(torch.cat(out_batch, dim=1))
    out_ref = torch.cat(ref_outs, dim=0)
    out_ref_packed = out_ref

    assert out_ref_packed.shape == out_packed.shape
    assert torch.allclose(out_packed, out_ref_packed, atol=1e-3, rtol=1e-3)

    out_packed.sum().backward()
    out_ref_packed.sum().backward()

    for (name, param), (_, param_ref) in zip(mixer_packed.named_parameters(), mixer_ref.named_parameters()):
        if param.requires_grad:
            torch.testing.assert_close(
                _param_grad(param),
                _param_grad(param_ref),
                atol=1e-3,
                rtol=1e-3,
                msg=f"Grad mismatch for parameter {name}",
            )


if __name__ == "__main__":
    pytest.main([__file__])
