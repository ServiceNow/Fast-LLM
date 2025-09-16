import inspect
import itertools
import pathlib
from functools import partial

import pytest
import torch
from mamba2 import Mamba2, NemotronHMamba2

from fast_llm.config import NoAutoValidate
from fast_llm.engine.checkpoint.config import CheckpointLoadConfig
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.schedule.config import ScheduleConfig
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.engine.schedule.schedule import Schedule
from fast_llm.layers.ssm.config import SSMConfig
from fast_llm.layers.ssm.llamba_block import SSMBlock
from fast_llm.layers.transformer.config import TransformerConfig, TransformerKwargs
from fast_llm.models.gpt.config import GPTBatchConfig
from fast_llm.models.ssm.config import HybridSSMBaseModelConfig, LLambaHuggingfaceCheckpointFormat
from fast_llm.models.ssm.model import HybridSSMModel

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


def get_hybrid_config(hybrid_block_layout=["t", "m2"], prediction_heads=1, default_mtp_type=None):
    hidden_size = 512
    config = HybridSSMBaseModelConfig(
        transformer=TransformerConfig(num_layers=len(hybrid_block_layout), hidden_size=hidden_size),
        ssm=SSMConfig(d_xb=hidden_size, dt_rank=10, d_inner=hidden_size * 2, state_size=16, head_dim=8),
        hybrid_block_layout=hybrid_block_layout,
        prediction_heads=prediction_heads,
        default_mtp_type=default_mtp_type,
        init_method_std_embed=0.02,
        init_method_min_embed=-0.02,
        init_method_max_embed=0.02,
        use_position_embeddings=True,
        tie_word_embeddings=False,
    )
    return config


@pytest.mark.skip("Disabled due to cartesia_pytorch installation issue")
@pytest.mark.slow
def test_load_from_llamba_checkpoint():
    """
    Test to check whether the of Fast-LLM and Huggingface checkpoint loading for Llamba-1B produce the same results.
    """
    import cartesia_pytorch.Llamba.llamba

    vocab_size = 128256  # from https://huggingface.co/cartesia-ai/Llamba-1B/blob/main/config.json
    batch_size = 2
    seq_length = 32

    path = pathlib.Path("/mnt/checkpoints_fml/pretrained_models/Llamba-1B")
    format = LLambaHuggingfaceCheckpointFormat

    x = torch.randint(0, vocab_size, (batch_size, seq_length), device="cuda")

    hf_model = cartesia_pytorch.Llamba.llamba.LMHeadModel.from_pretrained(path, strict=True).to("cuda")
    parameter_sum_hf = sum(p.detach().sum().cpu().item() for p in hf_model.parameters())
    hf_logits = hf_model(x)["logits"].cpu()
    del hf_model
    torch.cuda.empty_cache()

    # Create checkpoint load config
    checkpoint_config = CheckpointLoadConfig(path=path, format=format, model_weights=True, optimizer_state=False)
    # Initialize model
    model = HybridSSMModel.from_pretrained(checkpoint_config)
    param_sum = 0
    for stage in model.stages:
        for fsdp in stage.fsdps:
            if hasattr(fsdp, "_weight_shard"):
                param_sum += torch.sum(fsdp._weight_shard).item()
    assert torch.abs(torch.tensor(param_sum) - parameter_sum_hf) < 1e-1

    # model = GPTModel.from_pretrained(checkpoint_config)
    assert model.config.base_model.vocab_size == vocab_size
    schedule_config = ScheduleConfig()
    with NoAutoValidate():
        batch_config = GPTBatchConfig(micro_batch_size=batch_size, sequence_length=seq_length)
    batch_config.setup(DistributedConfig.from_dict({}))
    batch_config.validate()
    schedule_runner = ScheduleRunner(
        config=schedule_config,
        multi_stage=model,
        distributed_config=model.distributed.config,
    )
    schedule = Schedule(
        multi_stage=model,
        batch_config=batch_config,
        schedule_config=schedule_config,
        distributed_config=model.distributed.config,
        phase=PhaseType.inference,
    )
    schedule_runner.setup(model.distributed, optimizer=None)

    common_kwargs = {
        TransformerKwargs.sequence_first: True,
        TransformerKwargs.grad_output: False,
    }
    input_data = [(x, common_kwargs)]

    schedule_runner.run_step(iter([input_data]), schedule, iteration=0, return_metrics=True, preprocessed=True)

    logits = input_data[0][1]["logits"].cpu()
    assert torch.allclose(logits, hf_logits, atol=1e-2)


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

    # cu_seqlens = [0] + split_points + [seq_len]
    cu_seqlens = [0] + split_points + [seq_len]

    # index: for each chunk, we emit 0,1,...,length-1
    index = []
    for length in lengths:
        index.extend(range(length))

    # sanity check
    assert len(cu_seqlens) - 1 == packages_num
    assert sum(lengths) == seq_len
    assert len(index) == seq_len

    return cu_seqlens, index


# Quick and dirty test for Mamba2 varlen block from https://github.com/jxiw/M1/blob/d92b53faa640f8ebf624d3e9e771fe24648ef014/rl/verl/tests/pack_mamba/test_mamba_layer.py
# test that packed and not packed are producing the same result in terms of outputs and gradients
# TODO: integrate in the testing framework
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available")
@pytest.mark.skipif(not _mamba_available, reason="Mamba2 is not available")
@pytest.mark.parametrize(
    "mixer_cls, hybrid_block_layout, tollerance",
    [
        pytest.param(
            partial(NemotronHMamba2, block_index=0),
            ["nm2", "t"],
            1e-3,  # not 100% sure why, mamba2 requires lower tollerance (maybe its not really supporting packing)
            id="nemotron_hmamba2",
        ),
        pytest.param(
            partial(Mamba2, block_index=0),
            ["m2", "t"],
            1e-4,
            marks=pytest.mark.skipif(not _mamba_varlen, reason="Mamba2 varlen is not available"),
            id="mamba2",
        ),
    ],
)
def test_mamba_varlen_block(mixer_cls, hybrid_block_layout, tollerance, distributed_config, distributed):
    """
    Compare that the output and grads of packed and unpacked Mamba2 varlen block are the same.
    """
    hybrid_config = get_hybrid_config(hybrid_block_layout=hybrid_block_layout)

    tensor_space = TensorSpace(distributed_config=distributed_config)
    tensor_space.setup(distributed)
    hybrid_config.setup_tensor_space(tensor_space)
    layer_idx = 0

    block_packed = SSMBlock(
        hybrid_config.transformer,
        hybrid_config.ssm,
        mixer_cls=mixer_cls,
        tensor_space=tensor_space,
        block_index=layer_idx,
    )
    block_ref = SSMBlock(
        hybrid_config.transformer,
        hybrid_config.ssm,
        mixer_cls=mixer_cls,
        tensor_space=tensor_space,
        block_index=layer_idx,
    )
    device = "cuda"
    materialize_meta_tensors(block_packed, tensor_space)
    materialize_meta_tensors(block_ref, tensor_space)
    block_ref.load_state_dict(block_packed.state_dict())
    block_packed.to(device)
    block_ref.to(device)

    batch_size = 2
    seq_len = 64
    packages_num = 2
    hidden_dim = hybrid_config.transformer.hidden_size

    cu_seqlens, index = generate_random_cu_seqlens(seq_len, packages_num=packages_num)
    cu_seqlens = torch.tensor(cu_seqlens).cuda()
    ssm_position_ids = torch.tensor(index, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1).contiguous().cuda()
    seq_idx = (
        torch.cat(
            [
                torch.full((s,), i, dtype=torch.int32, device=cu_seqlens.device)
                for i, s in enumerate(cu_seqlens[1:] - cu_seqlens[:-1])
            ],
            dim=0,
        )
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )

    # Generate packed_hidden_states with random values for testing
    hidden_states_list = [
        torch.randn(l, hidden_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
        for l in (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    ]
    packed_hidden_states = torch.cat(hidden_states_list, dim=0).unsqueeze(0)
    packed_hidden_states = packed_hidden_states.expand(batch_size, -1, -1).contiguous()
    # hidden_states should be forwarded without cu_seqlens
    hidden_states = unpack(packed_hidden_states, cu_seqlens)

    # Check: sum of seq_len of item in hidden_states_list should be equal to seq_len of packed_hidden_states
    assert sum([hs.shape[0] for hs in hidden_states_list]) == packed_hidden_states.shape[1]
    # Check: max of seq_len of item in hidden_states_list should be equal to seq_len of hidden_states
    assert max([hs.shape[0] for hs in hidden_states_list]) == hidden_states.shape[1]

    output_states_packed = block_packed(
        packed_hidden_states,
        {"cu_seqlens": cu_seqlens, "seq_idx": seq_idx, "ssm_position_ids": ssm_position_ids, "sequence_first": False},
    )
    output_states_unpacked = block_ref(
        hidden_states.clone(), {"cu_seqlens": None, "seq_idx": None, "ssm_position_ids": None, "sequence_first": False}
    )
    assert output_states_packed.shape == packed_hidden_states.shape
    assert output_states_unpacked.shape == hidden_states.shape
    assert not torch.isnan(hidden_states).any()
    assert not torch.isinf(hidden_states).any()

    output_states_unpacked = pack(output_states_unpacked, cu_seqlens, batch_size)
    assert torch.allclose(output_states_packed, output_states_unpacked, atol=tollerance)

    loss = output_states_packed.sum()
    loss.backward()
    loss_ref = output_states_unpacked.sum()
    loss_ref.backward()
    assert torch.allclose(block_packed.mixer.conv1d_weight.grad, block_ref.mixer.conv1d_weight.grad, atol=tollerance)
    assert torch.allclose(block_packed.mixer.conv1d_bias.grad, block_ref.mixer.conv1d_bias.grad, atol=tollerance)
    assert torch.allclose(
        block_packed.mixer.in_proj.weight.grad_buffer, block_ref.mixer.in_proj.weight.grad_buffer, atol=tollerance
    )
    assert torch.allclose(
        block_packed.mixer.out_proj.weight.grad_buffer, block_ref.mixer.out_proj.weight.grad_buffer, atol=tollerance
    )
    assert torch.allclose(
        block_packed.mixer.dt_in_proj.weight.grad_buffer,
        block_ref.mixer.dt_in_proj.weight.grad_buffer,
        atol=tollerance,
    )

    assert torch.allclose(
        block_packed.mlp.layer_1.weight.grad_buffer, block_ref.mlp.layer_1.weight.grad_buffer, atol=tollerance
    )
    assert torch.allclose(
        block_packed.mlp.layer_1.bias.grad_buffer, block_ref.mlp.layer_1.bias.grad_buffer, atol=tollerance
    )
    assert torch.allclose(
        block_packed.mlp.layer_2.weight.grad_buffer, block_ref.mlp.layer_2.weight.grad_buffer, atol=tollerance
    )
    assert torch.allclose(
        block_packed.mlp.layer_2.bias.grad_buffer, block_ref.mlp.layer_2.bias.grad_buffer, atol=tollerance
    )


if __name__ == "__main__":
    pytest.main([__file__])
