import pathlib
from functools import partial

import pytest
import torch

from fast_llm.config import NoAutoValidate
from fast_llm.engine.checkpoint.config import CheckpointLoadConfig
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.schedule.config import ScheduleConfig
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.engine.schedule.schedule import Schedule
from fast_llm.layers.language_model.config import LanguageModelKwargs, LanguageModelLossNames
from fast_llm.layers.ssm.config import SSMBlockType
from fast_llm.layers.ssm.discrete_mamba2 import DiscreteMamba2
from fast_llm.layers.ssm.llamba_block import LlambaBlock
from fast_llm.layers.ssm.mamba_layer import MambaLayer
from fast_llm.layers.transformer.config import TransformerKwargs
from fast_llm.models.gpt.config import GPTBatchConfig, LlamaGPTHuggingfaceCheckpointFormat
from fast_llm.models.ssm.config import AprielSSMHHybridHuggingfaceCheckpointFormat, LLambaHuggingfaceCheckpointFormat
from fast_llm.models.ssm.model import HybridSSMBaseModel, HybridSSMModel
from tests.common import get_hybrid_config, materialize_meta_tensors

try:
    from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel as LMHeadModel
except ImportError:
    LMHeadModel = None

run_test = MambaLayer is not None and torch.cuda.is_available()


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


def get_hf_llamba_out(input_ids, path, format):
    if format == LLambaHuggingfaceCheckpointFormat:
        from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel as LMHeadModel
    elif format == LlamaGPTHuggingfaceCheckpointFormat:
        from transformers import LlamaForCausalLM as LMHeadModel
    else:
        raise ValueError(f"Invalid format: {format}")

    model = LMHeadModel.from_pretrained(path, strict=True).to("cuda")
    parameter_sum = sum(p.detach().cpu().numpy().sum() for p in model.parameters())
    print(f"Parameter sum: {parameter_sum}")
    output = model(input_ids)
    del model
    torch.cuda.empty_cache()
    return output, parameter_sum


@pytest.mark.slow
@pytest.mark.skipif(
    not run_test or LMHeadModel is None,
    reason=f"Skipping because one of the following: cartesia_pytorch.Llamba not installed or no CUDA available or Mamba not installed",
)
def test_load_from_llamba_checkpoint(distributed_config):
    """
    Test to check whether the of Fast-LLM and Huggingface checkpoint loading for Llamba-1B produce the same results.
    """
    vocab_size = 128256  # from https://huggingface.co/cartesia-ai/Llamba-1B/blob/main/config.json
    batch_size = 2
    seq_length = 32

    path = pathlib.Path("/mnt/checkpoints_fml/pretrained_models/Llamba-1B")
    format = LLambaHuggingfaceCheckpointFormat

    x = torch.randint(0, vocab_size, (batch_size, seq_length), device="cuda")
    hf_logits, parameter_sum_hf = get_hf_llamba_out(x, path, format)
    hf_logits = hf_logits["logits"].cpu()

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
    batch_config.setup(distributed_config)
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

    losses, success, metrics = schedule_runner.run_step(
        iter([input_data]), schedule, iteration=0, return_metrics=True, preprocessed=True
    )

    logits = input_data[0][1]["logits"].cpu()
    assert torch.allclose(logits, hf_logits, atol=1e-2)


def get_hf_apriel_hybrid_out(input_ids, path, format):
    from fast_llm.models.ssm.external.apriel_hybrid.modeling_ssm_hybrid_apriel import AprielSSMHybridForCausalLM

    model = AprielSSMHybridForCausalLM.from_pretrained(path, strict=True).to("cuda")
    parameter_sum = sum(p.detach().cpu().numpy().sum() for p in model.parameters())
    print(f"Parameter sum: {parameter_sum}")
    output = model(input_ids)
    del model
    torch.cuda.empty_cache()
    return output, parameter_sum


@pytest.mark.slow
@pytest.mark.skipif(
    not run_test
    and not pathlib.Path("/mnt/checkpoints/ssm/apriel_ssm_instruct_hybrid_ssm2nd_init_mambainlama_debug").exists(),
    reason=f"Skipping because no CUDA available or Mamba not installed",
)
def test_load_from_hybridssm_checkpoint(distributed_config):
    """
    Test to check whether the of Fast-LLM and Huggingface checkpoint loading for Llamba-1B produce the same results.
    """
    vocab_size = 131072  # from https://huggingface.co/cartesia-ai/Llamba-1B/blob/main/config.json
    batch_size = 2
    seq_length = 32

    path = pathlib.Path("/mnt/checkpoints/ssm/apriel_ssm_instruct_hybrid_ssm2nd_init_mambainlama_debug")
    format = AprielSSMHHybridHuggingfaceCheckpointFormat

    x = torch.randint(0, vocab_size, (batch_size, seq_length), device="cuda")
    hf_logits, parameter_sum_hf = get_hf_apriel_hybrid_out(x, path, format)
    hf_logits = hf_logits["logits"].cpu()

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


@pytest.mark.extra_slow
@pytest.mark.skipif(not run_test, reason="No CUDA available or Mamba not installed")
@pytest.mark.parametrize(
    "hybrid_block_layout,LAYER_CLS",
    [
        ([SSMBlockType.mamba, SSMBlockType.transformer], MambaLayer),
        ([SSMBlockType.mamba2_discrete, SSMBlockType.transformer], DiscreteMamba2),
    ],
    ids=["mamba", "discrete_mamba2"],
)
def test_mamba_layer(distributed_config, distributed, hybrid_block_layout, LAYER_CLS):
    hybrid_config = get_hybrid_config(hybrid_block_layout=hybrid_block_layout)
    tensor_space = TensorSpace(distributed_config=distributed_config)
    hybrid_config.setup_tensor_space(tensor_space)
    layer = LAYER_CLS(hybrid_config.ssm, layer_idx=0, tensor_space=tensor_space)
    tensor_space.setup(distributed)
    materialize_meta_tensors(layer, tensor_space)
    layer.to(distributed.device)

    batch_size = 2
    seq_length = 32
    hidden_size = hybrid_config.transformer.hidden_size
    x = torch.randn(batch_size, seq_length, hidden_size, device=distributed.device)

    # Run forward pass
    output, _ = layer(x, {})

    loss = output.sum()
    loss.backward()
    # Basic shape checkss
    assert output.shape == x.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


@pytest.mark.skipif(not run_test, reason="No CUDA available or Mamba not installed")
def test_mamba_block(distributed_config, distributed):
    hybrid_config = get_hybrid_config(hybrid_block_layout=["m", "t"])
    tensor_space = TensorSpace(distributed_config=distributed_config)
    tensor_space.setup(distributed)
    hybrid_config.setup_tensor_space(tensor_space)
    layer_idx = 0

    mixer_cls = partial(MambaLayer, layer_idx=layer_idx)
    block = LlambaBlock(
        hybrid_config.transformer,
        hybrid_config.ssm,
        mixer_cls=mixer_cls,
        tensor_space=tensor_space,
        layer_index=layer_idx,
    )

    materialize_meta_tensors(block, tensor_space)
    block.to("cuda")

    batch_size = 2
    seq_length = 32
    hidden_size = hybrid_config.transformer.hidden_size
    x = torch.randn(batch_size, seq_length, hidden_size, device=distributed.device)

    hidden_states = block(x, {})
    loss = hidden_states.sum()
    loss.backward()

    assert hidden_states.shape == x.shape
    assert not torch.isnan(hidden_states).any()
    assert not torch.isinf(hidden_states).any()


@pytest.mark.slow
@pytest.mark.skipif(not run_test, reason="No CUDA available or Mamba not installed")
@pytest.mark.parametrize(
    ("hybrid_block_layout"),
    [
        (["m", "t"]),
        (["m2d", "t"]),
    ],
    ids=["mamba", "discrete_mamba2"],
)
def test_hybrid_model_train_with_fast_mode(distributed_config, hybrid_block_layout):
    hybrid_config = get_hybrid_config(hybrid_block_layout=hybrid_block_layout)
    model = HybridSSMBaseModel(hybrid_config, distributed_config)
    distributed = Distributed(distributed_config)
    model.setup(distributed)
    tensor_space = model._tensor_space
    materialize_meta_tensors(model, tensor_space)
    model.to("cuda")

    batch_size = 2
    seq_length = 32
    x = torch.randint(0, 49152, (batch_size, seq_length), device="cuda")
    position_ids = torch.arange(seq_length, device="cuda", dtype=torch.int64)
    attention_mask = torch.ones((1, 1, 1, 1), device="cuda", dtype=torch.bool)  # will be broadcasted to right shape
    labels = torch.randint(0, 49152, (batch_size, seq_length), device="cuda")
    losses = {LanguageModelLossNames.language_model_loss: []}
    output = model(
        x,
        {
            "position_ids": position_ids,
            TransformerKwargs.sequence_first: False,
            TransformerKwargs.attention_mask: attention_mask,
            TransformerKwargs.attention_mask_value: -100,
            TransformerKwargs.grad_output: True,
            LanguageModelKwargs.labels: labels,
        },
        losses=losses,
    )
    loss = sum(losses[LanguageModelLossNames.language_model_loss])
    loss.backward()


# TODO: added this when inference enabled
# No inference for now
# @dataclass
# class InferenceParams:
#     max_seqlen: int
#     max_batch_size: int
#     sequence_len_offset: int = 0
#     key_value_memory_dict: dict = None

#     def __post_init__(self):
#         if self.key_value_memory_dict is None:
#             self.key_value_memory_dict = {}


# @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available")
# def test_hybrid_model_inference(distributed_config, hybrid_config):
#     hybrid_config.ssm.use_fast_path = False
#     model = HybridSSMBaseModel(hybrid_config, distributed_config)
#     distributed = Distributed(distributed_config)
#     model.setup(distributed)
#     tensor_space = model._tensor_space
#     materialize_meta_tensors(model, tensor_space)
#     model.to("cuda")
#     # print(model)

#     batch_size = 2
#     seq_length = 32
#     x = torch.randint(0, 49152, (batch_size, seq_length), device="cuda")
#     position_ids = torch.arange(seq_length, device="cuda", dtype=torch.int64)
#     attention_mask = torch.ones((1, 1, 1, 1), device="cuda", dtype=torch.bool)  # will be broadcasted to right shape
#     labels = torch.randint(0, 49152, (batch_size, seq_length), device="cuda")
#     max_new_tokens = 10

#     inference_params = InferenceParams(
#         max_seqlen=len(x[0]) + max_new_tokens, max_batch_size=x.shape[0], sequence_len_offset=0
#     )
#     losses = {LanguageModelLossNames.language_model_loss: []}

#     output = model(
#         x,
#         {
#             "position_ids": position_ids,
#             TransformerKwargs.sequence_first: True,
#             TransformerKwargs.attention_mask: attention_mask,
#             TransformerKwargs.attention_mask_value: -100,
#             TransformerKwargs.grad_output: True,
#             LanguageModelKwargs.labels: labels,
#             "inference_params": inference_params,
#         },
#         losses=losses,
#     )

if __name__ == "__main__":
    pytest.main(["-s", __file__])
