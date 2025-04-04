import pathlib
from functools import partial

import pytest
import torch

from fast_llm.config import NoAutoValidate
from fast_llm.engine.checkpoint.config import CheckpointLoadConfig
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.schedule.config import BatchConfig, ScheduleConfig
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.engine.schedule.schedule import Schedule
from fast_llm.layers.language_model.config import LanguageModelKwargs, LanguageModelLossNames
from fast_llm.layers.transformer.config import TransformerConfig, TransformerKwargs
from fast_llm.models.gpt.config import LlamaGPTHuggingfaceCheckpointFormat
from fast_llm.models.ssm.config import LLambaHuggingfaceCheckpointFormat

try:
    from lamba_block import LambaBlock

    from fast_llm.layers.ssm.config import MambaConfig
    from fast_llm.layers.ssm.discrete_mamba2 import DiscreteMamba2
    from fast_llm.layers.ssm.mamba_layer import MambaLayer
    from fast_llm.models.ssm.model import HybridSSMBaseModel, HybridSSMBaseModelConfig, HybridSSMModel
except ImportError:
    MambaLayer, LambaBlock, HybridSSMBaseModel, HybridSSMBaseModelConfig, DiscreteMamba2 = None, None, None, None, None
    # Mamba not isntalled, skipping tests

run_test = MambaLayer is not None and torch.cuda.is_available()


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


def get_hybrid_config(use_fast_path: bool = True, block_pattern=["t", "m", "t", "m"]):
    config = HybridSSMBaseModelConfig(
        transformer=TransformerConfig(num_layers=len(block_pattern)),
        ssm=MambaConfig(),
        block_pattern=block_pattern,
        init_method_std_embed=0.02,
        init_method_min_embed=-0.02,
        init_method_max_embed=0.02,
        use_position_embeddings=True,
        tie_word_embeddings=False,
        use_fast_path=use_fast_path,
    )
    return config


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
@pytest.mark.skipif(not run_test, reason="No CUDA available or Mamba not installed")
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
    hf_logits, parameter_sum = get_hf_llamba_out(x, path, format)
    hf_logits = hf_logits["logits"].cpu()

    # Create checkpoint load config
    checkpoint_config = CheckpointLoadConfig(path=path, format=format, model_weights=True, optimizer_state=False)
    # Initialize model
    model = HybridSSMModel.from_pretrained(checkpoint_config)
    param_sum_fll = 0
    for stage in model.stages:
        if hasattr(stage, "_weight_shard"):
            param_sum_fll += torch.sum(stage._weight_shard).item()

    # model = GPTModel.from_pretrained(checkpoint_config)
    assert model.config.base_model.vocab_size == vocab_size
    schedule_config = ScheduleConfig()
    with NoAutoValidate():
        batch_config = BatchConfig(micro_batch_size=batch_size, sequence_length=seq_length)
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


@pytest.mark.skipif(not run_test, reason="No CUDA available or Mamba not installed")
@pytest.mark.parametrize(
    "block_pattern,use_fast_path,LAYER_CLS",
    [
        (["m", "t"], True, MambaLayer),
        (["m", "t"], False, MambaLayer),
        (["m2", "t"], False, DiscreteMamba2),
    ],
    ids=["mamba-fast", "mamba-slow", "descrete_mamba2"],
)
def test_mamba_layer(distributed_config, distributed, block_pattern, use_fast_path, LAYER_CLS):
    hybrid_config = get_hybrid_config(use_fast_path, block_pattern=block_pattern)
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
    output = layer(x)

    loss = output.sum()
    loss.backward()
    # Basic shape checkss
    assert output.shape == x.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


@pytest.mark.skipif(not run_test, reason="No CUDA available or Mamba not installed")
def test_mamba_block(distributed_config, distributed):
    hybrid_config = get_hybrid_config(use_fast_path=True, block_pattern=["m", "t"])
    tensor_space = TensorSpace(distributed_config=distributed_config)
    tensor_space.setup(distributed)
    hybrid_config.setup_tensor_space(tensor_space)
    layer_idx = 0

    mixer_cls = partial(MambaLayer, layer_idx=layer_idx)
    block = LambaBlock(
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


@pytest.mark.skipif(not run_test, reason="No CUDA available or Mamba not installed")
@pytest.mark.parametrize(
    "block_pattern,use_fast_path",
    [
        (["m", "t"], True),
        (["m", "t"], False),
        (["m2", "t"], False),
    ],
    ids=["mamba-fast", "mamba-slow", "descrete_mamba2"],
)
def test_hybrid_model_train_with_fast_mode(distributed_config, block_pattern, use_fast_path):
    hybrid_config = get_hybrid_config(use_fast_path, block_pattern=block_pattern)
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
            TransformerKwargs.sequence_first: True,
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
