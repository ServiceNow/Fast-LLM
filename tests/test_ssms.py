import pytest
import torch

from torch import nn
from functools import partial

from dataclasses import dataclass
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.multi_stage import MultiStageModel
from fast_llm.layers.ssm.config import MambaConfig
from fast_llm.layers.ssm.mamba_layer import MambaLayer
from fast_llm.layers.ssm.mamba_block import MambaBlock
from fast_llm.layers.transformer.config import TransformerKwargs
from fast_llm.models.ssm.model import HybridBaseModel, HybridModelConfig
from fast_llm.layers.common.normalization import LayerNorm, RMSNorm
from fast_llm.tensor import ParameterMeta, TensorMeta
from fast_llm.layers.transformer.config import TransformerArchitectureConfig, TransformerConfig
from fast_llm.layers.language_model.config import LanguageModelKwargs, LanguageModelLossNames


def materialize_meta_tensors(model, tensor_space):
    # Initialize parameters that are on meta device
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            # Check if the parameter is a custom tensor type
            if hasattr(param, "tensor_name") and hasattr(param, "init_parameter"):
                # Create a new parameter of the same type
                param_data = param.new_empty(param.shape, device="cuda")
                # Initialize the parameter
                param.init_parameter(param_data, tensor_space.distributed)
                # Replace the parameter in the module
                module_path, param_name = name.rsplit(".", 1)
                module = model
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


@pytest.fixture
def hybrid_config():
    config = HybridModelConfig(
        transformer=TransformerConfig(num_layers=4),
        mamba_rms_norm=True,
        mamba_residual_in_fp32=True,
        mamba_fused_add_norm=True,
        block_pattern=["t", "m", "t", "m"],
        init_method_std_embed=0.02,
        init_method_min_embed=-0.02,
        init_method_max_embed=0.02,
        use_position_embeddings=True,
        tie_word_embeddings=False,
        use_fast_path=True,
    )
    return config


# @pytest.fixture
# def tensor_space(distributed_config):
#     tensor_space = TensorSpace(distributed_config)
#     tensor_space.setup(Distributed(config=distributed_config))
#     return tensor_space


@pytest.fixture
def mamba_config():
    config = MambaConfig(device="cuda")
    # config.setup_tensor_space(TensorSpace(config))
    return config


# def test_mamba_layer(distributed_config, distributed, mamba_config):
#     # Initialize layer

#     layer = MambaLayer(mamba_config)
#     tensor_space = TensorSpace(distributed_config=distributed_config)
#     tensor_space.setup(distributed)
#     materialize_meta_tensors(layer, tensor_space)
#     layer.to(distributed.device)


#     batch_size = 2
#     seq_length = 32
#     hidden_size = mamba_config.hidden_size
#     x = torch.randn(batch_size, seq_length, hidden_size, device=distributed.device)

#     # Run forward pass
#     output = layer(x)

#     loss = output.sum()
#     loss.backward()
#     # Basic shape checkss
#     assert output.shape == x.shape
#     assert not torch.isnan(output).any()
#     assert not torch.isinf(output).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available")
def test_mamba_block(distributed_config, distributed, mamba_config):

    factory_kwargs = {}

    norm_cls = partial(LayerNorm if not mamba_config.rms_norm else RMSNorm, eps=mamba_config.layernorm_epsilon)
    layer_idx = 0

    mixer_cls = partial(MambaLayer, layer_idx=layer_idx, **factory_kwargs)
    block = MambaBlock(
        mamba_config,
        mixer_cls=mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=mamba_config.fused_add_norm,
        residual_in_fp32=mamba_config.residual_in_fp32,
    )

    tensor_space = TensorSpace(distributed_config=distributed_config)
    tensor_space.setup(distributed)

    # model = MultiStageModel(mamba_config, distributed, tensor_space)

    materialize_meta_tensors(block, tensor_space)
    block.to("cuda")

    batch_size = 2
    seq_length = 32
    hidden_size = mamba_config.hidden_size
    x = torch.randn(batch_size, seq_length, hidden_size, device=distributed.device)

    hidden_states, residual = block(x)
    loss = hidden_states.sum()
    loss.backward()

    assert hidden_states.shape == x.shape
    assert not torch.isnan(hidden_states).any()
    assert not torch.isinf(hidden_states).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available")
def test_hybrid_model_train_with_fast_mode(distributed_config, hybrid_config):
    print(hybrid_config)
    model = HybridBaseModel(hybrid_config, distributed_config)
    tensor_space = TensorSpace(distributed_config=distributed_config)
    distributed = Distributed(distributed_config)
    model.setup(distributed)
    tensor_space.setup(distributed)
    materialize_meta_tensors(model, tensor_space)
    model.to("cuda")
    # print(model)

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available")
def test_hybrid_model_slow_mode(distributed_config, hybrid_config):
    print(hybrid_config)
    hybrid_config.use_fast_path = False
    model = HybridBaseModel(hybrid_config, distributed_config)
    tensor_space = TensorSpace(distributed_config=distributed_config)
    distributed = Distributed(distributed_config)
    model.setup(distributed)
    tensor_space.setup(distributed)
    materialize_meta_tensors(model, tensor_space)
    model.to("cuda")
    # print(model)

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


@dataclass
class InferenceParams:
    max_seqlen: int
    max_batch_size: int
    sequence_len_offset: int = 0
    key_value_memory_dict: dict = None

    def __post_init__(self):
        if self.key_value_memory_dict is None:
            self.key_value_memory_dict = {}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available")
def test_hybrid_model_inference(distributed_config, hybrid_config):
    print(hybrid_config)
    hybrid_config.use_fast_path = False
    model = HybridBaseModel(hybrid_config, distributed_config)
    tensor_space = TensorSpace(distributed_config=distributed_config)
    distributed = Distributed(distributed_config)
    model.setup(distributed)
    tensor_space.setup(distributed)
    materialize_meta_tensors(model, tensor_space)
    model.to("cuda")
    # print(model)

    batch_size = 2
    seq_length = 32
    x = torch.randint(0, 49152, (batch_size, seq_length), device="cuda")
    position_ids = torch.arange(seq_length, device="cuda", dtype=torch.int64)
    attention_mask = torch.ones((1, 1, 1, 1), device="cuda", dtype=torch.bool)  # will be broadcasted to right shape
    labels = torch.randint(0, 49152, (batch_size, seq_length), device="cuda")
    max_new_tokens = 10

    inference_params = InferenceParams(
        max_seqlen=len(x[0]) + max_new_tokens, max_batch_size=x.shape[0], sequence_len_offset=0
    )
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
            "inference_params": inference_params,
        },
        losses=losses,
    )


if __name__ == "__main__":
    pytest.main([__file__])
