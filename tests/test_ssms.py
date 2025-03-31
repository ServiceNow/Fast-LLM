from functools import partial

import pytest
import torch

from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.common.normalization import LayerNorm, RMSNorm
from fast_llm.layers.language_model.config import LanguageModelKwargs, LanguageModelLossNames
from fast_llm.layers.transformer.config import TransformerConfig, TransformerKwargs

try:
    from fast_llm.layers.ssm.config import MambaConfig
    from fast_llm.layers.ssm.mamba_block import MambaBlock
    from fast_llm.layers.ssm.mamba_layer import MambaLayer
    from fast_llm.models.ssm.model import HybridBaseModel, HybridBaseModelConfig
except ImportError:
    MambaLayer, MambaBlock, HybridBaseModel, HybridBaseModelConfig = None, None, None, None
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


@pytest.fixture
def hybrid_config():
    config = HybridBaseModelConfig(
        transformer=TransformerConfig(num_layers=4),
        ssm=MambaConfig(rms_norm=True, residual_in_fp32=True, fused_add_norm=True),
        block_pattern=["t", "m", "t", "m"],
        init_method_std_embed=0.02,
        init_method_min_embed=-0.02,
        init_method_max_embed=0.02,
        use_position_embeddings=True,
        tie_word_embeddings=False,
        use_fast_path=True,
    )
    return config


@pytest.mark.skipif(not run_test, reason="No CUDA available or Mamba not installed")
def test_mamba_layer(distributed_config, distributed, hybrid_config):

    tensor_space = TensorSpace(distributed_config=distributed_config)
    hybrid_config.setup_tensor_space(tensor_space)
    layer = MambaLayer(hybrid_config.ssm, layer_idx=0, tensor_space=tensor_space)
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
def test_mamba_block(distributed_config, distributed, hybrid_config):

    tensor_space = TensorSpace(distributed_config=distributed_config)
    tensor_space.setup(distributed)
    hybrid_config.setup_tensor_space(tensor_space)

    norm_cls = partial(
        LayerNorm if not hybrid_config.ssm.rms_norm else RMSNorm, eps=hybrid_config.ssm.layernorm_epsilon
    )
    layer_idx = 0

    mixer_cls = partial(MambaLayer, layer_idx=layer_idx)
    block = MambaBlock(
        hybrid_config.ssm, mixer_cls=mixer_cls, norm_cls=norm_cls, tensor_space=tensor_space, layer_index=layer_idx
    )

    materialize_meta_tensors(block, tensor_space)
    block.to("cuda")

    batch_size = 2
    seq_length = 32
    hidden_size = hybrid_config.transformer.hidden_size
    x = torch.randn(batch_size, seq_length, hidden_size, device=distributed.device)

    hidden_states, residual = block(x)
    loss = hidden_states.sum()
    loss.backward()

    assert hidden_states.shape == x.shape
    assert not torch.isnan(hidden_states).any()
    assert not torch.isinf(hidden_states).any()


@pytest.mark.skipif(not run_test, reason="No CUDA available or Mamba not installed")
def test_hybrid_model_train_with_fast_mode(distributed_config, hybrid_config):
    # hybrid_config_dict = hybrid_config.to_dict()

    model = HybridBaseModel(hybrid_config, distributed_config)
    distributed = Distributed(distributed_config)
    model.setup(distributed)
    tensor_space = model._tensor_space
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


# TODO: added tghis whgen inference enabled
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
#     model = HybridBaseModel(hybrid_config, distributed_config)
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
    pytest.main([__file__])
