import pytest
import torch

from torch import nn
from functools import partial
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.ssm.config import MambaConfig
from fast_llm.layers.ssm.mamba2_layer import Mamba2Layer
from fast_llm.layers.ssm.mamba_layer import MambaLayer
from fast_llm.layers.ssm.mamba_block import MambaBlock
from fast_llm.models.ssm.model import HybridBaseModel, HybridModelConfig
from fast_llm.layers.transformer.config import TransformerArchitectureConfig, TransformerConfig
try:
    from ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def materialize_meta_tensors(model, tensor_space):
    # Initialize parameters that are on meta device
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            # Check if the parameter is a custom tensor type
            if hasattr(param, 'tensor_name') and hasattr(param, 'init_parameter'):
                # Create a new parameter of the same type
                param_data = param.new_empty(param.shape, device="cuda")
                # Initialize the parameter
                param.init_parameter(param_data, tensor_space.distributed)
                # Replace the parameter in the module
                module_path, param_name = name.rsplit('.', 1)
                module = model
                for part in module_path.split('.'):
                    module = getattr(module, part)
                module._parameters[param_name] = torch.nn.Parameter(param_data, requires_grad=param.requires_grad)
            else:
                # Fallback for regular parameters
                try:
                    param.data = torch.empty(param.shape, device="cuda", dtype=param.dtype)
                    torch.nn.init.normal_(param.data, mean=0.0, std=0.02)
                except RuntimeError:
                    # If direct assignment fails, create a new parameter
                    module_path, param_name = name.rsplit('.', 1)
                    module = model
                    for part in module_path.split('.'):
                        module = getattr(module, part)
                    new_param = torch.nn.Parameter(
                        torch.normal(0.0, 0.02, param.shape, device="cuda", dtype=param.dtype),
                        requires_grad=param.requires_grad
                    )
                    module._parameters[param_name] = new_param
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
        block_pattern=['t', 'm', 't', 'm'],
        init_method_std_embed=0.02,
        init_method_min_embed=-0.02,
        init_method_max_embed=0.02,
        use_position_embeddings=True,
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

# def test_mamba2_layer_forward(distributed, ssm_config):
#     # Initialize layer
    
#     layer = Mamba2Layer(ssm_config)
#     layer.to(distributed.device)
    
#     # Create dummy input
#     batch_size = 2
#     seq_length = 32
#     hidden_size = ssm_config.hidden_size
#     x = torch.randn(batch_size, seq_length, hidden_size, device=distributed.device)
    
#     # Run forward pass
#     output = layer(x)
    
#     # Basic shape checks
#     assert output.shape == x.shape
#     assert not torch.isnan(output).any()
#     assert not torch.isinf(output).any()


# def test_mamba1_layer(distributed, mamba_config):
#     # Initialize layer
    
#     layer = MambaLayer(mamba_config)
#     layer.to(distributed.device)
    
#     # Create dummy input
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

# def test_mamba_block(distributed, mamba_config):

#     factory_kwargs = {}
    
#     norm_cls = partial(nn.LayerNorm if not mamba_config.rms_norm else RMSNorm, eps=mamba_config.layernorm_epsilon)
#     layer_idx = 0

#     mixer_cls = partial(MambaLayer, layer_idx=layer_idx, **factory_kwargs)
#     block = MambaBlock(
#         mamba_config,
#         mixer_cls=mixer_cls,
#         norm_cls=norm_cls,
#         fused_add_norm=mamba_config.fused_add_norm,
#         residual_in_fp32=mamba_config.residual_in_fp32,
#         )
#     block.to(distributed.device)

#     batch_size = 2
#     seq_length = 32
#     hidden_size = mamba_config.hidden_size
#     x = torch.randn(batch_size, seq_length, hidden_size, device=distributed.device)

#     hidden_states, residual = block(x)
#     loss = hidden_states.sum()
#     loss.backward()

#     assert hidden_states.shape == x.shape
#     assert not torch.isnan(hidden_states).any()
#     assert not torch.isinf(hidden_states).any()


def test_hybrid_model(distributed_config, hybrid_config):
    print(hybrid_config)
    model = HybridBaseModel(hybrid_config, distributed_config)
    tensor_space = TensorSpace(distributed_config=distributed_config)
    tensor_space.setup(Distributed(distributed_config))
    materialize_meta_tensors(model, tensor_space)
    model.to("cuda")
    # print(model)

    batch_size = 2
    seq_length = 32
    hidden_size = hybrid_config.transformer.hidden_size
    x = torch.randint(0, 49152, (batch_size, seq_length), device="cuda")
    position_ids = torch.arange(seq_length, device="cuda")
    output = model(x, {"position_ids": position_ids})
    print(output)

# def test_ssm_backward(distributed, tensor_space, ssm_config):
#     # Initialize layer
#     layer = SSMLayer(ssm_config, tensor_space)
    
#     # Create dummy input with gradients
#     batch_size = 2
#     seq_length = 32
#     hidden_size = ssm_config.hidden_size
#     x = torch.randn(
#         batch_size, seq_length, hidden_size, 
#         device=distributed.device, 
#         requires_grad=True
#     )
    
#     # Run forward and backward pass
#     output = layer(x, {})
#     loss = output.sum()
#     loss.backward()
    
#     # Check gradients
#     assert x.grad is not None
#     assert not torch.isnan(x.grad).any()
#     assert not torch.isinf(x.grad).any()

# @pytest.mark.parametrize("tensor_parallel", [1, 2])
# def test_ssm_tensor_parallel(tensor_parallel):
#     # Create distributed config with tensor parallelism
#     config = DistributedConfig(
#         tensor_parallel=tensor_parallel,
#         pipeline_parallel=1,
#         sequence_data_parallel=1,
#         local_world_size=max(tensor_parallel, 1),
#         world_size=max(tensor_parallel, 1),
#     )
    
#     distributed = Distributed(config=config)
#     tensor_space = TensorSpace(config)
#     tensor_space.setup(distributed)
    
#     # Initialize layer
#     ssm_config = SSMConfig(
#         hidden_size=64,
#         state_size=16,
#         num_blocks=4,
#         dt_min=0.001,
#         dt_max=0.1,
#         activation="gelu"
#     )
#     layer = SSMLayer(ssm_config, tensor_space)
    
#     # Create dummy input
#     batch_size = 2
#     seq_length = 32
#     x = torch.randn(
#         batch_size, seq_length, ssm_config.hidden_size, 
#         device=distributed.device
#     )
    
#     # Run forward pass
#     output = layer(x, {})
    
#     # Check output shape
#     assert output.shape == x.shape

# @pytest.mark.parametrize("sequence_length", [16, 32, 64])
# def test_ssm_variable_sequence_length(distributed, tensor_space, ssm_config, sequence_length):
#     # Initialize layer
#     layer = SSMLayer(ssm_config, tensor_space)
    
#     # Create dummy input with different sequence lengths
#     batch_size = 2
#     hidden_size = ssm_config.hidden_size
#     x = torch.randn(
#         batch_size, sequence_length, hidden_size, 
#         device=distributed.device
#     )
    
#     # Run forward pass
#     output = layer(x, {})
    
#     # Check output shape
#     assert output.shape == (batch_size, sequence_length, hidden_size)

if __name__ == "__main__":
    pytest.main([__file__])