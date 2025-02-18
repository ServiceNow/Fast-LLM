import pytest
import torch

from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.ssm.config import MambaConfig
from fast_llm.layers.ssm.mamba2_layer import MambaLayer

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

# @pytest.fixture
# def tensor_space(distributed_config):
#     tensor_space = TensorSpace(distributed_config)
#     tensor_space.setup(Distributed(config=distributed_config))
#     return tensor_space

@pytest.fixture
def ssm_config():
    return MambaConfig()

def test_ssm_forward(distributed, ssm_config):
    # Initialize layer
    layer = MambaLayer(ssm_config)
    
    # Create dummy input
    batch_size = 2
    seq_length = 32
    hidden_size = ssm_config.hidden_size
    x = torch.randn(batch_size, seq_length, hidden_size, device=distributed.device)
    
    # Run forward pass
    output = layer(x, {})
    
    # Basic shape checks
    assert output.shape == x.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

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
    test_ssm_forward(distributed, ssm_config)