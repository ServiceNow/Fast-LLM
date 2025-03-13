import pytest
import torch
import torch.nn.functional as F

from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.layers.transformer.config import TransformerConfig, TransformerKwargs, RoutingType
from fast_llm.layers.transformer.mixture_of_experts import MixtureOfExpertMLP
from fast_llm.tensor import TensorMeta, TensorDim
from fast_llm.engine.config_utils.logging import TensorLogsConfig, TensorLogs

def materialize_meta_tensors(moe, tensor_space):
    # Initialize parameters that are on meta device
    for name, param in moe.named_parameters():
        if param.device.type == "meta":
            # Check if the parameter is a custom tensor type
            if hasattr(param, 'tensor_name') and hasattr(param, 'init_parameter'):
                # Create a new parameter of the same type
                param_data = param.new_empty(param.shape, device="cuda")
                # Initialize the parameter
                param.init_parameter(param_data, tensor_space.distributed)
                # Replace the parameter in the module
                module_path, param_name = name.rsplit('.', 1)
                module = moe
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
                    module = moe
                    for part in module_path.split('.'):
                        module = getattr(module, part)
                    new_param = torch.nn.Parameter(
                        torch.normal(0.0, 0.02, param.shape, device="cuda", dtype=param.dtype),
                        requires_grad=param.requires_grad
                    )
                    module._parameters[param_name] = new_param
    return moe

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires CUDA")
class TestMoERouting:
    @pytest.fixture
    def moe_setup(self):
        """Setup a basic MoE layer for testing on CPU."""
        transformer_conf = TransformerConfig(
            num_layers=2,
            num_attention_heads=2,
            hidden_size=32,
            num_experts=4,
            num_experts_per_token=2,
            expert_routing_type=RoutingType.top_k_per_example,
            add_linear_biases=False,
            debug_transformer=True,
            expert_auxiliary_loss_coefficient=0.06,
            # dropless_moe=False,
        )
        # Configure for CPU
        distributed_config = DistributedConfig()    
        distributed = Distributed(distributed_config)#, use_cpu=True)
        tensor_space = TensorSpace(distributed_config=distributed_config)
        tensor_space.setup(distributed)
        transformer_conf.setup_tensor_space(tensor_space)
        
        moe = MixtureOfExpertMLP(transformer_conf, tensor_space, "test_moe")
        return moe, tensor_space
    
    @pytest.fixture(autouse=True)  # autouse=True makes this run for all tests
    def setup_logging(self):
        """Setup logging configuration for tests."""
        TensorLogs.config = TensorLogsConfig(
            save=False,
            show=False
        )
        yield
        # Clean up after tests
        TensorLogs.config = None
    
    @pytest.fixture
    def mock_router(self):
        def mock_router_forward(x):
            # Create predictable routing logits based on the input pattern
            batch_size = x.size(0)
            num_experts = 4
            logits = torch.zeros(batch_size, num_experts)
            
            # For inputs with high values in dims 0-3, route to expert 0
            logits[:, 0] = torch.sum(x[:, 0:4], dim=1)
            
            # For inputs with high values in dims 4-7, route to expert 1
            logits[:, 1] = torch.sum(x[:, 4:8], dim=1)
            
            # For inputs with high values in dims 8-11, route to expert 2
            logits[:, 2] = torch.sum(x[:, 8:12], dim=1)
            
            # For inputs with high values in dims 12-15, route to expert 3
            logits[:, 3] = torch.sum(x[:, 12:16], dim=1)
    
            return logits
        return mock_router_forward

    # def test_per_example_topk_routing(self, moe_setup, mock_router):
    #     """Test that per-example routing assigns the same experts to all tokens in an example."""
    #     moe, tensor_space = moe_setup # 4 experts

    #     # Patch the router forward method
    #     # original_router_forward = moe.router.forward
    #     moe.router.forward = mock_router
        
    #     # Setup test data
    #     batch_size = 2
    #     seq_len = 4
    #     hidden_size = 16
        
    #     # Create a sequence dimension
    #     sequence_dim = TensorDim("sequence", seq_len)
        
    #     # Create hidden states with a clear pattern (on CPU)
    #     hidden_states = torch.randn(batch_size, seq_len, hidden_size)+1
    #     hidden_states_batch_first = hidden_states.reshape(-1, hidden_size)
    #     assert (hidden_states_batch_first.view(batch_size, seq_len, hidden_size) == hidden_states).sum() == hidden_states.numel()
        
    #     # Setup kwargs for the routing function
    #     kwargs = {
    #         TransformerKwargs.sequence_q_dim: sequence_dim,
    #         TransformerKwargs.sequence_first: False,
    #     }
        
    #     # Create a losses dict to capture auxiliary losses
    #     losses = {
    #         "load_balancing_loss": [],
    #         "router_z_loss": []
    #     }
        
    #     # Call the per-example routing function
    #     scores, top_experts = moe._per_example_topk_routing(
    #         hidden_states_batch_first, 
    #         grad_output=1.0, 
    #         losses=losses, 
    #         kwargs=kwargs
    #     )
        
    #     # Check shape of outputs
    #     assert scores.shape == (batch_size * seq_len, moe._experts_per_token)
    #     assert top_experts.shape == (batch_size * seq_len, moe._experts_per_token)
        
    #     # Check that all tokens in the same example were routed to the same experts
    #     for b in range(batch_size):
    #         start_idx = b * seq_len
    #         end_idx = (b + 1) * seq_len
            
    #         # All tokens in the same example should have the same routing
    #         for i in range(start_idx + 1, end_idx):
    #             assert torch.all(top_experts[i] == top_experts[start_idx])
    #             assert torch.allclose(scores[i], scores[start_idx])
        
    #     # Check that the first example was routed to experts 0 and 2
    #     # first_example_experts = set(top_experts[0].tolist())
    #     # assert first_example_experts == {0, 2}
        
    #     # # Check that the second example was routed to experts 1 and 3
    #     # second_example_experts = set(top_experts[seq_len].tolist())
    #     # assert second_example_experts == {1, 3}
        
    #     # Check that load balancing loss was computed
    #     # assert len(losses["load_balancing_loss"]) == 1
        
    #     # Test with sequence_first=True
    #     kwargs[TransformerKwargs.sequence_first] = True
        
    #     # Reshape hidden states to be sequence-first
    #     hidden_states_seq_first = hidden_states.transpose(0, 1).reshape(-1, hidden_size)
    #     assert (hidden_states_seq_first.view(seq_len, batch_size, hidden_size) == hidden_states.transpose(0, 1)).sum() == hidden_states.numel()
        
    #     scores_seq_first, top_experts_seq_first = moe._per_example_topk_routing(
    #         hidden_states_seq_first, 
    #         grad_output=1.0, 
    #         losses=losses, 
    #         kwargs=kwargs
    #     )
        
    #     # Reshape back to batch-first for comparison
    #     scores_seq_first = scores_seq_first.reshape(seq_len, batch_size, -1).transpose(0, 1).reshape(-1, moe._experts_per_token)
    #     top_experts_seq_first = top_experts_seq_first.reshape(seq_len, batch_size, -1).transpose(0, 1).reshape(-1, moe._experts_per_token)
        
    #     # Check that results are the same regardless of sequence_first
    #     assert torch.all(top_experts == top_experts_seq_first)
    #     assert torch.allclose(scores, scores_seq_first)
    
    def test_per_example_routing_with_forward(self, moe_setup):
        """Test the full forward pass with per-example routing on CPU."""
        moe, tensor_space = moe_setup
        moe = materialize_meta_tensors(moe, tensor_space)
                        
        moe.to(device="cuda")
        # moe.eval()
        
        # Setup test data
        batch_size = 2
        seq_len = 8
        hidden_size = 32
        
        # Create a sequence dimension
        batch_seq_dim = TensorDim("batch_seq", batch_size * seq_len)
        sequence_dim = TensorDim("sequence", seq_len)
        hidden_dims = TensorDim("hidden", hidden_size)
        
        # Create input tensor on CPU
        input_tensor = torch.randn(batch_size * seq_len, hidden_size, device="cuda")
        
        # Setup kwargs for the forward function
        kwargs = {
            TransformerKwargs.hidden_dims: (batch_seq_dim, hidden_dims),
            TransformerKwargs.sequence_q_dim: sequence_dim,
            TransformerKwargs.sequence_first: False,
        }
        
        # Create a losses dict
        losses = {
            "load_balancing_loss": [],
            "router_z_loss": []
        }
        
        # Call the forward function
        output, _ = moe.forward(input_tensor, kwargs, losses)
        
        # Check output shape
        assert output.shape == input_tensor.shape
        
        # Check that load balancing loss was computed
        assert len(losses["load_balancing_loss"]) == 1

if __name__ == "__main__":
    pytest.main([__file__])