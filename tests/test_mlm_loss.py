import pytest
import torch
import torch.nn.functional as F

from fast_llm.layers.language_model.head import MLMHead
from fast_llm.layers.language_model.config import LanguageModelBaseConfig
from fast_llm.engine.config_utils.tensor_space import TensorSpace, DefaultDimNames, TensorDim
from fast_llm.layers.transformer.config import TransformerConfig, DiffusionMaskingConfig as ModelDiffusionMaskingConfig
from fast_llm.data.dataset.gpt.config import DiffusionMaskingConfig
from fast_llm.engine.distributed.config import DistributedConfig


def to_model_diffusion_config(data_cfg):
    # Only copy the fields that exist in the model config
    return ModelDiffusionMaskingConfig(
        enabled=data_cfg.enabled,
        # Set bidirectional_attention to default or as needed
        bidirectional_attention=True
    )


@pytest.fixture
def mlm_config():
    from fast_llm.data.dataset.gpt.config import DiffusionMaskingConfig
    data_diffusion_cfg = DiffusionMaskingConfig(
        enabled=True,
        epsilon=0.1,
        max_mask_prob=0.5,
        pad_prob=0.1,
        mask_token_id=103
    )
    transformer_config = TransformerConfig(
        hidden_size=768,
        num_layers=12,
        num_attention_heads=12,
        diffusion=to_model_diffusion_config(data_diffusion_cfg)
    )
    return LanguageModelBaseConfig(
        vocab_size=30522,
        transformer=transformer_config,
        tie_word_embeddings=False,
        parallel_embeddings=False,
        prediction_heads=1
    )


@pytest.fixture
def tensor_space():
    distributed_config = DistributedConfig()
    tensor_space = TensorSpace(distributed_config)
    tensor_space.add_tensor_dim(DefaultDimNames.scalar, 1)
    tensor_space.add_tensor_dim("hidden", 768)
    tensor_space.add_tensor_dim("vocab", 30522)
    return tensor_space


def test_mlm_loss_computation(mlm_config, tensor_space):
    mlm_head = MLMHead(mlm_config, tensor_space, prediction_distance=0)
    
    batch_size = 4
    seq_len = 8
    hidden_size = 768
    vocab_size = 30522
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    masked_indices = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    masked_indices[:, [2, 5]] = True  # Mask positions 2 and 5 in each sequence
    
    p_mask = torch.full((batch_size, seq_len), 0.15)  # 15% masking probability
    
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    kwargs = {
        'masked_indices': masked_indices,
        'p_mask': p_mask,
        'labels': labels
    }
    
    losses = {}
    output = mlm_head(hidden_states, kwargs, losses)
    

    assert output is not None
    assert isinstance(output, torch.Tensor)
    assert output.requires_grad
    

    assert losses  # losses dictionary should not be empty
    
    # Test with no masked positions
    kwargs['masked_indices'] = torch.zeros_like(masked_indices)
    losses = {}
    output_no_masks = mlm_head(hidden_states, kwargs, losses)
    assert output_no_masks is not None
    
    # Test with all positions masked
    kwargs['masked_indices'] = torch.ones_like(masked_indices)
    losses = {}
    output_all_masked = mlm_head(hidden_states, kwargs, losses)
    assert output_all_masked is not None


def test_mlm_loss_edge_cases(mlm_config, tensor_space):
    mlm_head = MLMHead(mlm_config, tensor_space, prediction_distance=0)
    
    hidden_states = torch.randn(1, 4, 768)
    masked_indices = torch.zeros(1, 4, dtype=torch.bool)
    masked_indices[0, 1] = True
    p_mask = torch.full((1, 4), 0.15)
    labels = torch.randint(0, 30522, (1, 4))
    
    kwargs = {
        'masked_indices': masked_indices,
        'p_mask': p_mask,
        'labels': labels
    }
    
    losses = {}
    output = mlm_head(hidden_states, kwargs, losses)
    assert output is not None
    
    p_mask = torch.full((1, 4), 0.01)
    kwargs['p_mask'] = p_mask
    losses = {}
    output = mlm_head(hidden_states, kwargs, losses)
    assert output is not None
    
    p_mask = torch.full((1, 4), 0.5)  # max_mask_prob from config
    kwargs['p_mask'] = p_mask
    losses = {}
    output = mlm_head(hidden_states, kwargs, losses)
    assert output is not None


def test_mlm_loss_backward(mlm_config, tensor_space):
    mlm_head = MLMHead(mlm_config, tensor_space, prediction_distance=0)
    
    hidden_states = torch.randn(2, 6, 768, requires_grad=True)
    masked_indices = torch.zeros(2, 6, dtype=torch.bool)
    masked_indices[:, [1, 4]] = True
    p_mask = torch.full((2, 6), 0.15)
    labels = torch.randint(0, 30522, (2, 6))
    
    kwargs = {
        'masked_indices': masked_indices,
        'p_mask': p_mask,
        'labels': labels
    }
    
    losses = {}
    output = mlm_head(hidden_states, kwargs, losses)
    
    output.backward()
    
    assert hidden_states.grad is not None
    assert not torch.isnan(hidden_states.grad).any()
    assert not torch.isinf(hidden_states.grad).any()
    
    assert hidden_states.grad.shape == hidden_states.shape 