import pytest
import torch

from fast_llm.layers.language_model.preprocessing import LLaDAMaskingPreprocessor
from fast_llm.layers.transformer.config import DiffusionMaskingConfig


@pytest.fixture
def masking_config():
    return DiffusionMaskingConfig(
        enabled=True,
        epsilon=0.1,
        max_mask_prob=0.5,
        pad_prob=0.1,
        mask_token_id=103
    )


def test_masking_basic():
    config = DiffusionMaskingConfig(
        enabled=True,
        epsilon=0.15,  # 15% minimum masking
        max_mask_prob=0.5,  # 50% maximum masking
        pad_prob=0.1,
        mask_token_id=103
    )
    
    preprocessor = LLaDAMaskingPreprocessor(config)
    
    batch_size = 4
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    input_ids[:, -2:] = 0  # Add padding at the end
    
    outputs = preprocessor(input_ids)
    
    masked_indices = outputs['masked_indices']
    p_mask = outputs['p_mask']
    masked_input = outputs['input_ids']
    
    assert masked_indices.shape == input_ids.shape
    assert p_mask.shape == input_ids.shape
    assert masked_input.shape == input_ids.shape
    
    padding_positions = (input_ids == 0)
    assert not masked_indices[padding_positions].any()
    assert (p_mask[padding_positions] == 0).all()
    
    non_pad_positions = ~padding_positions
    assert (p_mask[non_pad_positions] >= config.epsilon).all()
    assert (p_mask[non_pad_positions] <= config.max_mask_prob).all()
    
    assert (masked_input[masked_indices] == config.mask_token_id).all()
    
    unmasked_positions = ~masked_indices & non_pad_positions
    assert (masked_input[unmasked_positions] == input_ids[unmasked_positions]).all()


def test_masking_edge_cases():
    config = DiffusionMaskingConfig(
        enabled=True,
        epsilon=0.1,
        max_mask_prob=0.5,
        pad_prob=0.1,
        mask_token_id=103
    )
    
    preprocessor = LLaDAMaskingPreprocessor(config)
    
    input_ids = torch.randint(0, 1000, (1, 5))
    outputs = preprocessor(input_ids)
    assert outputs['masked_indices'].shape == (1, 5)
    assert outputs['p_mask'].shape == (1, 5)
    
    input_ids = torch.zeros(2, 4)
    outputs = preprocessor(input_ids)
    assert not outputs['masked_indices'].any()  # No tokens should be masked
    assert (outputs['p_mask'] == 0).all()  # All masking probs should be 0
    
    input_ids = torch.randint(1, 1000, (2, 4))  # All tokens are non-padding
    outputs = preprocessor(input_ids)
    assert outputs['masked_indices'].any()  # Some tokens should be masked
    assert (outputs['p_mask'] >= config.epsilon).all()  # All probs should be >= epsilon
    
    input_ids = torch.randint(1, 1000, (1, 1))
    outputs = preprocessor(input_ids)
    assert outputs['masked_indices'].shape == (1, 1)
    assert outputs['p_mask'].shape == (1, 1)


def test_masking_probabilities():
    config = DiffusionMaskingConfig(
        enabled=True,
        epsilon=0.1,
        max_mask_prob=0.5,
        pad_prob=0.1,
        mask_token_id=103
    )
    
    preprocessor = LLaDAMaskingPreprocessor(config)
    
    input_ids = torch.ones(3, 8) 
    input_ids[0, :] = torch.arange(1, 9)  # Increasing sequence
    input_ids[1, :] = torch.arange(8, 0, -1)  # Decreasing sequence
    input_ids[2, :] = 1  # Constant sequence
    
    n_trials = 100
    mask_counts = torch.zeros_like(input_ids)
    
    for _ in range(n_trials):
        outputs = preprocessor(input_ids)
        mask_counts += outputs['masked_indices'].float()
    
    empirical_probs = mask_counts / n_trials
    
    assert (empirical_probs >= config.epsilon - 0.05).all()  # Allow small deviation
    assert (empirical_probs <= config.max_mask_prob + 0.05).all()


def test_masking_deterministic():
    config = DiffusionMaskingConfig(
        enabled=True,
        epsilon=0.1,
        max_mask_prob=0.5,
        pad_prob=0.1,
        mask_token_id=103
    )
    
    preprocessor = LLaDAMaskingPreprocessor(config)
    

    torch.manual_seed(42)
    

    input_ids = torch.randint(1, 1000, (2, 6))
    
    torch.manual_seed(42)
    outputs1 = preprocessor(input_ids)
    
    torch.manual_seed(42)
    outputs2 = preprocessor(input_ids)
    
    assert torch.equal(outputs1['masked_indices'], outputs2['masked_indices'])
    assert torch.equal(outputs1['p_mask'], outputs2['p_mask'])
    assert torch.equal(outputs1['input_ids'], outputs2['input_ids'])


def test_masking_config_validation():
    with pytest.raises(ValueError):
        DiffusionMaskingConfig(
            enabled=True,
            epsilon=-0.1,  # Invalid negative value
            max_mask_prob=0.5,
            pad_prob=0.1,
            mask_token_id=103
        )
    
    with pytest.raises(ValueError):
        DiffusionMaskingConfig(
            enabled=True,
            epsilon=0.1,
            max_mask_prob=1.5,  # Invalid value > 1
            pad_prob=0.1,
            mask_token_id=103
        )
    
    with pytest.raises(ValueError):
        DiffusionMaskingConfig(
            enabled=True,
            epsilon=0.6,  # Greater than max_mask_prob
            max_mask_prob=0.5,
            pad_prob=0.1,
            mask_token_id=103
        )
    
    with pytest.raises(ValueError):
        DiffusionMaskingConfig(
            enabled=True,
            epsilon=0.1,
            max_mask_prob=0.5,
            pad_prob=-0.1,  # Invalid negative value
            mask_token_id=103
        ) 