import torch
import pytest
from fast_llm.data.data.gpt.data import prepare_batch

def get_data(batch_size, length, vocab_size):
    # `length` here excludes the first token (which acts like a <BOS>), hence the +1
    data_ids = torch.randint(0, vocab_size, (batch_size, length+1))
    padded = torch.zeros(batch_size, length+1, dtype=torch.bool)
    positions = torch.arange(length).unsqueeze(0).expand(batch_size, length)
    return data_ids, padded, positions

@pytest.mark.parametrize("data_ids, positions, padded, mask_token_id, vocab_size, context_length, p_mask, p_uniform", [
    (torch.tensor([[42, 67, 76, 14, 26]]),
     torch.tensor([[0, 1, 2, 3]]),
     torch.tensor([[False, False, False, False, False]]),
     100,
     100,
     -torch.ones(1, dtype=torch.int),
     torch.tensor([[0.5100]]),
     0.0,
     ),
])
def test_prepare_batch_basic(data_ids, positions, padded, mask_token_id, vocab_size, context_length, p_mask, p_uniform):
    torch.manual_seed(42)  # For reproducibility
    batch = prepare_batch(
        data_ids, positions, padded, mask_token_id, vocab_size, context_length, p_mask, p_uniform
    )
    expected = {
        'in_context': torch.tensor([[True, True, False, False]]),
        'in_mask': torch.tensor([[False, False, False, False]]),
        'in_uniform': torch.tensor([[False, False, False, False]]),
        'in_clean': torch.tensor([[False, False, True, True]]),
        'input_ids': torch.tensor([[42, 67, 76, 14]]),
        'target_ids': torch.tensor([[42, 67, 76, 14, 26]]),
        'loss_weights': torch.tensor([[1., 0., 0., 0.]])
    }
    for key, value in expected.items():
        assert torch.equal(batch[key], value), f"{key} mismatch: {batch[key]} != {value}"