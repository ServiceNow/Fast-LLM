import pytest
import torch

from fast_llm.data.data.gpt.data import prepare_masked_batch


@pytest.mark.parametrize(
    "data_ids, positions, padded, mask_token_id, vocab_size, context_length, p_mask, p_uniform, ar_factor, un_factor, last_factor, expected",
    [
        # Shift + Masked diffissuion test case
        (
            torch.tensor([[42, 67, 76, 14, 26]]),
            torch.tensor([[0, 1, 2, 3]]),
            torch.tensor([[False, False, False, False, False]]),
            100,
            100,
            -torch.ones(1, dtype=torch.int),
            torch.Tensor([0.51]),
            0.0,
            0.0,
            0.0,
            0.0,
            {
                "in_context": torch.tensor([[False, False, False, False]]),
                "in_mask": torch.tensor([[False, False, True, False]]),
                "in_uniform": torch.tensor([[False, False, False, False]]),
                "in_clean": torch.tensor([[True, True, False, True]]),
                "input_ids": torch.tensor([[42, 67, 100, 14]]),
                "loss_weights": torch.tensor([[0.0, 1.9608, 0.0, 0.0]]),
            },
        ),
        # Shift + AR context + Masked diffusion test case
        (
            torch.tensor([[42, 67, 76, 14, 26, 42, 67, 76, 26]]),
            torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]]),
            torch.tensor([[False, False, False, False, False, False, False, False, False]]),
            100,
            100,
            torch.ones(1, dtype=torch.int),
            torch.Tensor([0.51]),
            0.0,
            1.0,
            0.0,
            0.0,
            {
                "in_context": torch.tensor([[True, True, False, False, False, False, False, False]]),
                "in_mask": torch.tensor([[False, False, True, False, True, False, True, False]]),
                "in_uniform": torch.tensor([[False, False, False, False, False, False, False, False]]),
                "in_clean": torch.tensor([[False, False, False, True, False, True, False, True]]),
                "input_ids": torch.tensor([[42, 67, 100, 14, 100, 42, 100, 76]]),
                "loss_weights": torch.tensor([[1.0000, 1.9608, 0.0000, 1.9608, 0.0000, 1.9608, 0.0000, 0.0000]]),
            },
        ),
        # Shift + AR context + Masked diffusion + Uniform flip test case
        (
            torch.tensor([[42, 67, 76, 14, 26, 42, 67, 76, 26]]),
            torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]]),
            torch.tensor([[False, False, False, False, False, False, False, False, False]]),
            100,
            100,
            torch.ones(1, dtype=torch.int),
            torch.Tensor([0.51]),
            1.0,
            1.0,
            1.0,
            0.0,
            {
                "in_context": torch.tensor([[True, True, False, False, False, False, False, False]]),
                "in_mask": torch.tensor([[False, False, True, False, True, False, True, False]]),
                "in_uniform": torch.tensor([[False, False, False, True, False, True, False, True]]),
                "in_clean": torch.tensor([[False, False, False, False, False, False, False, False]]),
                "input_ids": torch.tensor(
                    [[42, 67, 100, 6, 100, 76, 100, 11]]
                ),  # new uniformly shuffled tokens 14->6 67->76 26->11
                "loss_weights": torch.tensor([[1.0000, 1.9608, 0.0000, 1.9608, 0.0000, 1.9608, 0.0000, 0.0000]]),
            },
        ),
    ],
)
def test_prepare_batch_basic(
    data_ids,
    positions,
    padded,
    mask_token_id,
    vocab_size,
    context_length,
    p_mask,
    p_uniform,
    ar_factor,
    un_factor,
    last_factor,
    expected,
):
    torch.manual_seed(42)  # For reproducibility
    batch = prepare_masked_batch(
        data_ids,
        positions,
        padded,
        mask_token_id,
        vocab_size,
        context_length,
        p_mask,
        p_uniform,
        ar_factor,
        un_factor,
        last_factor,
    )

    for key, value in expected.items():
        if key == "loss_weights":
            assert torch.allclose(batch[key], value, atol=1e-4), f"{key} mismatch: {batch[key]} != {value}"
        else:
            assert torch.equal(batch[key], value), f"{key} mismatch: {batch[key]} != {value}"
