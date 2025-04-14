import pytest
import torch
import numpy as np
from typing import List

from fast_llm.data.data.gpt.data import gpt_data_collate_fn, GPTBatch
from fast_llm.data.dataset.gpt.sampled import GPTSample

class TestGPTDataCollator:
    @pytest.fixture
    def sample_batch(self) -> List[GPTSample]:
        sample1 = GPTSample(
            token_ids=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64),
            loss_masking_spans=np.array([], dtype=np.int64),
            sequence_lengths=np.array([10], dtype=np.int64),
        )
        sample2 = GPTSample(
            token_ids=np.array([11, 12, 13, 14, 15, 0, 0, 0, 0, 0], dtype=np.int64),  # with padding
            loss_masking_spans=np.array([], dtype=np.int64),
            sequence_lengths=np.array([5], dtype=np.int64),
        )
        return [sample1, sample2]

    def test_basic_collation(self, sample_batch):
        """Test basic collation without masking"""
        batch = gpt_data_collate_fn(
            batch=sample_batch,
            use_loss_masking_spans=False,
            cross_document_attention=True,
            random_token_masking=False,
        )

        assert isinstance(batch, GPTBatch)
        assert batch.token_ids.shape == (2, 10)
        assert torch.equal(batch.token_ids[0], torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
        assert torch.equal(batch.token_ids[1], torch.tensor([11, 12, 13, 14, 15, 0, 0, 0, 0, 0]))

    def test_masking_disabled_padding_handling(self, sample_batch):
        """Test that padding tokens are properly handled when masking is disabled"""
        batch = gpt_data_collate_fn(
            batch=sample_batch,
            use_loss_masking_spans=False,
            cross_document_attention=True,
            random_token_masking=False,
        )

        assert torch.equal(batch.token_ids[1, 5:], torch.zeros(5, dtype=torch.long))
        assert torch.equal(batch.attention_mask[1, 5:], torch.ones(5, dtype=torch.long))

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_masking_distribution(self, sample_batch, seed):
        """Test the distribution of masking (80% MASK, 10% random, 10% unchanged)"""
        torch.manual_seed(seed)
        
        vocab_size = 1000
        mask_token_id = 999
        masking_probability = 1.0  
        
        batch = gpt_data_collate_fn(
            batch=sample_batch,
            use_loss_masking_spans=False,
            cross_document_attention=True,
            random_token_masking=True,
            masking_probability=masking_probability,
            mask_token_id=mask_token_id,
            vocab_size=vocab_size,
            mask_replace_prob=0.8,
            random_replace_prob=0.1,
        )

        non_padding = batch.token_ids != 0
        total_tokens = non_padding.sum().item()
        
        masked_tokens = (batch.token_ids == mask_token_id).sum().item()
        original_tokens = torch.eq(batch.token_ids[non_padding], batch.labels[non_padding]).sum().item()
        
        mask_prop = masked_tokens / total_tokens
        original_prop = original_tokens / total_tokens
        random_prop = 1 - mask_prop - original_prop
        
        assert 0.75 <= mask_prop <= 0.85, f"MASK token proportion {mask_prop} outside expected range"
        assert 0.05 <= original_prop <= 0.15, f"Unchanged token proportion {original_prop} outside expected range"
        assert 0.05 <= random_prop <= 0.15, f"Random token proportion {random_prop} outside expected range"

    def test_padding_not_masked(self, sample_batch):
        """Test that padding tokens are not masked"""
        batch = gpt_data_collate_fn(
            batch=sample_batch,
            use_loss_masking_spans=False,
            cross_document_attention=True,
            random_token_masking=True,
            masking_probability=1.0,  # Try to mask everything
            mask_token_id=999,
            vocab_size=1000,
        )

        # Check that padding tokens remain unchanged
        assert torch.equal(batch.token_ids[1, 5:], torch.zeros(5, dtype=torch.long))
        # Check that labels for padding tokens are -100
        assert torch.equal(batch.labels[1, 5:], -100 * torch.ones(5, dtype=torch.long))

    def test_label_handling(self, sample_batch):
        """Test that labels are properly set for masked and unmasked tokens"""
        torch.manual_seed(42)
        
        batch = gpt_data_collate_fn(
            batch=sample_batch,
            use_loss_masking_spans=False,
            cross_document_attention=True,
            random_token_masking=True,
            masking_probability=0.5,
            mask_token_id=999,
            vocab_size=1000,
        )

        non_masked = batch.token_ids != 999
        assert torch.all(batch.labels[non_masked & (batch.token_ids != 0)] == -100)
        
        masked = batch.token_ids == 999
        assert torch.all(batch.labels[masked] != -100)

    def test_invalid_config(self, sample_batch):
        """Test that invalid configurations raise appropriate errors"""
        with pytest.raises(AssertionError):
            gpt_data_collate_fn(
                batch=sample_batch,
                use_loss_masking_spans=False,
                cross_document_attention=True,
                random_token_masking=True, 
                mask_token_id=999,
            )

        with pytest.raises(AssertionError):
            gpt_data_collate_fn(
                batch=sample_batch,
                use_loss_masking_spans=False,
                cross_document_attention=True,
                random_token_masking=True,
                vocab_size=1000, 
            ) 