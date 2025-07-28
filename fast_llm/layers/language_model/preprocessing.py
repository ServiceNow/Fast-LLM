import logging
import typing

import torch

from fast_llm.engine.base_model.config import Preprocessor
from fast_llm.engine.config_utils.tensor_space import DefaultDimNames, TensorDim, TensorSpace
from fast_llm.layers.language_model.config import LanguageModelBaseConfig, LanguageModelKwargs
from fast_llm.layers.transformer.config import TransformerKwargs
from fast_llm.tensor import TensorMeta
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class PositionEmbeddingPreprocessor(Preprocessor):
    _scalar_dim: TensorDim
    _rotary_embedding_frequencies: torch.Tensor
    _position_ids: torch.Tensor
    _tensor_cache_max_sequence_length: int = -1

    def __init__(
        self,
        config: LanguageModelBaseConfig,
        tensor_space: TensorSpace,
    ):
        self._config = config
        assert config.use_absolute_position_embeddings
        self._tensor_space = tensor_space
        self._distributed_config = self._tensor_space.distributed_config
        self._scalar_dim = self._tensor_space[DefaultDimNames.scalar]

    def _create_tensors(self, sequence_length: int) -> None:
        if sequence_length <= self._tensor_cache_max_sequence_length:
            return
        self._tensor_cache_max_sequence_length = sequence_length

        Assert.leq(sequence_length, self._config.num_absolute_position_embeddings)
        self._position_ids = torch.arange(
            0, sequence_length, device=self._tensor_space.distributed.device, dtype=torch.int64
        )

    def preprocess(self, batch, kwargs: dict[str, typing.Any]) -> None:
        self._create_tensors(kwargs[TransformerKwargs.sequence_length])
        sequence_k = kwargs[TransformerKwargs.sequence_k_dim].size
        sequence_q = kwargs[TransformerKwargs.sequence_q_dim].size
        if (sequence_lengths := kwargs.get(TransformerKwargs.sequence_lengths)) is not None:
            position_ids = torch.stack(
                [torch.cat([torch.arange(x) for x in sample_lens]) for sample_lens in sequence_lengths]
            ).to(self._tensor_space.distributed.device, dtype=torch.int64)
            position_ids = position_ids[:, sequence_k - sequence_q : sequence_k]
            if kwargs[TransformerKwargs.sequence_first]:
                position_ids = position_ids.transpose(0, 1)
            kwargs[LanguageModelKwargs.position_ids] = position_ids
        else:
            kwargs[LanguageModelKwargs.position_ids] = self._position_ids[
                sequence_k - sequence_q : sequence_k
            ].unsqueeze(int(kwargs[TransformerKwargs.sequence_first]))

    def preprocess_meta(self, kwargs: dict[str, typing.Any]) -> None:
        # Position embeddings will be broadcast.
        sequence_q_dim = kwargs[TransformerKwargs.sequence_q_dim]
        kwargs[LanguageModelKwargs.position_ids] = TensorMeta.from_dims(
            (
                (sequence_q_dim, self._scalar_dim)
                if kwargs[TransformerKwargs.sequence_first]
                else (self._scalar_dim, sequence_q_dim)
            ),
            tensor_name=LanguageModelKwargs.position_ids,
            dtype=torch.int64,
        )


class PreferenceSpanPreprocessor(Preprocessor):
    def __init__(self, config: LanguageModelBaseConfig, tensor_space: TensorSpace):
        self._config = config
        self._tensor_space = tensor_space
        self._distributed_config = self._tensor_space.distributed_config
        self._scalar_dim = self._tensor_space[DefaultDimNames.scalar]

    def preprocess_meta(self, kwargs: dict[str, typing.Any]) -> None:
        return

    def preprocess(self, batch, kwargs: dict[str, typing.Any]) -> None:
        sequence_q = kwargs[TransformerKwargs.sequence_q_dim].size
        sequence_k = kwargs[TransformerKwargs.sequence_k_dim].size
        sequence_offset = sequence_k - sequence_q + 1  # +1 for shift in labels

        if LanguageModelKwargs.chosen_spans not in kwargs or LanguageModelKwargs.rejected_spans not in kwargs:
            raise ValueError("Expected chosen spans or rejected spans to be found within the batch.")

        chosen_spans = kwargs[LanguageModelKwargs.chosen_spans]
        chosen_valid_spans = []
        for spans in chosen_spans:
            if not spans.numel():
                continue
            # only keep spans within the sequence or partially within the sequence
            valid_spans = spans[(spans[0] <= sequence_k) & (spans[1] >= sequence_offset)][0]
            if valid_spans.numel():
                # if span is partially within the sequence, truncate parts of spans that are outside of the sequence
                valid_spans[0].clamp_(min=sequence_offset)
                valid_spans[1].clamp_(max=sequence_k)
                valid_spans -= sequence_offset

                chosen_valid_spans.append(valid_spans)
        kwargs[LanguageModelKwargs.chosen_spans] = chosen_valid_spans

        rejected_spans = kwargs[LanguageModelKwargs.rejected_spans]
        rejected_valid_spans = []
        for spans in rejected_spans:
            if not spans.numel():
                continue
            # only keep spans within the sequence or partially within the sequence
            valid_spans = spans[(spans[0] <= sequence_k) & (spans[1] >= sequence_offset)][0]
            if valid_spans.numel():
                # if span is partially within the sequence, truncate parts of spans that are outside of the sequence
                valid_spans[0].clamp_(min=sequence_offset)
                valid_spans[1].clamp_(max=sequence_k)
                valid_spans -= sequence_offset

                rejected_valid_spans.append(valid_spans)
        kwargs[LanguageModelKwargs.rejected_spans] = rejected_valid_spans
