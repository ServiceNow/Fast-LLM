import logging
import typing

import torch

from fast_llm.engine.base_model.config import Preprocessor
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.layers.ssm.config import SSMKwargs
from fast_llm.layers.transformer.config import TransformerKwargs
from fast_llm.models.ssm.config import HybridSSMBaseModelConfig
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class Mamba2Preprocessor(Preprocessor):
    def __init__(self, config: HybridSSMBaseModelConfig, tensor_space: TensorSpace):
        self._config = config
        self._tensor_space = tensor_space
        self._distributed_config = self._tensor_space.distributed_config
        self._transformer_dim_names = config.transformer._transformer_dim_names

    def preprocess(self, batch, kwargs: dict[str, typing.Any]) -> None:
        """
        Simplified preprocessor that does not take into account micro-sequences.
        """
        if TransformerKwargs.sequence_lengths not in kwargs:
            return
        sequence_lengths = kwargs[TransformerKwargs.sequence_lengths]
        if TransformerKwargs.cu_seqlens_k in kwargs:
            # already set this in the transformer preprocessor, so we can use it here
            cu_seqlens_k = kwargs[TransformerKwargs.cu_seqlens_k]
            cu_seqlens_q = kwargs[TransformerKwargs.cu_seqlens_q]
            Assert.eq(
                cu_seqlens_k.shape[0],
                cu_seqlens_q.shape[0],
                msg="cu_seqlens_k and cu_seqlens_q have different lengths, is micro_sequence_length being used? This is currently not supported for Mamba.",
            )
            Assert.all_equal(cu_seqlens_k, cu_seqlens_q)
            cu_seqlens = cu_seqlens_k
        else:
            seqlens = torch.cat(sequence_lengths)
            cu_seqlens = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int32, device=self._tensor_space.distributed.device),
                    torch.cumsum(seqlens, dim=0, dtype=torch.int32).to(self._tensor_space.distributed.device),
                )
            )
        kwargs[SSMKwargs.cu_seqlens] = cu_seqlens
        # from https://github.com/jxiw/M1/blob/d92b53faa640f8ebf624d3e9e771fe24648ef014/rl/verl/verl/models/mamba/hybrid_wrapper.py#L152
        kwargs[SSMKwargs.seq_idx] = torch.cat(
            [
                torch.full((s,), i, dtype=torch.int32, device=cu_seqlens.device)
                for i, s in enumerate(cu_seqlens[1:] - cu_seqlens[:-1])
            ],
            dim=0,
        ).unsqueeze(0)

        sequence_lengths = kwargs.get(TransformerKwargs.sequence_lengths)
        sequence_k = kwargs[TransformerKwargs.sequence_k_dim].size
        sequence_q = kwargs[TransformerKwargs.sequence_q_dim].size
        position_ids = torch.stack(
            [torch.cat([torch.arange(x) for x in sample_lens]) for sample_lens in sequence_lengths]
        ).to(self._tensor_space.distributed.device, dtype=torch.int64)
        position_ids = position_ids[
            :, sequence_k - sequence_q : sequence_k
        ]  # this is only needed if we do micro-sequences?
        kwargs[SSMKwargs.ssm_position_ids] = position_ids.to(torch.int32)
