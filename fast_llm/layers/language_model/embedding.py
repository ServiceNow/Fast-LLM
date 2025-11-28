import typing

import torch

from fast_llm.core.distributed import set_generator
from fast_llm.core.ops import gather, reduce_forward, split
from fast_llm.engine.base_model.config import ResourceUsageConfig
from fast_llm.engine.config_utils.initialization import init_normal_
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.engine.distributed.config import DistributedConfig, DistributedDimNames
from fast_llm.layers.block.block import Block
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.layers.language_model.config import LanguageModelEmbeddingsConfig, LanguageModelKwargs
from fast_llm.tensor import TensorMeta
from fast_llm.utils import Assert

WORD_EMBEDDINGS_WEIGHT = "word_embeddings_weight"


class LanguageModelEmbedding[ConfigType: LanguageModelEmbeddingsConfig](Block[ConfigType]):
    """
    A language model embedding layer.
    Consists of word embeddings (tensor-parallel or sequence-tensor-parallel),
    together with optional absolute position embeddings and dropout.
    """

    # Ensure the layer is on its own stage.
    layer_count: float = 1000.0
    _config: ConfigType

    # Preprocessing
    _rotary_embedding_frequencies: torch.Tensor
    _position_ids: torch.Tensor
    _tensor_cache_max_sequence_length: int = -1

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        hidden_dim: TensorDim,
        lr_scale: float | None,
        peft: PeftConfig | None,
    ):
        super().__init__(
            config,
            distributed_config,
            hidden_dim=hidden_dim,
            lr_scale=lr_scale,
            peft=peft,
        )
        self._residual_dtype = (
            self._distributed_config.optimization_dtype
            if self._config.full_precision_residual
            else self._distributed_config.compute_dtype
        ).torch
        self._sequence_parallel = self._distributed_config.sequence_tensor_parallel
        self._vocab_parallel = self._distributed_config.tensor_parallel > 1 and self._config.vocab_parallel
        self._parallel_dim = self._distributed_config.get_distributed_dim(DistributedDimNames.tensor)
        vocab_dim = TensorDim("vocab", self._config.vocab_size, self._parallel_dim if self._vocab_parallel else None)

        if self._vocab_parallel:
            self._vocab_start_index = self._distributed_config.tensor_rank * vocab_dim.size
            self._vocab_end_index = (self._distributed_config.tensor_rank + 1) * vocab_dim.size

        self.word_embeddings_weight = self._config.word_embeddings.get_parameter(
            (vocab_dim, self._hidden_dim),
            default_initialization=init_normal_(std=self._hidden_size**-0.5),
            lr_scale=self._lr_scale,
            peft=self._peft,
        )
        self.position_embeddings_weight = self._config.position_embeddings.get_parameter(
            (TensorDim("position_embeddings", self._config.num_position_embeddings), self._hidden_dim),
            default_initialization=init_normal_(std=self._hidden_size**-0.5),
            allow_sequence_tensor_parallel=not self._vocab_parallel,
            lr_scale=self._lr_scale,
            peft=self._peft,
        )

    @torch.compile
    def _forward(
        self,
        input_: torch.Tensor | None,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor | None,
        mask_inputs: bool,
        embedding_map: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        Assert.eq(position_ids is None, self.position_embeddings_weight is None)
        group = self._parallel_dim.group
        if self._vocab_parallel:
            token_mask = (token_ids >= self._vocab_start_index) * (token_ids < self._vocab_end_index)
            masked_input = (token_ids - self._vocab_start_index) * token_mask
            embeddings = torch.embedding(self.word_embeddings_weight, masked_input) * token_mask.unsqueeze(2)  # noqa
            embeddings = reduce_forward(embeddings, group)
            # TODO: Input masking of position embeddings inconsistant with non-vocab-parallel
            if self.position_embeddings_weight is not None:
                embeddings = embeddings + torch.nn.functional.embedding(position_ids, self.position_embeddings_weight)

            if input_ is not None:
                # TODO: Accumulate redundant with masking?
                if self._sequence_parallel:
                    input_ = gather(input_, group=group, dim=0)
                # Out-of-place equivalent of `embeddings[embedding_map] += input_`
                embeddings = embeddings.index_put(embedding_map, input_[: embedding_map[0].size(0)], accumulate=True)

            if self._sequence_parallel:
                embeddings = split(embeddings, group=group, dim=0)
        else:
            if self._sequence_parallel:
                token_ids = split(token_ids, group=group, dim=0)
                if self.position_embeddings_weight is not None:
                    position_ids = split(position_ids, group=group, dim=0)
            # handle masked tokens
            if mask_inputs:
                token_mask = token_ids >= 0
                token_ids = token_ids * token_mask
            embeddings = torch.embedding(self.word_embeddings_weight, token_ids)
            if self.position_embeddings_weight is not None:
                embeddings = embeddings + torch.nn.functional.embedding(position_ids, self.position_embeddings_weight)
            if mask_inputs:
                embeddings = embeddings * token_mask.unsqueeze(2)

            if input_ is not None:
                # TODO: Accumulate redundant with masking?
                if self._sequence_parallel:
                    #  TODO:: Filter and shift embedding map instead? (needs cuda sync)
                    input_ = gather(input_, group=group, dim=0)
                    embeddings_ = embeddings.new_zeros(embeddings.shape[0] * group.size(), *embeddings.shape[1:])
                    embeddings_ = embeddings_.index_put(
                        embedding_map, input_[: embedding_map[0].size(0)], accumulate=True
                    )
                    embeddings = embeddings + split(embeddings_, group=group, dim=0)
                else:
                    embeddings = embeddings.index_put(
                        embedding_map, input_[: embedding_map[0].size(0)], accumulate=True
                    )

        with set_generator(
            self._distributed.tp_generator if self._sequence_parallel else self._distributed.pp_generator
        ):
            embeddings = torch.dropout(embeddings, self._config.dropout, self.training)
        return embeddings.to(dtype=self._residual_dtype)

    def forward(
        self,
        input_: torch.Tensor | TensorMeta,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict | None = None,
    ) -> torch.Tensor:
        if isinstance(input_, TensorMeta):
            return TensorMeta.from_dims(
                kwargs[LanguageModelKwargs.hidden_dims],
                tensor_name=f"{self.module_name} output",
                dtype=self._residual_dtype,
            )
        if (embedding_map := kwargs.get(LanguageModelKwargs.embedding_map)) is None:
            # Language model: input_ contains token ids.
            token_ids = input_
            input_ = None
        else:
            # Multimodal case: input_ contains encoder output, token ids stores in kwargs.
            # TODO: Support multiple encoders.
            # TODO: Support pipeline-parallel.
            token_ids = kwargs.get(LanguageModelKwargs.token_ids)
            # Drop the placeholder batch dimension, remove patch padding.
            input_ = input_.squeeze(int(kwargs[LanguageModelKwargs.sequence_first]))

        out = self._forward(
            input_,
            token_ids,
            kwargs.get(LanguageModelKwargs.position_ids),
            # TODO ====== Vision ====== Review input masking.
            kwargs.get(LanguageModelKwargs.mask_inputs),
            embedding_map,
        )
        self._debug(out, None, kwargs.get(LanguageModelKwargs.hidden_dims), kwargs)
        return out

    def get_compute_usage(self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig) -> int:
        # TODO: Add marginal compute? (embeddings)
        return 0

    def preprocess(self, kwargs: dict[str, typing.Any]) -> None:
        if not self._config.position_embeddings.enabled:
            return
        self._create_position_embeddings(kwargs[LanguageModelKwargs.sequence_length], self._distributed.device)
        sequence_k = kwargs[LanguageModelKwargs.sequence_k_dim].size
        sequence_q = kwargs[LanguageModelKwargs.sequence_q_dim].size
        if not self._config.cross_document_position_embeddings:
            position_ids = torch.stack(
                [
                    torch.cat([torch.arange(x) for x in sample_lens])
                    for sample_lens in kwargs[LanguageModelKwargs.sequence_lengths]
                ]
            ).to(self._distributed.device, dtype=torch.int64)
            position_ids = position_ids[:, sequence_k - sequence_q : sequence_k]
            if kwargs[LanguageModelKwargs.sequence_first]:
                position_ids = position_ids.transpose(0, 1)
            kwargs[LanguageModelKwargs.position_ids] = position_ids
        else:
            kwargs[LanguageModelKwargs.position_ids] = self._position_ids[
                sequence_k - sequence_q : sequence_k
            ].unsqueeze(int(kwargs[LanguageModelKwargs.sequence_first]))

    def _create_position_embeddings(self, sequence_length: int, device: torch.device) -> None:
        if sequence_length <= self._tensor_cache_max_sequence_length:
            return
        self._tensor_cache_max_sequence_length = sequence_length

        Assert.leq(sequence_length, self._config.num_position_embeddings)
        self._position_ids = torch.arange(0, sequence_length, device=device, dtype=torch.int64)
