import typing

import torch

from fast_llm.core.distributed import set_generator
from fast_llm.core.ops import reduce_forward, split
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

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
        *,
        hidden_dim: TensorDim,
        lr_scale: float | None,
        peft: PeftConfig | None,
        return_input: bool = False,
    ):
        if return_input:
            raise NotImplementedError()
        super().__init__(
            config,
            distributed_config,
            hidden_dim=hidden_dim,
            lr_scale=lr_scale,
            peft=peft,
            return_input=return_input,
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
    def _forward(self, input_: torch.Tensor, position_ids: torch.Tensor | None, mask_inputs: bool) -> torch.Tensor:
        Assert.eq(position_ids is None, self.position_embeddings_weight is None)
        group = self._parallel_dim.group
        if self._vocab_parallel:
            input_mask = (input_ >= self._vocab_start_index) * (input_ < self._vocab_end_index)
            masked_input = (input_ - self._vocab_start_index) * input_mask
            embeddings = torch.embedding(self.word_embeddings_weight, masked_input) * input_mask.unsqueeze(2)  # noqa
            embeddings = reduce_forward(embeddings, group)
            if self.position_embeddings_weight is not None:
                embeddings = embeddings + torch.nn.functional.embedding(position_ids, self.position_embeddings_weight)
            if self._sequence_parallel:
                embeddings = split(embeddings, group=group, dim=0)
        else:
            if self._sequence_parallel:
                input_ = split(input_, group=group, dim=0)
                if self.position_embeddings_weight is not None:
                    position_ids = split(position_ids, group=group, dim=0)
            # handle masked tokens
            if mask_inputs:
                input_mask = input_ >= 0
                masked_input = input_ * input_mask
                embeddings = torch.embedding(self.word_embeddings_weight, masked_input)
            else:
                embeddings = torch.embedding(self.word_embeddings_weight, input_)
            if self.position_embeddings_weight is not None:
                embeddings = embeddings + torch.nn.functional.embedding(position_ids, self.position_embeddings_weight)
            if mask_inputs:
                embeddings = embeddings * input_mask.unsqueeze(2)
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
                tensor_name="Embedding output",
                dtype=self._residual_dtype,
            )
        return self._forward(
            input_, kwargs.get(LanguageModelKwargs.position_ids), kwargs.get(LanguageModelKwargs.mask_inputs)
        )

    def get_compute_usage(self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig) -> int:
        # TODO: Add marginal compute? (embeddings)
        return 0
