import torch

from fast_llm.core.distributed import set_generator
from fast_llm.core.ops import reduce_forward, split
from fast_llm.engine.base_model.base_model import Layer
from fast_llm.layers.language_model.config import LanguageModelBaseConfig, LanguageModelDimNames, LanguageModelKwargs
from fast_llm.layers.transformer.config import TransformerDimNames, TransformerKwargs
from fast_llm.tensor import ParameterMeta, TensorMeta, TensorSpace, init_normal_
from fast_llm.utils import Assert

WORD_EMBEDDINGS_WEIGHT = "word_embeddings_weight"


class LanguageModelEmbedding(Layer):
    """
    A language model embedding layer.
    Consists of word embeddings (tensor-parallel or sequence-tensor-parallel),
    together with optional absolute position embeddings and dropout.
    """

    # Ensure the layer is on its own stage.
    layer_count: float = 1000.0

    def __init__(
        self,
        config: LanguageModelBaseConfig,
        tensor_space: TensorSpace,
    ):
        super().__init__()
        config.validate()
        self._distributed_config = tensor_space.distributed_config
        self._tensor_space = tensor_space
        self._residual_dtype = (
            self._distributed_config.optimization_dtype
            if config.transformer.full_precision_residual
            else self._distributed_config.training_dtype
        )
        self._group_size = self._distributed_config.tensor_parallel
        self._sequence_parallel = self._distributed_config.sequence_tensor_parallel
        self._parallel_embeddings = tensor_space.distributed_config.tensor_parallel > 1 and config.parallel_embeddings
        self._dropout_p = config.transformer.hidden_dropout
        self._use_absolute_position_embeddings = config.use_absolute_position_embeddings

        hidden_dim = tensor_space.get_tensor_dim(TransformerDimNames.hidden)
        vocab_dim = tensor_space.get_tensor_dim(
            LanguageModelDimNames.vocab_tp if self._parallel_embeddings else LanguageModelDimNames.vocab
        )

        if self._parallel_embeddings:
            self._vocab_start_index = self._distributed_config.tensor_rank * vocab_dim.size
            self._vocab_end_index = (self._distributed_config.tensor_rank + 1) * vocab_dim.size

        self.word_embeddings_weight = ParameterMeta.from_dims(
            (vocab_dim, hidden_dim),
            init_method=init_normal_(std=config.init_method_std_embed),
        )
        if self._use_absolute_position_embeddings:
            self.position_embeddings_weight = ParameterMeta.from_dims(
                (tensor_space.get_tensor_dim(LanguageModelDimNames.position_embed), hidden_dim),
                init_method=init_normal_(std=config.init_method_std_embed),
                allow_sequence_tensor_parallel=not config.parallel_embeddings,
            )

    @torch.compile
    def _forward(self, input_: torch.Tensor, position_ids: torch.Tensor | None):
        Assert.eq(position_ids is not None, self._use_absolute_position_embeddings)
        group = self._tensor_space.distributed.tensor_group
        if self._parallel_embeddings:
            input_mask = (input_ >= self._vocab_start_index) * (input_ < self._vocab_end_index)
            masked_input = (input_ - self._vocab_start_index) * input_mask
            embeddings = torch.embedding(self.word_embeddings_weight, masked_input) * input_mask.unsqueeze(2)  # noqa
            embeddings = reduce_forward(embeddings, group)
            if self._use_absolute_position_embeddings:
                embeddings = embeddings + torch.nn.functional.embedding(position_ids, self.position_embeddings_weight)
            if self._sequence_parallel:
                embeddings = split(embeddings, group=group, dim=0)
        else:
            if self._sequence_parallel:
                input_ = split(input_, group=group, dim=0)
                if self._use_absolute_position_embeddings:
                    position_ids = split(position_ids, group=group, dim=0)
            embeddings = torch.embedding(self.word_embeddings_weight, input_)
            if self._use_absolute_position_embeddings:
                embeddings = embeddings + torch.nn.functional.embedding(position_ids, self.position_embeddings_weight)
        with set_generator(
            self._tensor_space.distributed.tp_generator
            if self._sequence_parallel
            else self._tensor_space.distributed.pp_generator
        ):
            embeddings = torch.dropout(embeddings, self._dropout_p, self.training)
        return embeddings.to(dtype=self._residual_dtype)

    def forward(self, input_, kwargs, losses: dict | None = None, metrics: dict | None = None):
        if isinstance(input_, TensorMeta):
            return TensorMeta.from_dims(
                kwargs[TransformerKwargs.hidden_dims],
                tensor_name="Embedding output",
                dtype=self._residual_dtype,
            )
        return self._forward(input_, kwargs.get(LanguageModelKwargs.position_ids))
