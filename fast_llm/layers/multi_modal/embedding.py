import typing

import torch

from fast_llm.core.distributed import set_generator
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.layers.language_model.config import LanguageModelBaseConfig, LanguageModelKwargs
from fast_llm.layers.language_model.embedding import LanguageModelEmbedding
from fast_llm.layers.transformer.config import TransformerKwargs
from fast_llm.layers.vision_encoder.config import VisionModelKwargs
from fast_llm.layers.vision_encoder.encoder import VisionEncoder
from fast_llm.tensor import TensorMeta


class MultiModalEmbedding(LanguageModelEmbedding):
    """
    Multi-modal embedding layer to combine embeddings from text, image and more modalities.
    """

    def __init__(
        self,
        config: LanguageModelBaseConfig,
        tensor_space: TensorSpace,
    ):
        super().__init__(config, tensor_space)
        self.vision_encoder = VisionEncoder(config, tensor_space)

    def _forward(
        self,
        input_: torch.Tensor,
        position_ids: torch.Tensor | None,
        images: torch.Tensor | None,
        image_sizes: torch.Tensor | None,
        image_positions: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        image_embeddings = self.vision_encoder(images, kwargs={VisionModelKwargs.image_sizes: image_sizes})
        # TODO Soham: offset position ids
        img_tokens_seen = 0
        image_idx = 0
        text_embeddings = super()._forward(input_, position_ids)
        embeddings = []
        for sample_idx, positions in enumerate(image_positions):
            embedding_parts = []
            for position in positions:
                image = images[image_idx]
                image_tokens = (image.size[1] // self._config.vision_encoder.encoder.patch_size) * (
                    image.size[2] // self._config.vision_encoder.encoder.patch_size
                )
                image_idx += 1
                img_tokens_seen += image_tokens
                embedding_parts.append(text_embeddings[sample_idx, :position])
                embedding_parts.append(image_embeddings[img_tokens_seen : img_tokens_seen + image_tokens])
            embedding_parts.append(text_embeddings[sample_idx, position + image_tokens :])
            embeddings.append(torch.cat(embedding_parts, dim=0))
        embeddings = torch.stack(embeddings, dim=0)
        with set_generator(
            self._tensor_space.distributed.tp_generator
            if self._sequence_parallel
            else self._tensor_space.distributed.pp_generator
        ):
            embeddings = torch.dropout(embeddings, self._dropout_p, self.training)
        return embeddings.to(self._residual_dtype)

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict | None = None,
    ) -> torch.Tensor:
        if isinstance(input_, TensorMeta):
            return TensorMeta.from_dims(
                kwargs[TransformerKwargs.hidden_dims],
                tensor_name="Embedding output",
                dtype=self._residual_dtype,
            )
        return self._forward(
            input_,
            kwargs.get(LanguageModelKwargs.position_ids),
            kwargs.get(VisionModelKwargs.images),
            kwargs.get(VisionModelKwargs.image_sizes),
            kwargs.get(VisionModelKwargs.image_positions),
        )
