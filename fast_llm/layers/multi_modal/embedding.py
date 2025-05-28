import typing

import torch

from fast_llm.core.distributed import set_generator
from fast_llm.core.ops import gather, reduce_forward, split
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.layers.audio_encoder.config import AudioEncoderKwargs
from fast_llm.layers.language_model.config import LanguageModelBaseConfig, LanguageModelKwargs
from fast_llm.layers.language_model.embedding import LanguageModelEmbedding
from fast_llm.layers.transformer.config import TransformerKwargs
from fast_llm.layers.vision_encoder.config import VisionEncoderKwargs
from fast_llm.layers.vision_encoder.preprocessing import get_num_image_tokens
from fast_llm.tensor import TensorMeta
from fast_llm.utils import Assert, div


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

    @torch.compile
    def _forward(
        self,
        input_: torch.Tensor,
        tokens: torch.Tensor,
        position_ids: torch.Tensor | None,
        image_positions: list[torch.Tensor] | None,
        image_sizes: list[list[tuple[int, int]]] | None,
        audio_positions: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        """
        Forward pass for the multi-modal embedding layer.
        Args:
            input_: The input tensor (image embeddings).
            tokens: The tokenized text input.
            position_ids: The position ids for the text input.
            image_positions: The positions of the image tokens in the input.
            image_sizes: The sizes of the images in the input.
        Returns:
            The combined embeddings for text and images.
        """
        Assert.eq(position_ids is not None, self._use_absolute_position_embeddings)
        group = self._tensor_space.distributed.tensor_group
        if self._parallel_embeddings:
            token_mask = (tokens >= self._vocab_start_index) * (tokens < self._vocab_end_index)
            masked_tokens = (tokens - self._vocab_start_index) * token_mask
            embeddings = torch.embedding(self.word_embeddings_weight, masked_tokens) * token_mask.unsqueeze(2)  # noqa
            embeddings = reduce_forward(embeddings, group)
            if self._use_absolute_position_embeddings:
                embeddings = embeddings + torch.nn.functional.embedding(position_ids, self.position_embeddings_weight)
            embeddings = embeddings.clone()
            input_ = gather(input_, group, dim=0)
            # TODO: Toby implement audio
            for sample_idx, (positions, sizes) in enumerate(zip(image_positions, image_sizes)):
                image_embedding_offset = 0
                for position, size in zip(positions, sizes):
                    num_image_tokens = get_num_image_tokens(
                        *size,
                        self._config.vision_encoder.patch_size,
                        image_break=self._config.vision_encoder.image_break_token is not None,
                    )
                    # Calculate the patch dimensions
                    patch_width = div(size[0], self._config.vision_encoder.patch_size)
                    patch_height = div(size[1], self._config.vision_encoder.patch_size)

                    # Process row by row for both sequence parallel and non-parallel cases
                    for row in range(patch_height):
                        # Calculate source and destination starting positions
                        row_start_src = image_embedding_offset + row * patch_width
                        row_start_dst = position + row * (patch_width + 1)

                        # Always use full patch_width
                        tokens_in_row = patch_width

                        if self._sequence_parallel:
                            # Copy with dimensions swapped for sequence parallel case
                            embeddings[row_start_dst : row_start_dst + tokens_in_row, sample_idx] = input_[
                                row_start_src : row_start_src + tokens_in_row, sample_idx
                            ]
                        else:
                            # Copy with normal dimension ordering
                            embeddings[sample_idx, row_start_dst : row_start_dst + tokens_in_row] = input_[
                                sample_idx, row_start_src : row_start_src + tokens_in_row
                            ]

                    # Move to the next image in the input tensor
                    image_embedding_offset += num_image_tokens
            if self._sequence_parallel:
                embeddings = split(embeddings, group=group, dim=0)
        else:
            if self._sequence_parallel:
                tokens = split(tokens, group=group, dim=0)
                if self._use_absolute_position_embeddings:
                    position_ids = split(position_ids, group=group, dim=0)
                # TODO Soham: get image positions for current split. Maybe in preprocessing?
                # for positions in image_positions:
                #     if positions > self._distributed_config.tensor_rank
            # mask padded tokens
            token_mask = tokens >= 0
            masked_tokens = tokens * token_mask
            embeddings = torch.embedding(self.word_embeddings_weight, masked_tokens) * token_mask.unsqueeze(2)
            embeddings = embeddings.clone()
            for sample_idx, (positions, sizes) in enumerate(zip(image_positions, image_sizes)):
                image_embedding_offset = 0
                for position, size in zip(positions, sizes):
                    num_image_tokens = get_num_image_tokens(
                        *size,
                        self._config.vision_encoder.patch_size,
                        image_break=self._config.vision_encoder.image_break_token is not None,
                    )
                    # Calculate the patch dimensions
                    patch_width = div(size[0], self._config.vision_encoder.patch_size)
                    patch_height = div(size[1], self._config.vision_encoder.patch_size)

                    # Process row by row
                    for row in range(patch_height):
                        # Calculate source and destination starting positions
                        row_start_src = image_embedding_offset + row * patch_width
                        row_start_dst = position + row * (patch_width + 1)

                        # Copy row by row
                        embeddings[sample_idx, row_start_dst : row_start_dst + patch_width] = input_[
                            sample_idx, row_start_src : row_start_src + patch_width
                        ]

                    # Move to the next image in the input tensor
                    image_embedding_offset += num_image_tokens

            audio_position_idx = 0
            for sample_idx, positions in enumerate(audio_positions):
                for position in positions:
                    num_audio_tokens = input_.shape[1]  # TODO: Toby better way to get this?
                    embeddings[sample_idx, position : position + num_audio_tokens] = input_[audio_position_idx]
                    audio_position_idx += 1

            if self._use_absolute_position_embeddings:
                embeddings = embeddings + torch.nn.functional.embedding(position_ids, self.position_embeddings_weight)
        with set_generator(
            self._tensor_space.distributed.tp_generator
            if self._sequence_parallel
            else self._tensor_space.distributed.pp_generator
        ):
            embeddings = torch.dropout(embeddings, self._dropout_p, self.training)
        return embeddings.to(dtype=self._residual_dtype)

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
        # TODO: How do we support both Audio and Vision?
        position_ids = kwargs.get(LanguageModelKwargs.position_ids)
        image_sizes = kwargs.get(VisionEncoderKwargs.image_sizes, [])
        image_positions = kwargs.get(VisionEncoderKwargs.image_positions, [])
        audio_positions = kwargs.get(AudioEncoderKwargs.audio_positions, [])
        tokens = kwargs.get(LanguageModelKwargs.tokens)

        return self._forward(input_, tokens, position_ids, image_positions, image_sizes, audio_positions)
