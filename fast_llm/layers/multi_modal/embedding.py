import typing

import torch

from fast_llm.core.distributed import set_generator
from fast_llm.core.ops import reduce_forward, split
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.layers.language_model.config import LanguageModelBaseConfig, LanguageModelKwargs
from fast_llm.layers.language_model.embedding import LanguageModelEmbedding
from fast_llm.layers.transformer.config import TransformerKwargs
from fast_llm.layers.vision_encoder.config import VisionEncoderKwargs
from fast_llm.layers.vision_encoder.preprocessing import get_num_patches
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

    # @torch.compile
    def _forward(
        self,
        input_: torch.Tensor,
        tokens: torch.Tensor,
        position_ids: torch.Tensor | None,
        image_positions: list[torch.Tensor] | None,
        image_sizes: list[list[tuple[int, int]]] | None,
    ) -> torch.Tensor:
        """
        Forward pass for the multi-modal embedding layer.
        Args:
            input_: The input tensor (image embeddings after patch merging).
                    Shape: (max_patches, batch, hidden) - patches first, matches original design
            tokens: The tokenized text input.
            position_ids: The position ids for the text input.
            image_positions: The positions of the image tokens in the input.
            image_sizes: The sizes of the images in the input (in pixels, after resize).
        Returns:
            The combined embeddings for text and images.
        """
        Assert.eq(position_ids is not None, self._use_absolute_position_embeddings)
        group = self._tensor_space.distributed.tensor_group

        # Compute effective patch size accounting for patch merging
        # When spatial_merge_size > 1, patches are merged, reducing token count
        effective_patch_size = (
            self._config.vision_encoder.patch_size * self._config.vision_encoder.spatial_merge_size
        )

        if self._sequence_parallel:
            micro_seqlen = input_.size(0)
            patch_start_offset = self._distributed_config.tensor_rank * micro_seqlen
            patch_end_offset = (self._distributed_config.tensor_rank + 1) * micro_seqlen
        else:
            patch_start_offset = 0
            patch_end_offset = input_.size(0)  # patches dimension is dim 0 (patches, batch, hidden)

        if self._parallel_embeddings:
            token_mask = (tokens >= self._vocab_start_index) * (tokens < self._vocab_end_index)
            masked_tokens = (tokens - self._vocab_start_index) * token_mask
            embeddings = torch.embedding(self.word_embeddings_weight, masked_tokens) * token_mask.unsqueeze(2)  # noqa
            # Cloning since we will modify the embeddings in-place
            embeddings = embeddings.clone()
            # the embeddings tensor are full-sized, but we might get a split of the patch embeddings
            # We need to determine the offset in the embeddings tensor for each sample
            # and also account for the special image tokens if applicable
            for sample_idx, (positions, sizes) in enumerate(zip(image_positions, image_sizes)):
                image_embedding_offset = 0
                for position, size in zip(positions, sizes):
                    # Use effective patch size for merged patch count
                    num_patches = get_num_patches(*size, effective_patch_size)
                    if image_embedding_offset + num_patches < patch_start_offset:
                        image_embedding_offset += num_patches
                        continue
                    if self._config.vision_encoder.image_break_token is not None:
                        patch_height = div(size[0], effective_patch_size)
                        patch_width = div(size[1], effective_patch_size)
                        for row in range(patch_height):
                            row_start_src = image_embedding_offset + row * patch_width
                            row_start_dst = position + row * (patch_width + 1)
                            if row_start_src > patch_end_offset:
                                break
                            if row_start_src + patch_width <= patch_start_offset:
                                continue

                            input_start_index = max(row_start_src, patch_start_offset) - patch_start_offset
                            input_end_index = min(row_start_src + patch_width, patch_end_offset) - patch_start_offset
                            embeddings_start_index = row_start_dst + max(patch_start_offset - row_start_src, 0)
                            embeddings_end_index = (
                                row_start_dst + patch_width - max(row_start_src + patch_width - patch_end_offset, 0)
                            )
                            # input_ shape: (patches, batch, hidden) - always patches first
                            if self._sequence_parallel:
                                embeddings[embeddings_start_index:embeddings_end_index, sample_idx] = input_[
                                    input_start_index:input_end_index, sample_idx
                                ]
                            else:
                                # input_ shape: (patches, batch, hidden) - always patches first from VisionAdapter
                                embeddings[sample_idx, embeddings_start_index:embeddings_end_index] = input_[
                                    input_start_index:input_end_index, sample_idx
                                ]
                    else:
                        input_start_index = max(image_embedding_offset, patch_start_offset) - patch_start_offset
                        input_end_index = (
                            min(image_embedding_offset + num_patches, patch_end_offset) - patch_start_offset
                        )
                        embedding_start_index = position - max(patch_start_offset - image_embedding_offset, 0)
                        embedding_end_index = (
                            position + num_patches - max(image_embedding_offset + num_patches - patch_end_offset, 0)
                        )
                        embeddings[sample_idx, embedding_start_index:embedding_end_index] = input_[
                            input_start_index:input_end_index, sample_idx
                        ]
                    image_embedding_offset += num_patches
                    if image_embedding_offset > patch_end_offset:
                        break
            embeddings = reduce_forward(embeddings, group)
            if self._use_absolute_position_embeddings:
                embeddings = embeddings + torch.nn.functional.embedding(position_ids, self.position_embeddings_weight)
            if self._sequence_parallel:
                embeddings = split(embeddings, group=group, dim=0)
        else:
            if self._sequence_parallel:
                tokens = split(tokens, group=group, dim=0)
                if self._use_absolute_position_embeddings:
                    position_ids = split(position_ids, group=group, dim=0)
            # mask padded tokens
            token_mask = tokens >= 0
            masked_tokens = tokens * token_mask
            embeddings = torch.embedding(self.word_embeddings_weight, masked_tokens) * token_mask.unsqueeze(2)
            embeddings = embeddings.clone()
            for sample_idx, (positions, sizes) in enumerate(zip(image_positions, image_sizes)):
                image_embedding_offset = 0
                for position, size in zip(positions, sizes):
                    # Use effective patch size for merged patch count
                    num_patches = get_num_patches(*size, effective_patch_size)
                    if self._config.vision_encoder.image_break_token is not None:
                        patch_height = div(size[0], effective_patch_size)
                        patch_width = div(size[1], effective_patch_size)

                        for row in range(patch_height):
                            row_start_src = image_embedding_offset + row * patch_width
                            row_start_dst = position + row * (patch_width + 1)

                            # input_ shape: (patches, batch, hidden)
                            embeddings[sample_idx, row_start_dst : row_start_dst + patch_width] = input_[
                                row_start_src : row_start_src + patch_width, sample_idx
                            ]
                    else:
                        # input_ shape: (patches, batch, hidden)
                        embeddings[sample_idx, position : position + num_patches] = input_[
                            image_embedding_offset : image_embedding_offset + num_patches, sample_idx
                        ]
                    # Move to the next image in the input tensor
                    image_embedding_offset += num_patches

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
        position_ids = kwargs.get(LanguageModelKwargs.position_ids)
        image_sizes = kwargs.get(VisionEncoderKwargs.image_sizes)
        image_positions = kwargs.get(VisionEncoderKwargs.image_positions)
        tokens = kwargs.get(LanguageModelKwargs.tokens)

        return self._forward(input_, tokens, position_ids, image_positions, image_sizes)
