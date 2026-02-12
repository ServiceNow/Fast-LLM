import typing

import torch

from fast_llm.core.ops import gather, split
from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.functional.triton.mlp import torch_mlp_activation
from fast_llm.layers.common.linear import Linear
from fast_llm.layers.common.normalization import RMSNorm
from fast_llm.layers.transformer.config import TransformerDimNames, TransformerKwargs
from fast_llm.layers.vision_encoder.config import VisionEncoderConfig, VisionEncoderDimNames, VisionEncoderKwargs
from fast_llm.tensor import TensorMeta, init_normal_, init_zeros_


class PatchMerger(torch.nn.Module):
    """
    Learned merging of spatial_merge_size ** 2 patches.
    Reduces the number of vision tokens by merging spatial patches.
    """

    def __init__(
        self,
        hidden_size: int,
        spatial_merge_size: int,
        patch_size: int,
        tensor_space: TensorSpace,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size

        merging_layer_in_channels = tensor_space.get_tensor_dim(VisionEncoderDimNames.merging_layer_in_channels)
        vision_hidden_dim = tensor_space.get_tensor_dim(VisionEncoderDimNames.out_channels)
        self.merging_layer = Linear(
            merging_layer_in_channels,
            vision_hidden_dim,
            bias=False,
            weight_init_method=init_normal_(),
            bias_init_method=None,
            allow_no_grad=True,  # May be skipped when batch has no images
        )

    def forward(
        self,
        image_features: torch.Tensor,
        image_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_features: Flattened patch features, shape (total_patches, hidden_size)
            image_sizes: Original image sizes before patching, shape (num_images, 2)
                         Each row is (height, width) in pixels.

        Returns:
            Merged features with reduced spatial resolution
        """
        # Convert image sizes (in pixels) to patch grid sizes
        patch_grid_sizes = [
            (image_sizes[i][0] // self.patch_size, image_sizes[i][1] // self.patch_size)
            for i in range(len(image_sizes))
        ]
        tokens_per_image = [h * w for h, w in patch_grid_sizes]
        d = self.hidden_size

        permuted_tensors = []
        for image_index, image_tokens in enumerate(image_features.split(tokens_per_image)):
            h, w = patch_grid_sizes[image_index]
            # Reshape to 2D grid: (h * w, d) -> (1, d, h, w)
            image_grid = image_tokens.view(h, w, d).permute(2, 0, 1).unsqueeze(0)
            # Unfold extracts spatial_merge_size x spatial_merge_size blocks
            # Output shape: (1, d * merge^2, num_merged_patches)
            grid = torch.nn.functional.unfold(
                image_grid,
                kernel_size=self.spatial_merge_size,
                stride=self.spatial_merge_size,
            )
            # Reshape: (d * merge^2, num_merged_patches) -> (num_merged_patches, d * merge^2)
            grid = grid.view(d * self.spatial_merge_size**2, -1).t()
            permuted_tensors.append(grid)

        image_features = torch.cat(permuted_tensors, dim=0)
        image_features = self.merging_layer(image_features)
        return image_features


class VisionAdapter(Layer):
    """
    Vision adapter (projector) layer for the LLM.

    Architecture: RMSNorm -> PatchMerger -> Linear -> Activation -> Linear
    """

    def __init__(self, config: VisionEncoderConfig, tensor_space: TensorSpace):
        super().__init__()
        self._activation_type = config.adapter_activation_type
        self._tensor_space = tensor_space
        self._sequence_parallel = tensor_space.distributed_config.sequence_tensor_parallel

        vision_hidden_dim = tensor_space.get_tensor_dim(VisionEncoderDimNames.out_channels)
        vision_hidden_size = vision_hidden_dim.size
        adapter_intermediate_dim = tensor_space.get_tensor_dim(VisionEncoderDimNames.adapter_size)
        llm_hidden_dim = tensor_space.get_tensor_dim(TransformerDimNames.hidden)

        self.norm = RMSNorm(vision_hidden_dim, eps=config.adapter_norm_eps)

        self.patch_merger = PatchMerger(
            hidden_size=vision_hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            patch_size=config.patch_size,
            tensor_space=tensor_space,
        )

        self.layer_1 = Linear(
            vision_hidden_dim,
            llm_hidden_dim,
            bias=config.adapter_bias,
            weight_init_method=init_normal_(),
            bias_init_method=init_zeros_() if config.adapter_bias else None,
        )
        self.layer_2 = Linear(
            llm_hidden_dim,
            llm_hidden_dim,
            bias=config.adapter_bias,
            weight_init_method=init_normal_(),
            bias_init_method=init_zeros_() if config.adapter_bias else None,
        )

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict[str, typing.Any] | None = None,
    ) -> torch.Tensor:
        if isinstance(input_, TensorMeta):
            return TensorMeta.from_dims(
                kwargs[TransformerKwargs.hidden_dims],
                tensor_name="Vision adapter output",
                dtype=input_.dtype,
            )

        image_sizes = kwargs.get(VisionEncoderKwargs.image_sizes)

        # image_sizes is a list of lists: [[img1_size, img2_size, ...], [...], ...]
        # Each inner list contains image sizes for one sample in the batch
        has_images = (
            image_sizes is not None
            and len(image_sizes) > 0
            and any(len(sizes) > 0 for sizes in image_sizes)
        )

        # Always apply norm to ensure gradients flow through it
        hidden_states = self.norm(input_)

        if has_images:
            group = self._tensor_space.distributed.tensor_group if self._sequence_parallel else None
            tensor_parallel = self._tensor_space.distributed_config.tensor_parallel
            batch_size = len(image_sizes)
            hidden_size = hidden_states.size(-1)
            
            # PatchMerger needs full spatial structure for unfold operation.
            # When sequence parallel is enabled, patches are split across workers.
            # Gather all patches before PatchMerger, then split after.
            if self._sequence_parallel:
                hidden_states = gather(hidden_states, group, dim=0)
            
            # Process each sample separately and collect merged patches
            merged_samples = []
            patch_offset = 0
            max_merged_patches = 0
            
            for sample_image_sizes in image_sizes:
                if len(sample_image_sizes) == 0:
                    # No images for this sample - will be padded later
                    merged_samples.append(None)
                    continue
                
                # Count pre-merge patches for this sample
                num_patches_in_sample = sum(
                    (h // self.patch_merger.patch_size) * (w // self.patch_merger.patch_size)
                    for h, w in sample_image_sizes
                )
                
                # Extract and merge this sample's patches
                sample_patches = hidden_states[patch_offset : patch_offset + num_patches_in_sample]
                merged = self.patch_merger(sample_patches, sample_image_sizes)
                merged_samples.append(merged)
                
                max_merged_patches = max(max_merged_patches, merged.size(0))
                patch_offset += num_patches_in_sample
            
            # Pad and stack into batched format: (batch, max_patches, hidden)
            # For sequence parallel, pad to be divisible by tensor_parallel
            if self._sequence_parallel and max_merged_patches % tensor_parallel != 0:
                max_merged_patches = ((max_merged_patches + tensor_parallel - 1) // tensor_parallel) * tensor_parallel
            
            batched_output = []
            for merged in merged_samples:
                if merged is None:
                    # Sample with no images - all padding
                    padded = torch.zeros(
                        max_merged_patches, hidden_size,
                        dtype=hidden_states.dtype,
                        device=hidden_states.device
                    )
                else:
                    # Pad to max_merged_patches
                    pad_size = max_merged_patches - merged.size(0)
                    if pad_size > 0:
                        padding = torch.zeros(
                            pad_size, hidden_size,
                            dtype=merged.dtype,
                            device=merged.device
                        )
                        padded = torch.cat([merged, padding], dim=0)
                    else:
                        padded = merged
                batched_output.append(padded)
            
            # Stack to (batch, max_patches, hidden) then transpose to (patches, batch, hidden)
            # This matches the original embedding code's expected format
            hidden_states = torch.stack(batched_output, dim=0)  # (batch, patches, hidden)
            hidden_states = hidden_states.transpose(0, 1)  # (patches, batch, hidden)
            
            # For sequence parallel: split on patches dimension (dim=0)
            if self._sequence_parallel:
                hidden_states = split(hidden_states, group, dim=0)  # (micro_patches, batch, hidden)
        # When no images, patch_merger is skipped - its weights have allow_no_grad=True

        hidden_states = self.layer_1(hidden_states)
        hidden_states = torch_mlp_activation(
            input_=hidden_states,
            gated=False,
            activation_type=self._activation_type,
        )
        hidden_states = self.layer_2(hidden_states)

        return hidden_states
    