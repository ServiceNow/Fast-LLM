import functools
import typing

import torch
from transformers import PixtralVisionConfig
from transformers.models.pixtral.modeling_pixtral import PixtralTransformer

from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.layers.language_model.config import LanguageModelBaseConfig
from fast_llm.layers.transformer.config import TransformerKwargs
from fast_llm.layers.vision_encoder.adapter import VisionAdapter
from fast_llm.layers.vision_encoder.config import VisionEncoderDimNames, VisionModelKwargs
from fast_llm.tensor import ParameterMeta, TensorMeta, init_normal_


def position_ids_in_meshgrid(patch_embeddings_list, max_width):
    positions = []
    for patch in patch_embeddings_list:
        height, width = patch.shape[-2:]
        mesh = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        h_grid, v_grid = torch.stack(mesh, dim=-1).reshape(-1, 2).chunk(2, -1)
        ids = h_grid * max_width + v_grid
        positions.append(ids[:, 0])
    return torch.cat(positions)


def generate_block_attention_mask(patch_embeds_list, tensor):
    dtype = tensor.dtype
    device = tensor.device
    seq_len = tensor.shape[1]
    d_min = torch.finfo(dtype).min
    causal_mask = torch.full((seq_len, seq_len), fill_value=d_min, dtype=dtype, device=device)

    block_end_idx = torch.tensor(patch_embeds_list).cumsum(-1)
    block_start_idx = torch.tensor([0] + patch_embeds_list[:-1]).cumsum(-1)
    for start, end in zip(block_start_idx, block_end_idx):
        causal_mask[start:end, start:end] = 0

    causal_mask = causal_mask[None, None, :, :].expand(tensor.shape[0], 1, -1, -1)
    return causal_mask


# TODO Soham: should this just be nn.Module?
class VisionEncoder(Layer):
    """
    A vision encoder layer for creating token embeddings from vision model
    """

    def __init__(self, config: LanguageModelBaseConfig, tensor_space: TensorSpace):
        super().__init__()

        self._config = config.vision_encoder
        self._distributed_config = tensor_space.distributed_config
        with torch.device("meta"):
            # TODO Soham options to fix rotary:
            # 1. load PixtralTransformer instead of PixtralVisionModel. Required to implement conv2d, ln_pre separately and store positional embeddings in kwargs_meta
            # 2. set self.vision_encoder.position_embeddings = PixtralRotaryEmbedding(config) outside of meta scope
            config = PixtralVisionConfig(
                hidden_size=self._config.encoder.hidden_size,
                intermediate_size=self._config.encoder.intermediate_size,
                num_hidden_layers=self._config.encoder.num_hidden_layers,
                num_attention_heads=self._config.encoder.num_attention_heads,
                num_channels=self._config.encoder.num_channels,
                image_size=self._config.encoder.image_size,
                patch_size=self._config.encoder.patch_size,
                hidden_act=self._config.encoder.hidden_act,
                attention_dropout=self._config.encoder.attention_dropout,
                rope_theta=self._config.encoder.rope_theta,
                initializer_range=self._config.encoder.initializer_range,
            )
            self.patch_conv = torch.nn.Conv2d(
                in_channels=3,
                out_channels=self._config.encoder.hidden_size,
                kernel_size=self._config.encoder.patch_size,
                stride=self._config.encoder.patch_size,
                bias=False,
            )
            self.patch_conv.weight = ParameterMeta.from_dims(
                tuple(
                    TensorDim(f"patch_conv_weight_{idx}", size)
                    for idx, size in enumerate(self.patch_conv.weight.shape)
                ),
                init_method=init_normal_(),
            )
            self.norm = self._config.encoder.pre_norm.get_layer(
                tensor_space.get_tensor_dim(VisionEncoderDimNames.out_channels)
            )
            self.vision_transformer = PixtralTransformer(config)
            # self.vision_encoder = PixtralVisionModel(config)
        param_names = []
        # gather all names first. PyTorch complains if we do it in the loop
        for name, param in self.vision_transformer.named_parameters():
            param_names.append(name)
        for name in param_names:
            *module_path, stem = name.split(".")
            module = functools.reduce(getattr, module_path, self.vision_transformer)
            param = self.vision_transformer.get_parameter(name)
            setattr(
                module,
                stem,
                ParameterMeta.from_dims(
                    tuple(TensorDim(f"{name}_{idx}", size) for idx, size in enumerate(param.shape)),
                    init_method=init_normal_(),
                ),
            )
            # none_params = [key for key, value in module._parameters.items() if value is None]
            # for key in none_params:
            #     module._parameters.pop(key)
        self.adapter = VisionAdapter(
            intermediate_size=tensor_space.get_tensor_dim(VisionEncoderDimNames.intermediate_size),
            tensor_space=tensor_space,
        )

    def _forward(
        self, input_: torch.Tensor, image_sizes: torch.Tensor, inv_freq: torch.Tensor, image_width: int
    ) -> torch.Tensor:
        patch_embeddings = self.patch_conv(input_)
        patch_embeddings_list = [
            embedding[..., : image_size[0], : image_size[1]]
            for embedding, image_size in zip(patch_embeddings, image_sizes)
        ]
        patch_embeddings = torch.cat([p.flatten(1).T for p in patch_embeddings_list], dim=0).unsqueeze(0)
        patch_embeddings = self.norm(patch_embeddings)
        position_ids = position_ids_in_meshgrid(patch_embeddings_list, image_width // self._config.encoder.patch_size)
        freqs = inv_freq[position_ids]
        with torch.autocast(device_type=input_.device.type):
            cos = freqs.cos()
            sin = freqs.sin()
        cos = cos.to(dtype=input_.dtype)
        sin = sin.to(dtype=input_.dtype)

        attention_mask = generate_block_attention_mask(
            [p.shape[-2] * p.shape[-1] for p in patch_embeddings_list], patch_embeddings
        )

        (out,) = self.vision_transformer(
            patch_embeddings,
            attention_mask=attention_mask,
            position_embeddings=(cos, sin),
            output_attentions=False,
            return_dict=False,
        )

        return self.adapter(out)

    def forward(
        self,
        input_: torch.Tensor,
        kwargs: dict[str, typing.Any],
        losses: dict[str, typing.Any] | None = None,
        metrics: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(input_, TensorMeta):
            return TensorMeta.from_dims(
                kwargs[TransformerKwargs.hidden_dims],
                tensor_name="Vision Output",
                dtype=self._distributed_config.training_dtype.torch,
            )
        return self._forward(
            input_,
            kwargs[VisionModelKwargs.image_sizes][:1],
            kwargs[VisionModelKwargs.rotary_inv_freq],
            image_width=kwargs[VisionModelKwargs.image_size],
        )
        # return self.adapter(self.vision_encoder(input_, kwargs[VisionModelKwargs.image_sizes]))
