import functools
import typing

import torch
from transformers import PixtralVisionConfig, PixtralVisionModel

from fast_llm.engine.base_model.base_model import Layer
from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace
from fast_llm.layers.language_model.config import LanguageModelBaseConfig
from fast_llm.layers.transformer.config import TransformerKwargs
from fast_llm.layers.vision_encoder.adapter import VisionAdapter
from fast_llm.layers.vision_encoder.config import VisionEncoderDimNames
from fast_llm.tensor import ParameterMeta, TensorMeta, init_normal_


class VisionEncoder(Layer):
    """
    A vision encoder layer for creating token embeddings from vision model
    """

    def __init__(self, config: LanguageModelBaseConfig, tensor_space: TensorSpace):
        super().__init__()

        self._config = config.vision_encoder
        self._distributed_config = tensor_space.distributed_config
        with torch.device("meta"):
            if self._config.encoder.path:
                self._vision_encoder = PixtralVisionModel.from_pretrained(
                    self._config.encoder.path, torch_dtype=self._distributed_config.training_dtype.torch
                )
            else:
                self._vision_encoder = PixtralVisionModel(
                    PixtralVisionConfig(
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
                )
        param_names = []
        # gather all names first. PyTorch complains if we do it in the loop
        for name, param in self._vision_encoder.named_parameters():
            param_names.append(name)
        for name in param_names:
            *module_path, stem = name.split(".")
            module = functools.reduce(getattr, module_path, self._vision_encoder)
            param = self._vision_encoder.get_parameter(name)
            setattr(
                module,
                stem,
                ParameterMeta.from_dims(
                    tuple(TensorDim(f"{name}_{idx}", size) for idx, size in enumerate(param.shape)),
                    init_method=init_normal_(),
                ),
            )
            none_params = [key for key, value in module._parameters.items() if value is None]
            for key in none_params:
                module._parameters.pop(key)
        self._adapter = VisionAdapter(
            intermediate_size=tensor_space.get_tensor_dim(VisionEncoderDimNames.intermediate_size),
            tensor_space=tensor_space,
        )

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
        return self._vision_encoder(input_)
