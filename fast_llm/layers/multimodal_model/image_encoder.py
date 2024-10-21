import logging
import copy
import torch

from fast_llm.layers.language_model.config import LanguageModelBaseConfig, LanguageModelDimNames, LanguageModelKwargs
from fast_llm.layers.multimodal_model.config import MultimodalModelKwargs, MultimodalModelBaseConfig
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.tensor import ParameterMeta, TensorMeta, TensorDim, init_normal_

logger = logging.getLogger(__name__)

class ImageEncoder(torch.nn.Module):
    
    # Ensure the layer is on its own stage.
    layer_count: float = 1000.0
    
    def __init__(
        self,
        config: LanguageModelBaseConfig,
        tensor_space: TensorSpace,
    ):
        super(ImageEncoder, self).__init__()
        self._distributed_config = tensor_space.distributed_config
        self._tensor_space = tensor_space
        self._residual_dtype = (
            self._distributed_config.optimization_dtype
            if config.transformer.full_precision_residual
            else self._distributed_config.training_dtype
        ).torch
        self.image_encoder_type = config.multimodal_model.image_encoder_type

        if self.image_encoder_type.lower() == "clip":
            import open_clip

            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="laion2b_s32b_b82k"
            )

            self.visual_encoder = model.visual
            self.visual_encoder.output_tokens = True
            self.ln_vision = copy.deepcopy(self.visual_encoder.ln_post)
        else:
            logger.error(f'Unknown image encoder specified: {self.image_encoder_type.lower()}')

        # Replace all parameters with Parameter(MetaParameter(...))
        with torch.no_grad():
            for name, param in self.named_parameters():
                module = self
                name_parts = name.split('.')
                # We have to traverse to the correct parent module and change the parameter there
                for part in name_parts[:-1]:
                    module = getattr(module, part)

                # Replace prameter with FastLLM meta parameter
                setattr(module, name_parts[-1], self.get_fastllm_parameter(name, param))
        
    def get_fastllm_parameter(self, param_name, param):
        param_dims = tuple([TensorDim(name=f'{param_name}_{idx}', global_size=x, parallel_dim=None) for idx, x in enumerate(param.shape)])
        return ParameterMeta(param.to("meta"), tensor_name=param_name, dims=param_dims, init_method=init_normal_(std=0.02), requires_grad=True, allow_no_grad=True)

    def _forward(self, input_: tuple[torch.Tensor], losses: dict | None = None, metrics: dict | None = None):
        if not self.image_encoder_type.lower() == "clip":
            raise ValueError(f'clip is the only image encoder type currrently supported')

        # TODO: Remove padding images
        # _bsz_im, num_img, ch, im_width, im_height = image_input
        # image_input = image_input.view(_bsz_im * num_img, *image_input.shape[2:])
        # num_values_per_image = image_input.shape[1:].numel()
        # real_images_inds = (image_input == 0.0).sum(dim=(-1, -2, -3)) != num_values_per_image
        # image_input = image_input[real_images_inds].contiguous()

        # (bsz, num_img, ch, im_h, im_w) -> (bsz*num_img, ch, im_h, im_w)


        # Convert the input images tensor to residual dtype. This is torch.float32 by default
        input_ = input_.to(self._residual_dtype)

        _bsz_im, num_img, ch, im_width, im_height = input_.shape
        input_ = input_.view(_bsz_im * num_img, *input_.shape[2:]).contiguous()
        
        out = self.visual_encoder(input_)[1]
        out = self.ln_vision(out)

        # (bsz*num_img, im_tokens, h) -> (bsz, num_img, im_tokens, h)
        out = out.view(_bsz_im, num_img, *out.shape[1:]).contiguous()

        return out.to(dtype=self._residual_dtype)

    def forward(self, input_, kwargs, losses: dict | None = None, metrics: dict | None = None):
        if input_ is None:
            raise ValueError(f'You must define a max_num_images > 0 if image_encoder is enabled')

        if isinstance(input_, TensorMeta):
            return TensorMeta.from_dims(
                kwargs[MultimodalModelKwargs.image_encoder_hidden_dims],
                tensor_name="Image encoder output",
                dtype=self._residual_dtype,
            )

        return self._forward(input_)