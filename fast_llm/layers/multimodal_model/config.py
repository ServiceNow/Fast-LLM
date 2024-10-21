import enum
from fast_llm.config import Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.engine.base_model.config import BaseModelArchitectureConfig, BaseModelConfig
from fast_llm.engine.config_utils.tensor_space import TensorDim, TensorSpace

from fast_llm.utils import Assert

class MultimodalModelDimNames:
    # Image encoder dimensions
    max_num_images = "max_num_images"
    image_pixel_count = "image_pixel_count"
    num_image_tokens = "num_image_tokens"
    image_encoder_hidden_size = "image_encoder_hidden_size"

class MultimodalModelKwargs:
    image_encoder_hidden_dims = "image_encoder_hidden_dims"
    adapter_hidden_dims = "adapter_hidden_dims"

class ImageEncoderType(str, enum.Enum):
    clip = "clip"
    docowl = "docowl"

@config_class()
class MultimodalModelArchitectureConfig(BaseModelArchitectureConfig):
    _abstract = False
    
    image_encoder_hidden_size: int = Field(
        default=1024,
        desc="Hidden size of image encoder.",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    num_image_tokens: int = Field(
        default=256,
        desc="Number of image tokens.",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    max_num_images: int = Field(
        default=10,
        desc="Max. number of images in a sample. We pad to ensure shapes are consistent.",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    image_resolution: int = Field(
        default=448,
        desc="Resolution of image",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )

    def _validate(self):
        super()._validate()

    def setup_tensor_space(self, tensor_space: TensorSpace):
        tensor_space.add_tensor_dim(TensorDim(MultimodalModelDimNames.max_num_images, self.max_num_images))
        tensor_space.add_tensor_dim(TensorDim(MultimodalModelDimNames.num_image_tokens, self.num_image_tokens))
        tensor_space.add_tensor_dim(TensorDim(MultimodalModelDimNames.image_pixel_count, self.image_resolution * self.image_resolution))
        tensor_space.add_tensor_dim(TensorDim(MultimodalModelDimNames.image_encoder_hidden_size, self.image_encoder_hidden_size))


@config_class()
class MultimodalModelBaseConfig(MultimodalModelArchitectureConfig, BaseModelConfig):
    """
    A configuration class for defining the model configuration of encoder and adapter components of multi-modal model.
    """
    _abstract = False

    image_encoder_type: ImageEncoderType = Field(
        default=ImageEncoderType.clip,
        desc="Type of image encoder",
        hint=FieldHint.feature,
    )
    
    def _validate(self):
        super()._validate()