import typing

import torch.nn

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.layers.common.linear.config import AffineLinearConfig, LinearConfig, WeightConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    pass

torch.nn.Linear


@config_class()
class MambaConfig(Config):
    _abstract = False

    # Layers
    z_layer: AffineLinearConfig = Field(
        desc="Configuration for the z layer.",
        hint=FieldHint.architecture,
    )
    x_layer: AffineLinearConfig = Field(
        desc="Configuration for the x layer.",
        hint=FieldHint.architecture,
    )
    #  TODO: Conv config? Bias?
    convolution_layer: WeightConfig = Field(
        desc="Configuration for the convolution weight.",
        hint=FieldHint.architecture,
    )
    # TODO: Can be confused with `x_layer`
    x_projection_layer: LinearConfig = Field(
        desc="Configuration for the x projection layer.",
    )
    dt_layer: AffineLinearConfig = Field(
        desc="Configuration for the dt projection layer.",
    )
    a_log_layer: WeightConfig = Field(
        desc="Configuration for the A_log layer.",
        hint=FieldHint.architecture,
    )
    d_layer: WeightConfig = Field(
        desc="Configuration for the D layer.",
        hint=FieldHint.architecture,
    )
    # TODO: note, if bias is used there is a problem in the MambaInnerFn.backward for the bias grads.
    #  I think this bias is not used in other mamba repos.
    output_layer: LinearConfig = Field(
        desc="Configuration for the output layer.",
    )

    # Model dimensions
    state_size: int = Field(
        default=16,
        desc="State size.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    conv_kernel_dimension: int = Field(
        default=4,
        desc="Conv kernel dimension.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    dt_rank: None | int = Field(
        default=None,
        desc="Rank of the Δ projection matrix. If 'None', will be set to ceil(hidden_size/16)",
        hint=FieldHint.architecture,
    )
    d_inner: None | int = Field(
        default=None,
        desc="Inner dimension.",
        hint=FieldHint.core,
    )

    def _validate(self):
        super()._validate()
        self.z_layer.default = AffineLinearConfig(
            bias=False,
            weight_initialization=init_normal_(0, (self.hidden_size * scale) ** -0.5),
            bias_initialization=init_zeros_,
            lr_scale=None,
            enable_peft=False,
        )
