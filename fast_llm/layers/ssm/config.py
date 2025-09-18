import enum
import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class, skip_valid_if_none
from fast_llm.engine.config_utils.parameter import ParameterConfig
from fast_llm.layers.common.linear.config import AffineLinearConfig, CausalConv1dConfig, LinearConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.engine.config_utils.initialization import Initializer


class SSMBlockType(enum.StrEnum):
    """
    An enum for the available mamba types for the MLP layer.
    """

    mamba = "m"
    mamba2_discrete = "m2d"
    mamba2 = "m2"
    transformer = "t"

    def get_mixer_class(self):
        if self == SSMBlockType.mamba:
            from fast_llm.layers.ssm.mamba import Mamba

            return Mamba
        elif self == SSMBlockType.mamba2:
            from fast_llm.layers.ssm.mamba2 import Mamba2

            return Mamba2
        elif self == SSMBlockType.mamba2_discrete:
            from fast_llm.layers.ssm.discrete_mamba2 import DiscreteMamba2

            return DiscreteMamba2
        else:
            raise NotImplementedError(self)


class DTInitType(enum.StrEnum):
    constant = "constant"
    random = "random"

    def get_init_method(self, scale: float) -> "Initializer":
        from fast_llm.engine.config_utils.initialization import init_fill_, init_uniform_centered_

        return init_fill_(scale) if self == DTInitType.constant else init_uniform_centered_(scale)


@config_class(registry=True)
class MixerConfig(Config):
    """
    Base config class for all mixers.
    TODO: Generalize to include Attention
    """

    _abstract = True


@config_class()
class SSMConfig(MixerConfig):
    # Layers
    # [Mamba, Mamba2, DiscreteMamba2]
    z_layer: AffineLinearConfig = Field(
        desc="Configuration for the z layer.",
        hint=FieldHint.architecture,
    )
    # [Mamba, Mamba2, DiscreteMamba2]
    x_layer: AffineLinearConfig = Field(
        desc="Configuration for the x layer.",
        hint=FieldHint.architecture,
    )
    # [Mamba, Mamba2, DiscreteMamba2]
    convolution_layer: CausalConv1dConfig = Field(
        desc="Configuration for the convolution layer.",
        hint=FieldHint.architecture,
    )
    # [Mamba, Mamba2, DiscreteMamba2]
    d_weight: ParameterConfig = Field(
        desc='Configuration for the D "skip" weight.',
        hint=FieldHint.architecture,
    )
    # [Mamba, Mamba2, DiscreteMamba2]
    output_layer: AffineLinearConfig = Field(
        desc="Configuration for the output layer.",
        hint=FieldHint.architecture,
    )

    # Model dimensions
    # head_size [Mamba, Mamba2, DiscreteMamba2]
    state_size: int = Field(
        default=16,
        desc="State size.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    # [Mamba, Mamba2, DiscreteMamba2]
    # c_size [Mamba, Mamba2, DiscreteMamba2]?
    d_inner: int = Field(
        default=None,
        desc="Inner dimension.",
        hint=FieldHint.core,
    )

    # Learning rate
    # lr_scale [MambaLayer, Mamba2, DiscreteMamba2]
    mamba_lr_scale: float | None = Field(
        default=None,
        desc="Learning rate scale for Mamba blocks.",
        hint=FieldHint.feature,
        valid=skip_valid_if_none(check_field(Assert.geq, 0)),
    )


@config_class()
class MambaBaseConfig(SSMConfig):
    """
    Common configuration for Mamba and Mamba2.
    """

    _abstract = False

    # Layers
    dt_layer: AffineLinearConfig = Field(
        desc="Configuration for the dt layer.",
        hint=FieldHint.architecture,
    )
    a_log_weight: ParameterConfig = Field(
        desc="Configuration for the a_log layer weight.",
        hint=FieldHint.architecture,
    )

    # Model dimensions
    #  [Mamba, Mamba2]
    dt_rank: int = Field(
        default=None,
        desc="Rank of the Δ projection matrix. If 'None', will be set to ceil(hidden_size/16)",
        hint=FieldHint.architecture,
    )

    # Initialization
    # dt_bias_initialization_min [Mamba, Mamba2]
    dt_min: float = Field(
        default=0.001,
        desc="Minimum step size for discretization",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    # dt_bias_initialization_max [Mamba, Mamba2]
    dt_max: float = Field(
        default=0.1,
        desc="Maximum step size for discretization",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    # dt_bias_initialization_floor [Mamba, Mamba2]
    dt_init_floor: float = Field(
        default=1e-4,
        desc="Minimum value for initializing dt",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )

    def _validate(self) -> None:
        super()._validate()
        Assert.geq(self.dt_max, self.dt_min)


@config_class(dynamic_type={MixerConfig: "mamba"})
class MambaConfig(MambaBaseConfig):
    """
    Configuration for Mamba.
    """

    # Layers
    # TODO: Can be confused with `x_layer`
    x_projection_layer: LinearConfig = Field(
        desc="Configuration for the x projection layer.",
        hint=FieldHint.architecture,
    )

    def _validate(self) -> None:
        super()._validate()
        Assert.none(self.convolution_layer.activation)
        # TODO: (Oleksiy) If bias is used there is a problem in the MambaInnerFn.backward for the bias grads.
        #  I think this bias is not used in other mamba repos.
        assert not self.output_layer.bias.enabled


@config_class(dynamic_type={MixerConfig: "mamba_2"})
class Mamba2Config(MambaBaseConfig):
    """
    Configuration for Mamba2.
    TODO: Actually a variation of Mamba 2.
    """

    _abstract = False

    # Layers
    # [Mamba2, DiscreteMamba2]
    b_layer: AffineLinearConfig = Field(
        desc="Configuration for the b layer.",
        hint=FieldHint.architecture,
    )
    # [Mamba2, DiscreteMamba2]
    c_layer: AffineLinearConfig = Field(
        desc="Configuration for the c layer.",
        hint=FieldHint.architecture,
    )
    dt_input_layer: AffineLinearConfig = Field(
        desc="Configuration for the dt input projection layer.",
        hint=FieldHint.architecture,
    )

    # Model dimensions
    # xb_size [Mamba2]
    d_xb: int = Field(
        default=None,
        desc="Dimension of the xB in Mamba2 blocks.",
        hint=FieldHint.architecture,
    )

    # Model options
    # repeat_xb_before_conv [Mamba2]
    repeat_kv_before_conv: bool = Field(
        default=True,
        desc="Whether to repeat x and B before (True) or after (False) the conv1d in Mamba2 blocks.",
        hint=FieldHint.architecture,
    )

    # Initialization
    # dt_weight_initialization_method [Mamba2]
    dt_init: DTInitType = Field(
        default=DTInitType.random,
        desc="Initialization method for dt",
        hint=FieldHint.core,
    )
    # dt_weight_initialization_scale [Mamba2]
    dt_scale: float = Field(
        default=1.0,
        desc="Scale for dt",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )


@config_class(dynamic_type={MixerConfig: "discrete_mamba_2"})
class DiscreteMamba2Config(SSMConfig):
    """
    Configuration for DiscreteMamba2.
    """

    _abstract = False
    # Layers
    # [Mamba2, DiscreteMamba2]
    b_layer: AffineLinearConfig = Field(
        desc="Configuration for the b layer.",
        hint=FieldHint.architecture,
    )
    # [Mamba2, DiscreteMamba2]
    c_layer: AffineLinearConfig = Field(
        desc="Configuration for the c layer.",
        hint=FieldHint.architecture,
    )

    # Model dimensions
    # head_groups [DiscreteMamba2]
    n_qk_heads: int = Field(
        default=32,
        desc="Number of QK heads.",
        hint=FieldHint.architecture,
    )
    # heads [DiscreteMamba2]
    n_v_heads: int = Field(
        default=32,
        desc="Number of V heads.",
        hint=FieldHint.architecture,
    )
    # chunk_size [DiscreteMamba2]
    chunk_size: int = Field(
        default=256,
        desc="Chunk size for Mamba2 blocks.",
        hint=FieldHint.architecture,
    )
