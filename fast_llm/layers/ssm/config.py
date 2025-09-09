import enum
import math
import typing

from fast_llm.config import Field, FieldHint, check_field, config_class
from fast_llm.engine.config_utils.initialization import InitializationConfig, Initializer, LambdaInitializer
from fast_llm.engine.config_utils.parameter import ParameterConfig
from fast_llm.layers.block.config import MixerConfig
from fast_llm.layers.common.linear.config import AffineLinearConfig, CausalConv1dConfig, LinearConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    import torch

    from fast_llm.layers.ssm.discrete_mamba2 import DiscreteMamba2
    from fast_llm.layers.ssm.mamba import Mamba
    from fast_llm.tensor import ParameterMeta


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
        default=2048,
        desc="Inner dimension.",
        hint=FieldHint.core,
    )

    # Model options
    add_linear_biases: bool = Field(
        default=True,
        desc="Add biases to linear layers. May be overridden for individual layers.",
        hint=FieldHint.architecture,
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
        default=64,
        desc="Rank of the Δ projection matrix.",
        hint=FieldHint.architecture,
    )


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

    @property
    def layer_class(self) -> "type[Mamba]":
        from fast_llm.layers.ssm.mamba import Mamba

        return Mamba


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
        default=1024,
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

    @property
    def layer_class(self) -> "type[Mamba2]":
        from fast_llm.layers.ssm.mamba2 import Mamba2

        return Mamba2


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

    @property
    def layer_class(self) -> "type[DiscreteMamba2]":
        from fast_llm.layers.ssm.discrete_mamba2 import DiscreteMamba2

        return DiscreteMamba2


@config_class(dynamic_type={InitializationConfig: "mamba_dt_bias"})
class MambaDTBiasInitializationConfig(InitializationConfig):
    """
    Configuration for the common Mamba DT bias initialization scheme.
    """

    _abstract = False
    # dt_bias_initialization_min [Mamba, Mamba2]
    min_step_size: float = Field(
        default=0.001,
        desc="Minimum step size for discretization",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    # dt_bias_initialization_max [Mamba, Mamba2]
    max_step_size: float = Field(
        default=0.1,
        desc="Maximum step size for discretization",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    # dt_bias_initialization_floor [Mamba, Mamba2]
    floor: float = Field(
        default=1e-4,
        desc="Minimum value for initializing dt",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )

    def _validate(self) -> None:
        super()._validate()
        Assert.geq(self.max_step_size, self.min_step_size)

    def get_initializer(self) -> Initializer:
        return init_dtprojbias(self.min_step_size, self.max_step_size, self.floor)


@config_class(dynamic_type={InitializationConfig: "mamba_a"})
class MambaAInitializationConfig(InitializationConfig):
    """
    Initialization configuration for Mamba A parameter.
    Not particularly useful outside the default A initialization, but still made available for convenience.
    """

    _abstract = False
    # dt_bias_initialization_min [Mamba, Mamba2]
    state_size: int = Field(
        desc="State size. Needs to be repeated here so the initializer knows about it.",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )
    # dt_bias_initialization_max [Mamba, Mamba2]
    d_inner: int = Field(
        desc="Inner dimension. Needs to be repeated here so the initializer knows about it.",
        hint=FieldHint.core,
        valid=check_field(Assert.gt, 0),
    )

    def get_initializer(self) -> Initializer:
        return init_a(self.state_size, self.d_inner)


def init_dtprojbias(
    min_step_size: float = 0.001, max_step_size: float = 0.1, floor: float = 1e-4
) -> LambdaInitializer:
    def init_(meta: "ParameterMeta", tensor: "torch.Tensor", generator: "torch.Generator"):  # noqa
        import torch

        tensor.uniform_(math.log(min_step_size), math.log(max_step_size), generator=generator).exp_().clamp_min_(floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        tensor.add_(torch.log(-torch.expm1(-tensor)))

    return LambdaInitializer(init_)


def init_a(d_state, d_inner) -> LambdaInitializer:
    def init_(meta: "ParameterMeta", tensor: "torch.Tensor", generator: "torch.Generator") -> None:  # noqa
        import torch

        Assert.eq(tensor.numel(), d_state * d_inner)
        torch.log(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=tensor.device)
            .unsqueeze(0)
            .expand(d_inner, d_state),
            out=tensor,
        )

    return LambdaInitializer(init_, requires_global_initialization=True)
