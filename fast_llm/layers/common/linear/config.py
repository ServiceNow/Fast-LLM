import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.engine.config_utils.initialization import Initialization, init_uniform_centered_, init_zeros_
from fast_llm.engine.config_utils.parameter import OptionalParameterConfig, ParameterConfig, combine_lr_scales
from fast_llm.engine.config_utils.tensor_dim import TensorDim, scalar_dim
from fast_llm.functional.config import ActivationType
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.layers.common.linear.convolution import CausalConv1d
    from fast_llm.layers.common.linear.linear import LinearBase


@config_class()
class LinearBaseConfig(Config):
    """
    Configuration for a linear-like layer without bias.
    """

    weight: ParameterConfig = Field(
        desc="Configuration for the weight.",
        hint=FieldHint.architecture,
    )
    lr_scale: float | None = Field(
        default=None,
        desc="Scaling factor for the layer learning rate."
        " Combines multiplicatively with the scale set by the parent layer and individual parameters, if applicable.",
        hint=FieldHint.feature,
    )


@config_class()
class AffineLinearBaseConfig(LinearBaseConfig):
    """
    Configuration for a linear-like layer with optional bias.
    """

    bias: OptionalParameterConfig = Field(
        desc="Use bias.",
        hint=FieldHint.architecture,
    )


@config_class()
class LinearConfig(LinearBaseConfig):
    apply_peft: bool | None = Field(
        default=None,
        desc="Wrap this layer ."
        " Otherwise, treat the layer as a non-peft layer (may be frozen)."
        " If not provided, the default set by the parent layer will be used.",
        hint=FieldHint.feature,
    )

    def get_layer(
        self,
        in_dim: TensorDim,
        out_dim: TensorDim,
        *,
        default_weight_initialization: Initialization,
        default_apply_peft: bool = False,
        sequence_parallel: bool = False,
        transposed_weight: bool = False,
        lr_scale: float | None,
        peft: PeftConfig | None,
    ) -> "LinearBase":
        from fast_llm.layers.common.linear.linear import InputParallelLinear, Linear, OutputParallelLinear

        lr_scale = combine_lr_scales(lr_scale, self.lr_scale)
        weight = self.weight.get_parameter(
            (in_dim, out_dim) if transposed_weight else (out_dim, in_dim),
            default_initialization=default_weight_initialization,
            lr_scale=lr_scale,
            peft=None,
        )

        if in_dim.parallel_dim is not None:
            assert out_dim.parallel_dim is None
            out = InputParallelLinear(
                weight,
                None,
                transposed_weight=transposed_weight,
                parallel_dim=in_dim.parallel_dim,
                sequence_parallel=sequence_parallel,
            )
        elif out_dim.parallel_dim is not None:
            out = OutputParallelLinear(
                weight,
                None,
                transposed_weight=transposed_weight,
                parallel_dim=out_dim.parallel_dim,
                sequence_parallel=sequence_parallel,
            )
        else:
            assert not sequence_parallel
            out = Linear(weight, None, transposed_weight=transposed_weight)

        if peft is not None:
            out = peft.apply_linear(out, default_apply_peft if self.apply_peft is None else self.apply_peft)

        return out


@config_class()
class AffineLinearConfig(AffineLinearBaseConfig, LinearConfig):
    def get_layer(
        self,
        in_dim: TensorDim,
        out_dim: TensorDim,
        *,
        default_weight_initialization: Initialization,
        default_bias_initialization: Initialization = init_zeros_,
        default_add_bias: bool = True,
        default_apply_peft: bool = False,
        sequence_parallel: bool = False,
        transposed_weight: bool = False,
        lr_scale: float | None,
        peft: PeftConfig | None,
    ) -> "LinearBase":
        from fast_llm.layers.common.linear.linear import InputParallelLinear, Linear, OutputParallelLinear

        lr_scale = combine_lr_scales(lr_scale, self.lr_scale)
        weight = self.weight.get_parameter(
            (in_dim, out_dim) if transposed_weight else (out_dim, in_dim),
            default_initialization=default_weight_initialization,
            lr_scale=lr_scale,
            peft=None,
        )
        bias = self.bias.get_parameter(
            (out_dim,),
            default_initialization=default_bias_initialization,
            lr_scale=lr_scale,
            default_enabled=default_add_bias,
            peft=None,
        )
        if in_dim.parallel_dim is not None:
            assert out_dim.parallel_dim is None
            out = InputParallelLinear(
                weight,
                bias,
                transposed_weight=transposed_weight,
                parallel_dim=in_dim.parallel_dim,
                sequence_parallel=sequence_parallel,
            )
        elif out_dim.parallel_dim is not None:
            out = OutputParallelLinear(
                weight,
                bias,
                transposed_weight=transposed_weight,
                parallel_dim=out_dim.parallel_dim,
                sequence_parallel=sequence_parallel,
            )
        else:
            assert not sequence_parallel
            out = Linear(weight, bias, transposed_weight=transposed_weight)

        if peft is not None:
            out = peft.apply_linear(out, default_apply_peft if self.apply_peft is None else self.apply_peft)

        return out


@config_class()
class CausalConv1dConfig(AffineLinearBaseConfig):
    """
    Configuration for a 1d causal convolution, as used in mamba layers.
    """

    kernel_size: int = Field(
        default=4,
        desc="Convolution kernel size.",
        hint=FieldHint.architecture,
        valid=check_field(Assert.gt, 0),
    )
    activation: ActivationType | None = Field(
        default=None,
        hint=FieldHint.architecture,
    )

    def get_layer(
        self,
        in_dim: TensorDim,
        *,
        default_weight_initialization: Initialization | None = None,
        default_bias_initialization: Initialization | None = None,
        default_add_bias: bool = True,
        default_activation: ActivationType = ActivationType.identity,
        lr_scale: float | None,
        peft: PeftConfig | None,
    ) -> "CausalConv1d":
        from fast_llm.layers.common.linear.convolution import CausalConv1d

        kernel_dim = TensorDim("convolution_kernel", self.kernel_size)

        if default_weight_initialization is None:
            default_weight_initialization = init_uniform_centered_(
                (in_dim.global_size * kernel_dim.global_size) ** -0.5
            )
        if default_bias_initialization is None:
            default_bias_initialization = init_uniform_centered_((in_dim.global_size * kernel_dim.global_size) ** -0.5)

        lr_scale = (combine_lr_scales(lr_scale, self.lr_scale),)
        weight = self.weight.get_parameter(
            (in_dim, scalar_dim, kernel_dim),
            default_initialization=default_weight_initialization,
            lr_scale=lr_scale,
            peft=peft,
        )
        bias = self.bias.get_parameter(
            (in_dim,),
            default_initialization=default_bias_initialization,
            lr_scale=lr_scale,
            default_enabled=default_add_bias,
            peft=peft,
        )
        return CausalConv1d(
            weight, bias, activation=default_activation if self.activation is None else self.activation
        )
