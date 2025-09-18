import typing

from fast_llm.config import Config, Field, FieldHint, check_field, config_class
from fast_llm.engine.config_utils.initialization import Initializer, init_uniform_centered_, init_zeros_
from fast_llm.engine.config_utils.parameter import OptionalParameterConfig, ParameterConfig
from fast_llm.engine.config_utils.tensor_dim import TensorDim, scalar_dim
from fast_llm.functional.config import ActivationType
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
        desc="Initialization configuration for the weight.",
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
    def get_layer(
        self,
        in_dim: TensorDim,
        out_dim: TensorDim,
        *,
        default_weight_initializer: Initializer,
        sequence_parallel: bool = False,
        transposed_weight: bool = False,
        lr_scale: float | None,
    ) -> "LinearBase":
        from fast_llm.layers.common.linear.linear import InputParallelLinear, Linear, OutputParallelLinear

        weight = self.weight.get_parameter(
            (in_dim, out_dim) if transposed_weight else (out_dim, in_dim),
            default_initializer=default_weight_initializer,
            lr_scale=lr_scale,
        )
        if in_dim.parallel_dim is not None:
            assert out_dim.parallel_dim is None
            return InputParallelLinear(
                weight,
                None,
                transposed_weight=transposed_weight,
                parallel_dim=in_dim.parallel_dim,
                sequence_parallel=sequence_parallel,
            )
        elif out_dim.parallel_dim is not None:
            return OutputParallelLinear(
                weight,
                None,
                transposed_weight=transposed_weight,
                parallel_dim=out_dim.parallel_dim,
                sequence_parallel=sequence_parallel,
            )
        else:
            assert not sequence_parallel
            return Linear(weight, None, transposed_weight=transposed_weight)


@config_class()
class AffineLinearConfig(AffineLinearBaseConfig, LinearConfig):
    def get_layer(
        self,
        in_dim: TensorDim,
        out_dim: TensorDim,
        *,
        default_weight_initializer: Initializer,
        default_bias_initializer: Initializer = init_zeros_,
        default_add_bias: bool = True,
        sequence_parallel: bool = False,
        transposed_weight: bool = False,
        lr_scale: float | None,
    ) -> "LinearBase":
        from fast_llm.layers.common.linear.linear import InputParallelLinear, Linear, OutputParallelLinear

        weight = self.weight.get_parameter(
            (in_dim, out_dim) if transposed_weight else (out_dim, in_dim),
            default_initializer=default_weight_initializer,
            lr_scale=lr_scale,
        )
        bias = self.bias.get_parameter(
            (out_dim,),
            default_initializer=default_bias_initializer,
            lr_scale=lr_scale,
            default_enabled=default_add_bias,
        )
        if in_dim.parallel_dim is not None:
            assert out_dim.parallel_dim is None
            return InputParallelLinear(
                weight,
                bias,
                transposed_weight=transposed_weight,
                parallel_dim=in_dim.parallel_dim,
                sequence_parallel=sequence_parallel,
            )
        elif out_dim.parallel_dim is not None:
            return OutputParallelLinear(
                weight,
                bias,
                transposed_weight=transposed_weight,
                parallel_dim=out_dim.parallel_dim,
                sequence_parallel=sequence_parallel,
            )
        else:
            assert not sequence_parallel
            return Linear(weight, bias, transposed_weight=transposed_weight)


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
        default_weight_initializer: Initializer | None = None,
        default_bias_initializer: Initializer | None = None,
        default_add_bias: bool = True,
        default_activation: ActivationType = ActivationType.identity,
        lr_scale: float | None,
    ) -> "CausalConv1d":
        from fast_llm.layers.common.linear.convolution import CausalConv1d

        kernel_dim = TensorDim("convolution_kernel", self.kernel_size)

        if default_weight_initializer is None:
            default_weight_initializer = init_uniform_centered_((in_dim.global_size * kernel_dim.global_size) ** -0.5)
        if default_bias_initializer is None:
            default_bias_initializer = init_uniform_centered_((in_dim.global_size * kernel_dim.global_size) ** -0.5)

        weight = self.weight.get_parameter(
            (in_dim, scalar_dim, kernel_dim),
            default_initializer=default_weight_initializer,
            lr_scale=lr_scale,
        )
        bias = self.bias.get_parameter(
            (in_dim,),
            default_initializer=default_bias_initializer,
            lr_scale=lr_scale,
            default_enabled=default_add_bias,
        )
        return CausalConv1d(
            weight, bias, activation=default_activation if self.activation is None else self.activation
        )
