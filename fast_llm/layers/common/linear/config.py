import typing

from fast_llm.config import Config, Field, FieldHint, config_class
from fast_llm.engine.config_utils.initialization import InitializationConfig
from fast_llm.layers.common.peft.config import PeftConfig
from fast_llm.utils import Assert, combine_lr_scales

if typing.TYPE_CHECKING:
    from fast_llm.engine.config_utils.tensor_dim import TensorDim
    from fast_llm.tensor import ParameterMeta


# TODO: Move
@config_class()
class WeightConfig(Config):
    initialization: InitializationConfig = Field(
        desc="Initialization configuration.",
        hint=FieldHint.feature,
    )
    lr_scale: float | None = Field(
        default=None,
        desc="Scaling factor for the learning rate.",
        hint=FieldHint.feature,
    )

    # Fixed defaults don't make sense because each parent layer uses its own.
    # Instead, we use this variable to set defaults dynamically.
    # This can either be a constant,
    # or may point to another config, ex. to set a default for all layers in a model.
    default: typing.Self = Field(init=False)

    def _validate(self) -> None:
        if hasattr(self, "default"):
            Assert.eq(type(self), type(self.default))
            self.default.validate()
            with self._set_implicit_default():
                self._apply_default()
        super()._validate()

    def _apply_default(self) -> None:
        # Excluding initialization to avoid messing the config, evaluate at runtime instead.
        if self.lr_scale is None:
            self.lr_scale = self.default.lr_scale

    def get_weight(
        self,
        *dims: "TensorDim",
        auto_grad_accumulation: bool,
        lr_scale: float | None,
        weight_decay: bool = True,
        peft: PeftConfig | None,
        allow_sequence_tensor_parallel: bool = True,
    ) -> "ParameterMeta":
        from fast_llm.tensor import ParameterMeta

        # TODO: Recurse to self.default.default?
        if not self.initialization.is_default:
            initialization = self.initialization
        elif hasattr(self, "default") and not self.default.initialization.is_default:
            initialization = self.default.initialization
        else:
            raise ValueError("Missing default value for `initialization`.")

        out = ParameterMeta.from_dims(
            dims,
            init_method=initialization.get_initializer(),
            auto_grad_accumulation=auto_grad_accumulation,
            lr_scale=combine_lr_scales(self.lr_scale, lr_scale),
            weight_decay=weight_decay,
            allow_sequence_tensor_parallel=allow_sequence_tensor_parallel,
        )
        if peft is not None:
            out = peft.apply_weight(out)
        return out


@config_class()
class LinearBaseConfig(Config):
    """
    Configuration for a linear-like layer without bias.
    """

    weight_initialization: InitializationConfig = Field(
        desc="Initialization configuration for the weight.",
        hint=FieldHint.feature,
    )
    lr_scale: float | None = Field(
        default=None,
        desc="Scaling factor for the learning rate.",
        hint=FieldHint.feature,
    )
    # Fixed defaults don't make sense because each parent layer uses its own.
    # Instead, we use this variable to set defaults dynamically.
    # This can either be a constant,
    # or may point to another config, ex. to set a default for all layers in a model.
    default: typing.Self = Field(init=False)

    def _validate(self) -> None:
        if hasattr(self, "default"):
            Assert.custom(isinstance, self.default, type(self))
            self.default.validate()
            with self._set_implicit_default():
                self._apply_default()
        super()._validate()

    def _apply_default(self) -> None:
        # Excluding initialization to avoid messing the config, evaluate at runtime instead.
        if self.lr_scale is None:
            self.lr_scale = self.default.lr_scale

    def get_weight(
        self,
        in_dim: "TensorDim",
        out_dim: "TensorDim",
        *,
        auto_grad_accumulation: bool,
        transposed: bool = False,
        lr_scale: float | None = None,
        # Note: weights created this way are treated as generic parameters WRT peft.
        peft: PeftConfig | None,
    ) -> "ParameterMeta":
        from fast_llm.tensor import ParameterMeta

        # TODO: Recurse to self.default.default?
        if not self.weight_initialization.is_default:
            initialization = self.weight_initialization
        elif hasattr(self, "default") and not self.default.weight_initialization.is_default:
            initialization = self.default.weight_initialization
        else:
            raise ValueError("Missing default value for `weight_initialization`.")

        out = ParameterMeta.from_dims(
            (in_dim, out_dim) if transposed else (out_dim, in_dim),
            init_method=initialization.get_initializer(),
            auto_grad_accumulation=auto_grad_accumulation,
            lr_scale=combine_lr_scales(self.lr_scale, lr_scale),
        )
        if peft is not None:
            out = peft.apply_weight(out)
        return out


@config_class()
class AffineLinearBaseConfig(LinearBaseConfig):
    """
    Configuration for a linear-like layer with optional bias.
    """

    bias_initialization: InitializationConfig = Field(
        desc="Initialization configuration for the bias.",
        hint=FieldHint.feature,
    )
    # TODO: Could use dynamic type instead?
    bias: bool = Field(
        default=None,
        desc="Use bias.",
        hint=FieldHint.architecture,
    )

    def _apply_default(self) -> None:
        super()._apply_default()
        if self.bias is None:
            self.bias = self.default.bias
        if self.bias is None:
            raise ValueError("Missing default value for `bias`.")

    def get_bias(
        self,
        out_dim: "TensorDim",
        *,
        auto_grad_accumulation: bool,
        lr_scale: float | None = None,
        # Note: biases created this way are treated as generic parameters WRT peft.
        peft: PeftConfig | None,
    ) -> "ParameterMeta | None":
        if self.bias:
            from fast_llm.tensor import ParameterMeta

            # TODO: Recurse to self.default.default?
            if not self.bias_initialization.is_default:
                initialization = self.bias_initialization.get_initializer()
            elif hasattr(self, "default") and not self.default.bias_initialization.is_default:
                initialization = self.default.bias_initialization
            else:
                raise ValueError("Missing default value for `bias_initialization`.")

            out = ParameterMeta.from_dims(
                (out_dim,),
                init_method=initialization.get_initializer(),
                weight_decay=False,
                auto_grad_accumulation=auto_grad_accumulation,
                lr_scale=combine_lr_scales(self.lr_scale, lr_scale),
            )
            if peft is not None:
                out = peft.apply_weight(out)
            return out
        else:
            return None


@config_class()
class LinearConfig(LinearBaseConfig):
    """
    Configuration for a linear layer without bias.
    """

    apply_peft: bool = Field(
        default=None,
        desc="Apply peft on this layer if defined. Otherwise, treat the layer as a non-peft layer (may be frozen).",
        hint=FieldHint.feature,
    )

    def _apply_default(self) -> None:
        super()._apply_default()
        if self.apply_peft is None:
            self.apply_peft = self.default.apply_peft
        if self.apply_peft is None:
            raise ValueError("Missing default value for `apply_peft`.")

    def get_layer(
        self,
        in_dim: "TensorDim",
        out_dim: "TensorDim",
        *,
        sequence_parallel: bool = False,
        transposed_weight: bool = False,
        auto_weight_grad_accumulation: bool = False,
        lr_scale: float | None,
        peft: PeftConfig | None,
    ) -> "LinearLike":
        from fast_llm.layers.common.linear.linear import InputParallelLinear, Linear, OutputParallelLinear

        if in_dim.parallel_dim is not None:
            assert out_dim.parallel_dim is None
            cls = InputParallelLinear
        elif out_dim.parallel_dim is not None:
            cls = OutputParallelLinear
        else:
            assert not sequence_parallel
            cls = Linear
        out = cls(
            self,
            in_dim,
            out_dim,
            transposed_weight=transposed_weight,
            sequence_parallel=sequence_parallel,
            auto_weight_grad_accumulation=auto_weight_grad_accumulation,
            # `combine_lr_scales` called in `get_weight`.
            lr_scale=lr_scale,
        )
        if peft is not None:
            out = peft.apply_linear(out, self.apply_peft)
        return out


@config_class()
class AffineLinearConfig(AffineLinearBaseConfig, LinearConfig):
    def get_layer(
        self,
        in_dim: "TensorDim",
        out_dim: "TensorDim",
        *,
        sequence_parallel: bool = False,
        transposed_weight: bool = False,
        auto_weight_grad_accumulation: bool = False,
        auto_bias_grad_accumulation: bool = False,
        lr_scale: float | None,
        peft: PeftConfig | None,
    ) -> "LinearLike":
        from fast_llm.layers.common.linear.linear import (
            AffineInputParallelLinear,
            AffineLinear,
            AffineOutputParallelLinear,
        )

        if in_dim.parallel_dim is not None:
            assert out_dim.parallel_dim is None
            cls = AffineInputParallelLinear
        elif out_dim.parallel_dim is not None:
            cls = AffineOutputParallelLinear
        else:
            assert not sequence_parallel
            cls = AffineLinear
        out = cls(
            self,
            in_dim,
            out_dim,
            transposed_weight=transposed_weight,
            sequence_parallel=sequence_parallel,
            auto_weight_grad_accumulation=auto_weight_grad_accumulation,
            auto_bias_grad_accumulation=auto_bias_grad_accumulation,
            lr_scale=combine_lr_scales(self.lr_scale, lr_scale),
        )
        if peft is not None:
            out = peft.apply_linear(out, self.apply_peft)
        return out
