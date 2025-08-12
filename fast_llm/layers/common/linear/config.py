import typing

from fast_llm.config import Config, Field, FieldHint, config_class
from fast_llm.engine.config_utils.initialization import InitializationConfig
from fast_llm.layers.common.peft.config import PeftConfig

if typing.TYPE_CHECKING:
    from fast_llm.engine.config_utils.tensor_space import TensorDim
    from fast_llm.layers.common.linear.linear import InputParallelLinear, Linear, LinearLike, OutputParallelLinear


@config_class()
class LinearConfig(Config):
    bias: bool = Field(
        default=None,
        desc="Use bias.",
        hint=FieldHint.architecture,
    )
    weight_initialization: InitializationConfig = Field(
        desc="Initialization configuration for the weight.",
        hint=FieldHint.feature,
    )
    bias_initialization: InitializationConfig = Field(
        desc="Initialization configuration for the bias.",
        hint=FieldHint.feature,
    )
    lr_scale: float | None | tuple[float | None, ...] = (None,)
    apply_peft: bool = Field(
        default=None,
        desc="Apply peft on this layer if defined. Otherwise, treat the layer as a non-peft layer (may be frozen).",
        hint=FieldHint.feature,
    )
    # Fixed defaults don't make sense because each parent layer uses its own.
    # Instead, we use this variable to set defaults dynamically.
    # This can either be a constant,
    # or may point to another config, ex. to set a default for all layers in a model.
    default: "LinearConfig" = Field(init=False)

    def _validate(self) -> None:
        if hasattr(self, "default"):
            self.default.validate()
            with self._set_implicit_default():
                if self.bias is None:
                    self.bias = self.default.bias
                if self.weight_initialization.is_default:
                    self.weight_initialization = self.default.weight_initialization
                if self.bias_initialization.is_default:
                    self.bias_initialization = self.default.bias_initialization
                if self.lr_scale is None:
                    self.lr_scale = self.default.lr_scale
                if self.apply_peft is None:
                    self.apply_peft = self.default.apply_peft
        if None in (self.bias, self.weight_initialization, self.bias_initialization, self.lr_scale, self.apply_peft):
            raise ValueError("Missing default values for linear layer.")

        super()._validate()

    def get_layer(
        self,
        in_dim: TensorDim,
        out_dim: TensorDim,
        *,
        sequence_parallel: bool = False,
        transposed_weight: bool = False,
        auto_bias_grad_accumulation: bool = False,
        lr_scale: float | None | tuple[float | None, ...],
        peft: PeftConfig | None = None,
    ) -> "LinearLike":
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
            auto_bias_grad_accumulation=auto_bias_grad_accumulation,
            lr_scale=lr_scale,
        )
        if peft is not None:
            out = peft.apply_linear(out, self.apply_peft)
        return out
