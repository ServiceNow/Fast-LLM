import math
import typing

from fast_llm.config import Config, Field, FieldHint, config_class
from fast_llm.engine.config_utils.initialization import Initialization, InitializationConfig
from fast_llm.engine.config_utils.tensor_dim import TensorDim
from fast_llm.layers.common.peft.config import PeftConfig

if typing.TYPE_CHECKING:
    from fast_llm.tensor import ParameterMeta


def combine_lr_scales(*lr_scales: float | None | tuple[float | None, ...]):
    # Remove `None` entries.
    lr_scales = tuple(lr_scale for lr_scale in lr_scales if lr_scale is not None)
    if not lr_scales:
        # Everything is None
        return None
    tuple_length = None
    # Check if we have tuples, and determine the length.
    for lr_scale in lr_scales:
        if isinstance(lr_scale, tuple):
            if tuple_length is None:
                tuple_length = len(lr_scale)
            else:
                assert len(lr_scale) == tuple_length
    if tuple_length is None:
        # No tuple: simple product.
        return math.prod(lr_scales)
    else:
        # Tuple(s): use recursion.
        return tuple(
            combine_lr_scales(*[lr_scale[i] if isinstance(lr_scale, tuple) else lr_scale for lr_scale in lr_scales])
            for i in range(tuple_length)
        )


@config_class()
class ParameterConfig(Config):
    initialization: InitializationConfig = Field(
        desc="If provided, override the default initialization method set by the parent layer.",
        hint=FieldHint.feature,
    )
    lr_scale: float | None = Field(
        default=None,
        desc="Scaling factor for the parameter learning rate."
        " Combines multiplicatively with the scale set by the parent layer, if applicable.",
        hint=FieldHint.feature,
    )
    # TODO: Initialization, lr_scale

    def _validate(self) -> None:
        pass

    def get_parameter(
        self,
        dims: tuple[TensorDim, ...],
        *,
        default_initialization: Initialization,
        lr_scale: float | None,
        weight_decay: bool = True,
        allow_sequence_tensor_parallel: bool = True,
        peft: PeftConfig | None,
    ) -> "ParameterMeta":
        from fast_llm.tensor import ParameterMeta

        out = ParameterMeta.from_dims(
            dims,
            init_method=default_initialization if self.initialization.is_default else self.initialization,
            lr_scale=combine_lr_scales(lr_scale, self.lr_scale),
            weight_decay=weight_decay,
            allow_sequence_tensor_parallel=allow_sequence_tensor_parallel,
        )
        if peft is not None:
            out = peft.apply_weight(out)
        return out


@config_class()
class OptionalParameterConfig(ParameterConfig):
    enabled: bool | None = Field(
        default=None,
    )

    def _validate(self) -> None:
        pass

    def get_parameter(
        self,
        dims: tuple[TensorDim, ...],
        *,
        default_initialization: Initialization,
        lr_scale: float | None,
        weight_decay: bool = True,
        allow_sequence_tensor_parallel: bool = True,
        default_enabled: bool = False,
        peft: PeftConfig | None,
    ) -> "ParameterMeta|None":
        pass

        if (self.enabled is None and default_enabled) or self.enabled:
            return super().get_parameter(
                dims,
                default_initialization=default_initialization,
                lr_scale=lr_scale,
                weight_decay=weight_decay,
                allow_sequence_tensor_parallel=allow_sequence_tensor_parallel,
                peft=peft,
            )
        else:
            return None
