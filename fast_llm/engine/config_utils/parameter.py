import typing

from fast_llm.config import Config, Field, config_class
from fast_llm.engine.config_utils.initialization import Initializer
from fast_llm.engine.config_utils.tensor_dim import TensorDim

if typing.TYPE_CHECKING:
    from fast_llm.tensor import ParameterMeta


@config_class()
class ParameterConfig(Config):
    # TODO: Initialization, lr_scale

    def _validate(self) -> None:
        pass

    def get_parameter(
        self,
        dims: tuple[TensorDim, ...],
        default_initializer: Initializer,
        lr_scale: float | None,
        weight_decay: bool = True,
        allow_sequence_tensor_parallel: bool = True,
    ) -> "ParameterMeta":
        from fast_llm.tensor import ParameterMeta

        return ParameterMeta.from_dims(
            dims,
            init_method=default_initializer,
            lr_scale=lr_scale,
            weight_decay=weight_decay,
            allow_sequence_tensor_parallel=allow_sequence_tensor_parallel,
        )


@config_class()
class OptionalParameterConfig(ParameterConfig):
    enabled: bool | None = Field(
        default=None,
    )
    # TODO: Initialization, lr_scale

    def _validate(self) -> None:
        pass

    def get_parameter(
        self,
        dims: tuple[TensorDim, ...],
        default_initializer: Initializer,
        lr_scale: float | None,
        weight_decay: bool = True,
        allow_sequence_tensor_parallel: bool = True,
        default_enabled: bool = False,
    ) -> "ParameterMeta|None":
        from fast_llm.tensor import ParameterMeta

        if (self.enabled is None and default_enabled) or self.enabled:
            return ParameterMeta.from_dims(
                dims,
                init_method=default_initializer,
                lr_scale=lr_scale,
                weight_decay=weight_decay,
                allow_sequence_tensor_parallel=allow_sequence_tensor_parallel,
            )
        else:
            return None
