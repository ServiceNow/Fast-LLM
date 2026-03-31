import abc
import dataclasses
import enum
import typing

from fast_llm.config import MISSING, Config, Field, FieldHint, FieldVerboseLevel, config_class
from fast_llm.engine.config_utils.data_type import DataType
from fast_llm.engine.distributed.config import DistributedConfig
from fast_llm.utils import Assert, compare_nested, log

if typing.TYPE_CHECKING:
    import torch

    from fast_llm.engine.base_model.base_model import BaseModel
    from fast_llm.engine.distributed.distributed import Distributed


@config_class()
class ModuleConfig(Config):
    """
    Abstract config class for a base model.
    # TODO: Find better name?
    """

    _abstract = True

    def compare_architecture(
        self,
        model_config: typing.Self,
        log_fn: typing.Union[type[BaseException], typing.Callable] = ValueError,
    ):
        errors = compare_nested(self._get_architecture(), model_config._get_architecture())
        if errors:
            return log(
                f"Config comparison errors:\n  " + "\n".join(errors),
                log_fn=log_fn,
            )
        return None

    def _get_architecture(self) -> dict[str, typing.Any]:
        architecture = {}
        for name, field in self.fields():
            if not field.init or field._field_type == dataclasses._FIELD_CLASSVAR:
                continue
            assert isinstance(field, Field), f"{name}, {field}"
            if field.hint == FieldHint.architecture:
                architecture[name] = self._serialize_architecture_field(getattr(self, name, MISSING))
        return architecture

    def _serialize_architecture_field(self, value: typing.Any) -> typing.Any:
        if isinstance(value, ModuleConfig):
            # TODO: Make sure all nested configs have an architecture type hint?
            return value._get_architecture()
        elif isinstance(value, Config):
            # TODO: Explicitly prevent this case?
            return value.to_dict(verbose=FieldVerboseLevel.everything)
        elif isinstance(value, (list, tuple, set)):
            return [self._serialize_architecture_field(value_) for value_ in value]
        elif isinstance(value, dict):
            return {self._serialize_architecture_field(value_) for name, value_ in value.items()}
        else:
            return self._serialize_value(value)


def set_model_names(model: "torch.nn.Module"):
    from fast_llm.tensor import ParameterMeta

    for key, value in model.named_modules():
        value.module_name = key
    for key, value in model.named_parameters():
        Assert.custom(isinstance, value, ParameterMeta)
        # Rename to the parameter full name
        value.tensor_name = key


@config_class()
class BaseModelConfig(ModuleConfig):
    """
    Abstract config class for a base model.
    """

    def get_base_model(self, distributed_config: DistributedConfig) -> "BaseModel":

        model = self.base_model_class(self, distributed_config)
        # Storing the global name of each module and tensor.
        # Done here because it needs to run right after `model.__init__()`
        set_model_names(model)
        return model

    @property
    @abc.abstractmethod
    def base_model_class(self) -> type["BaseModel"]:
        pass


@dataclasses.dataclass
class ResourceUsageConfig:
    # Disable to get usage for current GPU only
    global_: bool = True
    # Enable to get hardware compute, i.e. include redundant computations.
    hardware: bool = False
    # Number of backward passes. Typically 1, may be 2 with full activation recomputation.
    forward: int = 1
    # Number of backward passes. Typically 1 for training, 0 for inference.
    backward: int = 1


class ReductionType(enum.StrEnum):
    """
    An enum to represent data types independently of third party libraries,
    so we can swap them more easily and allow for lazy imports.
    """

    sum = "float64"
    average = "float32"
    minimum = "float16"
    maximum = "bfloat16"

    @property
    def torch(self) -> "typing.Callable[[torch.Tensor], torch.Tensor]":
        if not _TORCH_REDUCTION_MAP:
            _set_torch_reduction_map()
        return _TORCH_REDUCTION_MAP[self]

    @property
    def distributed(self) -> "torch.distributed.ReduceOp.RedOpType":
        if not _DISTRIBUTED_REDUCTION_MAP:
            _set_distributed_reduction_map()
        return _DISTRIBUTED_REDUCTION_MAP[self]


_TORCH_REDUCTION_MAP: dict[ReductionType, "typing.Callable[[torch.Tensor], torch.Tensor]"] = {}


def _set_torch_reduction_map() -> None:
    import torch

    global _TORCH_REDUCTION_MAP

    _TORCH_REDUCTION_MAP = {
        ReductionType.sum: torch.sum,
        ReductionType.average: torch.mean,
        ReductionType.minimum: torch.min,
        ReductionType.maximum: torch.max,
    }


_DISTRIBUTED_REDUCTION_MAP: dict[ReductionType, "torch.distributed.ReduceOp.RedOpType"] = {}


def _set_distributed_reduction_map() -> None:
    import torch

    global _DISTRIBUTED_REDUCTION_MAP

    _DISTRIBUTED_REDUCTION_MAP = {
        ReductionType.sum: torch.distributed.ReduceOp.SUM,
        ReductionType.average: torch.distributed.ReduceOp.AVG,
        ReductionType.minimum: torch.distributed.ReduceOp.MIN,
        ReductionType.maximum: torch.distributed.ReduceOp.MAX,
    }


@dataclasses.dataclass()
class LossDef:
    # A name for the loss
    name: str
    dtype: DataType = DataType.float32
    reduction: ReductionType = ReductionType.sum

    def reduce(self, losses: "list[torch.Tensor]", distributed: "Distributed") -> "torch.Tensor | None":
        import torch

        from fast_llm.core.ops import reduce_op

        if losses or distributed.pipeline_group:
            if losses:
                reduced_loss = losses[0] if len(losses) == 1 else self.reduction.torch(torch.stack(losses))
                reduce_op(reduced_loss, group=distributed.data_group, op=self.reduction.distributed)
            else:
                reduced_loss = torch.zeros([1], dtype=self.dtype.torch, device=distributed.device)
            reduce_op(reduced_loss, group=distributed.pipeline_group, op=self.reduction.distributed)
            return reduced_loss
        else:
            return None
