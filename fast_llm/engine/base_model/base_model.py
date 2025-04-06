import abc
import dataclasses
import typing

import torch
import torch.nn

from fast_llm.config import Configurable
from fast_llm.engine.base_model.config import BaseModelArchitectureConfig, BaseModelConfig, Preprocessor
from fast_llm.engine.config_utils.tensor_space import TensorSpace
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.tensor import ParameterMeta, TensorMeta
from fast_llm.utils import Assert


class Module(torch.nn.Module, abc.ABC):
    """ """

    def forward(self, input_, kwargs):
        """
        Run a forward pass for the module, with autograd support.
        """
        raise NotImplementedError()


class Layer(Module):
    # Weight used to determine the stage size
    layer_count: float = 1.0

    @abc.abstractmethod
    def forward(
        self, input_: torch.Tensor, kwargs: dict, losses: dict | None = None, metrics: dict | None = None
    ) -> torch.Tensor:
        pass


class Sequential(Layer):
    def __init__(self, layers: list[Layer]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def __getitem__(self, item):
        return self.layers[item]

    def __iter__(self):
        return iter(self.layers)

    def __len__(self):
        return len(self.layers)

    def forward(
        self, input_: torch.Tensor, kwargs: dict, losses: dict | None = None, metrics: dict | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            input_ = layer(input_, kwargs, losses, metrics)
        return input_


@dataclasses.dataclass()
class LossDef:
    # A name for the loss
    name: str
    formatted_name: str
    # The number of times this loss is evaluated by the model for each micro-batch. Used as a denominator for averaging.
    # TODO: Allow variable count?  Would need a reduction across PP devices.
    count: int = 1
    dtype: torch.dtype = torch.float32


class SequentialLayers(Sequential, abc.ABC):
    # Small class defined to fix the MRO of BaseModel.__init__
    def __init__(self):
        super().__init__(self.get_layers())

    @abc.abstractmethod
    def get_layers(self) -> list[Layer]:
        pass


class BaseModel[ConfigType: BaseModelConfig](Configurable[ConfigType], SequentialLayers, abc.ABC):
    config_class: typing.ClassVar[type[BaseModelConfig]] = BaseModelConfig

    def __init__(
        self,
        config: BaseModelConfig,
        distributed_config: DistributedConfig,
    ):
        self._tensor_space = TensorSpace(distributed_config)
        config.setup_tensor_space(self._tensor_space)

        super().__init__(config)

        for key, value in self.named_parameters():
            Assert.custom(isinstance, value, ParameterMeta)
            # Rename to the parameter full name
            value.tensor_name = key

    @classmethod
    def architecture_cls(cls) -> type[BaseModelArchitectureConfig]:
        return cls.config_class.architecture_class

    @abc.abstractmethod
    def get_layers(self) -> list[Layer]:
        pass

    @abc.abstractmethod
    def setup(self, distributed: Distributed) -> None:
        pass

    @abc.abstractmethod
    def preprocess_meta(self, batch_meta: typing.Any, phase: PhaseType) -> list[tuple[TensorMeta, dict]]:
        pass

    @abc.abstractmethod
    def preprocess(
        self,
        batch: typing.Any,
        preprocessed_meta: list[tuple[TensorMeta, dict]] | None = None,
        *,
        phase: PhaseType,
        iteration: int,
        metrics: dict | None = None,
    ) -> list[tuple[torch.Tensor, dict]]:
        pass

    def get_tied_weights(self) -> dict[str, tuple[ParameterMeta, tuple[int, ...]]]:
        # For each tied weight, return the weight and the tuple of layers sharing it.
        # The weight should be defined in the first layer in the set.
        # Warning: This may return buffers instead of metas after stage setup.
        # The name (dict key) is used to insert the weight in the kwargs of the forward pass.
        return {}

    @property
    @abc.abstractmethod
    def loss_defs(self) -> list[LossDef]:
        pass

    def add_preprocessor(self, preprocessor: Preprocessor):
        # TODO: Generalize preprocessors.
        raise NotImplementedError()
