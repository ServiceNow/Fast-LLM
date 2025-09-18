import abc
import dataclasses
import typing

import torch
import torch.nn

from fast_llm.config import Configurable
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.tensor import ParameterMeta, TensorMeta
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    from fast_llm.engine.inference.runner import InferenceRunner


class Module(torch.nn.Module, abc.ABC):
    """ """

    _is_setup: bool = False
    _distributed: Distributed

    def __init__(self, distributed_config: DistributedConfig):
        self._distributed_config = distributed_config
        super().__init__()

    def setup(self, distributed: Distributed) -> None:
        assert not self._is_setup
        distributed.check_config(self._distributed_config)
        self._distributed = distributed
        self._is_setup = True


class Layer(Module):
    # Weight used to determine the stage size
    layer_count: float = 1.0

    @abc.abstractmethod
    def forward(
        self, input_: torch.Tensor, kwargs: dict, losses: dict | None = None, metrics: dict | None = None
    ) -> torch.Tensor:
        pass


class Sequential(Layer):
    def __init__(self, distributed_config: DistributedConfig):
        super().__init__(distributed_config)
        self.layers = torch.nn.ModuleList(self.get_layers())

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

    @abc.abstractmethod
    def get_layers(self) -> list[Layer]:
        pass

    def setup(self, distributed: Distributed) -> None:
        super().setup(distributed)
        for layer in self.layers:
            layer.setup(distributed)


@dataclasses.dataclass()
class LossDef:
    # A name for the loss
    name: str
    formatted_name: str
    # The number of times this loss is evaluated by the model for each micro-batch. Used as a denominator for averaging.
    # TODO: Allow variable count?  Would need a reduction across PP devices.
    count: int = 1
    dtype: torch.dtype = torch.float32


class BaseModel[ConfigType: BaseModelConfig](Configurable[ConfigType], Sequential):

    def __init__(
        self,
        config: BaseModelConfig,
        distributed_config: DistributedConfig,
    ):
        super().__init__(config, distributed_config)

        for key, value in self.named_parameters():
            Assert.custom(isinstance, value, ParameterMeta)
            # Rename to the parameter full name
            value.tensor_name = key

        # Reference models
        # TODO: Add basic handling (preprocessor) in this class.
        self._reference_models: dict[str, "InferenceRunner"] = {}

    @abc.abstractmethod
    def get_layers(self) -> list[Layer]:
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

    def add_reference_model(self, name: str, inference_runner: "InferenceRunner") -> None:
        assert name not in self._reference_models
        assert not self._is_setup
        self._reference_models[name] = inference_runner
