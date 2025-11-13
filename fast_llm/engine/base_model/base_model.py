import abc
import functools
import typing

import torch.nn

from fast_llm.config import Configurable
from fast_llm.engine.base_model.config import BaseModelConfig, LossDef, ResourceUsageConfig
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.tensor import ParameterMeta, TensorMeta

if typing.TYPE_CHECKING:
    from fast_llm.engine.inference.runner import InferenceRunner


class LayerBase(torch.nn.Module, abc.ABC):
    _is_setup: bool = False
    _distributed: Distributed

    def __init__(self, distributed_config: DistributedConfig):
        self._distributed_config = distributed_config
        super().__init__()

    def setup(self, distributed: Distributed) -> None:
        assert not self._is_setup
        for layer in self.get_layers():
            if layer is not self:
                layer.setup(distributed)
        distributed.check_config(self._distributed_config)
        self._distributed = distributed
        self._is_setup = True

    @abc.abstractmethod
    def get_layers(self) -> list["Layer"]:
        """
        The list of layers as meant to be seen by the Fast-LLM engine.
        May differ from the module configuration seen by pytorch.
        """

    def get_compute_usage(self, input_: TensorMeta, kwargs: dict[str, typing.Any], config: ResourceUsageConfig) -> int:
        out = 0
        for layer in self.get_layers():
            if layer is self:
                raise NotImplementedError()
            out += layer.get_compute_usage(input_, kwargs, config)
        return out

    def get_loss_definitions(self, count: int = 1) -> list[LossDef]:
        losses = []
        for layer in self.get_layers():
            if layer is not self:
                losses += layer.get_loss_definitions(count)
        return losses

    def preprocess(self, kwargs: dict[str, typing.Any]) -> None:
        for layer in self.get_layers():
            if layer is not self:
                layer.preprocess(kwargs)

    def unwrap(self) -> "LayerBase":
        # Get the actual module contained in this layer,
        # undoing any wrapping for the Fast-LLM engine (ex. `LayerBaseWithNamespace`)
        return self


class Layer(LayerBase):
    # Weight used to determine the stage size.
    layer_count: float = 1.0

    def get_layers(self) -> list["Layer"]:
        # Return a breakdown of the layer into atomic ones,
        # i.e. the list of layers from as seen from the Fast-LLM model.
        return [self]

    @abc.abstractmethod
    def forward(
        self, input_: torch.Tensor, kwargs: dict, losses: dict | None = None, metrics: dict | None = None
    ) -> torch.Tensor:
        pass

    def unwrap(self) -> "Layer":
        return self


class LayerBaseWithNamespace(LayerBase):
    """
    A layer base with its own namespace for preprocessing (kwargs),
     so that it doesn't inadvertently interact with other layers.
    TODO: Consider namespace for losses and metrics?
    """

    def __init__(self, layer: LayerBase, namespace: str = None):
        super().__init__(layer._distributed_config)
        self._layer = layer
        self._namespace = namespace
        self.get_compute_usage = self._layer.get_compute_usage
        self.module_name = self._layer.module_name

    def setup(self, distributed: Distributed) -> None:
        self._layer.setup(distributed)
        super().setup(distributed)

    def get_layers(self) -> list["Layer"]:
        """
        Wrap individual layers so the namespace is used in forward.
        """
        return self._layers_with_namespace

    def preprocess(self, kwargs: dict[str, typing.Any]) -> None:
        """
        Preprocess with namespace.
        """
        if self._namespace not in kwargs:
            kwargs[self._namespace] = kwargs.copy()
        self._layer.preprocess(kwargs[self._namespace])

    def unwrap(self) -> "LayerBase":
        return self._layer.unwrap()

    @functools.cached_property
    def _layers_with_namespace(self) -> list[Layer]:
        # This needs to be in a property because `module_name` is set after `__init__`.
        # Wrap each set of blocks with identical config in a namespace
        # using the unique module name of the first such block.
        return [LayerWithNamespace(layer, self._namespace) for layer in self._layer.get_layers()]


class LayerWithNamespace(LayerBaseWithNamespace, Layer):
    _layer: Layer

    def __init__(self, layer: Layer, namespace: str = None):
        super().__init__(layer, namespace)
        self.layer_count = self._layer.layer_count

    def get_layers(self) -> list["Layer"]:
        # Need to override since `LayerBaseWithNamespace.get_layers` comes first in the MRO.
        return [self]

    def forward(
        self, input_: torch.Tensor, kwargs: dict, losses: dict | None = None, metrics: dict | None = None
    ) -> torch.Tensor:
        if self._namespace in kwargs:
            kwargs = kwargs[self._namespace]
        else:
            # TODO: Forward meta doesn't go through preprocessing so doesn't have a namespace.
            #   Using kwargs as-is since it's generally unused.
            assert isinstance(input_, TensorMeta)
        return self._layer.forward(input_, kwargs, losses, metrics)

    def unwrap(self) -> "Layer":
        return self._layer.unwrap()


class BaseModel[ConfigType: BaseModelConfig](Configurable[ConfigType], LayerBase):

    def __init__(
        self,
        config: BaseModelConfig,
        distributed_config: DistributedConfig,
    ):
        super().__init__(config, distributed_config)

        # Reference models
        # TODO: Add basic handling (preprocessor) in this class.
        self._reference_models: dict[str, "InferenceRunner"] = {}

    @abc.abstractmethod
    def preprocess_meta(self, batch_meta: typing.Any, phase: PhaseType) -> list[tuple[TensorMeta, dict]]:
        # TODO Remove (Move batch splitting elsewhere)
        pass

    @abc.abstractmethod
    def preprocess_batch(
        self,
        batch: typing.Any,
        preprocessed_meta: list[tuple[TensorMeta, dict]] | None = None,
        *,
        phase: PhaseType,
        iteration: int,
        metrics: dict | None = None,
    ) -> list[tuple[torch.Tensor, dict]]:
        # TODO Move batch splitting elsewhere, align interface with LayerBase
        pass

    def get_tied_parameters(self) -> dict[str, list[ParameterMeta]]:
        """
        Return tuples of independently defined metas to tie together.
        Metas should be compatible, i.e. have the same tensor dimensions.
        Tied weights are named (dict keys) for convenience only.
        Warning: Initialization and optimization properties are defined on the first appearance of the tied weight.
          To prevent any confusion, the metas should be provided in the same order they appear in the model.
          TODO: Improve?
        Note: This may return buffers instead of metas after stage setup.
        """
        return {}

    def add_reference_model(self, name: str, inference_runner: "InferenceRunner") -> None:
        assert name not in self._reference_models
        assert not self._is_setup
        self._reference_models[name] = inference_runner
