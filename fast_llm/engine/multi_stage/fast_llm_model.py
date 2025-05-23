import logging
import typing

from fast_llm.config import UpdateType
from fast_llm.core.distributed import broadcast
from fast_llm.engine.checkpoint.config import CheckpointLoadConfig, CheckpointSaveConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, StageMode
from fast_llm.engine.multi_stage.multi_stage import MultiStageModel
from fast_llm.functional.triton.pointwise import triton_fill
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class FastLLMModel[ConfigType: FastLLMModelConfig](MultiStageModel[ConfigType]):
    config_class: typing.ClassVar[type[FastLLMModelConfig]] = FastLLMModelConfig
    _is_loaded: bool = False

    def save_checkpoint(
        self,
        config: CheckpointSaveConfig,
        extra_metadata: dict | None = None,
    ) -> None:
        # TODO: Handle barriers, ok file, mkdir, etc. here
        converter = config.format.get_handler_class()(self)
        fast_llm_metadata = self._config.to_metadata(
            config,
            shards=converter.get_shard_names(config),
            metadata={} if extra_metadata is None else extra_metadata,
        )
        converter.save(config, fast_llm_metadata)

    def load_checkpoint(self, config: CheckpointLoadConfig) -> dict[str, typing.Any] | None:
        # TODO: Simplify branching.
        # TODO: Test with more distributed configs.
        # TODO: Safety checks
        # TODO: Handle barriers, ok file, etc. here
        converter = config.format.get_handler_class()(self)
        metadata = converter.load(config)
        self._finalize_load(reset_optimizer=not config.optimizer_state)
        return metadata

    @classmethod
    def from_pretrained(
        cls,
        pretrained_config: CheckpointLoadConfig,
        *updates: dict[str | tuple[str, ...], typing.Any],
        optimizer_state_names: tuple[str, ...] | None = None,
        setup: bool = True,
        mode: StageMode = StageMode.training,
        use_cpu: bool = False,
        stage_filter: set | None = None,
    ) -> typing.Self:
        metadata = cls.config_class.load_metadata(pretrained_config)
        config = cls.config_class.from_dict(metadata.config, *updates, update_type=UpdateType.update)
        if mode.support_training:
            # TODO v0.3: Make metadata.shards mandatory?
            if metadata.shards:
                if optimizer_state_names is None:
                    optimizer_state_names = metadata.shards[1:]
                else:
                    Assert.eq(optimizer_state_names, metadata.shards[1:])
            elif optimizer_state_names is None:
                raise ValueError("`optimizer_state_names` is required")
        else:
            assert optimizer_state_names is None
            optimizer_state_names = ()

        model = cls(
            config,
            optimizer_state_names=tuple(optimizer_state_names),
            stage_filter=stage_filter,
        )

        if setup:
            model.setup(Distributed(config.distributed, use_cpu=use_cpu), mode=mode)

            if mode.on_device:
                if pretrained_config.model_weights:
                    model.load_checkpoint(pretrained_config)
                else:
                    model.initialize_weights()
        return model

    def initialize_weights(self, timeout: float | None = None) -> None:
        assert self._is_setup
        for stage in self._stages:
            stage.initialize_weights()
        for name, tied_parameter in self._tied_parameters.items():
            if tied_parameter.group is not None:
                for fsdp in self._stages[tied_parameter.main_stage].fsdps:
                    broadcast(fsdp.weight_shard, 0, tied_parameter.group, timeout=timeout)
        self._finalize_load(reset_optimizer=True)

    def _finalize_load(self, reset_optimizer: bool = True) -> None:
        if reset_optimizer:
            triton_fill(self._flat_shard[self._weight_shard_size :], 0.0)
        if self._mode.support_forward:
            self.invalidate_buffers()
        self._is_loaded = True
