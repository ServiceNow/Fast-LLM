import logging
import typing

from fast_llm.core.distributed import broadcast
from fast_llm.engine.checkpoint.config import CHECKPOINT_VERSION, CheckpointLoadConfig, CheckpointSaveConfig
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.config import FastLLMModelConfig, StageMode
from fast_llm.engine.multi_stage.multi_stage import MultiStageModel
from fast_llm.functional.triton.pointwise import triton_fill
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class FastLLMModel(MultiStageModel):
    _is_loaded: bool = False

    def save_checkpoint(
        self,
        config: CheckpointSaveConfig,
        extra_metadata: dict | None = None,
    ):
        # TODO: Handle barriers, ok file, mkdir, etc. here
        num_shards = self.num_state_shards if config.optimizer_state else 1
        fast_llm_metadata = {
            "checkpoint_type": config.format.name,
            "checkpoint_version": str(CHECKPOINT_VERSION),
            "fast_llm_config": self._fast_llm_config.to_serialized(),
            "state_shard_names": list(self._state_shard_names[:num_shards]),
            "metadata": {} if extra_metadata is None else extra_metadata,
        }
        converter = config.format.get_handler_class()(self)
        converter.save(config, fast_llm_metadata)

    def load_checkpoint(self, config: CheckpointLoadConfig):
        # TODO: Simplify branching.
        # TODO: Test with more distributed configs.
        # TODO: Safety checks
        # TODO: Handle barriers, ok file, etc. here
        fast_llm_metadata = self.config_class.load_metadata(config)
        converter = config.format.get_handler_class()(self)
        converter.load(config, fast_llm_metadata)
        self._finalize_load(reset_optimizer=not config.optimizer_state)
        return fast_llm_metadata.get("metadata")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_config: CheckpointLoadConfig,
        default_config: FastLLMModelConfig = None,
        *,
        config_updates: dict[str | tuple[str, ...], typing.Any] | None = None,
        optimizer_state_names: tuple[str, ...] | None = None,
        setup: bool = True,
        mode: StageMode = StageMode.training,
        use_cpu: bool = False,
        stage_filter: set | None = None,
    ):
        metadata = cls.config_class.load_metadata(pretrained_config)
        config = cls.config_class.from_metadata(pretrained_config, metadata, default_config, config_updates)
        if mode.support_training:
            if "state_shard_names" in metadata:
                if optimizer_state_names is None:
                    optimizer_state_names = metadata["state_shard_names"][1:]
                else:
                    Assert.eq(optimizer_state_names, metadata["state_shard_names"][1:])
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

    def initialize_weights(self):
        assert self._is_setup
        for stage in self._stages:
            stage.initialize_weights()
        for name, tied_parameter in self._tied_parameters.items():
            if tied_parameter.group is not None:
                broadcast(self._stages[tied_parameter.main_stage].weight_shard, 0, tied_parameter.group)
        self._finalize_load(reset_optimizer=True)

    def _finalize_load(self, reset_optimizer: bool = True):
        if reset_optimizer:
            triton_fill(self._state_shard[1:], 0.0)
        if self._mode.support_forward:
            self.invalidate_buffers()
        self._is_loaded = True
