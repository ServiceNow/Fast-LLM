import os
import pathlib
import typing

import transformers.modeling_outputs

from fast_llm.config import NoAutoValidate
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.huggingface.config import HuggingfaceModelConfig
from fast_llm.engine.multi_stage.config import CheckpointType, PretrainedCheckpointConfig, PretrainedConfig, StageMode
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.schedule.config import BatchConfig, ScheduleConfig
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.engine.schedule.schedule import Schedule


class HuggingfacePreTrainedModel(transformers.PreTrainedModel):
    config_class: typing.ClassVar[type[HuggingfaceModelConfig]] = HuggingfaceModelConfig
    model_class: typing.ClassVar[type[FastLLMModel]] = FastLLMModel
    config: HuggingfaceModelConfig
    # base_model_prefix = ""
    # _no_split_modules = None
    # _supports_cache_class = False
    # _tied_weights_keys = []

    def __init__(self, config: HuggingfaceModelConfig, fast_llm_model: FastLLMModel, **kwargs):
        assert self.model_class.config_class is config.model_config_class
        assert config.fast_llm_config is fast_llm_model.fast_llm_config
        assert isinstance(config, self.config_class)
        super().__init__(config, **kwargs)
        self._fast_llm_config = config.fast_llm_config
        self._fast_llm_model = fast_llm_model
        self._distributed_config = self._fast_llm_config.distributed
        # TODO: Support distributed models?
        assert self._distributed_config.world_size == 1
        self._schedule_config = ScheduleConfig()
        # We only need a basic schedule and don't care about dimensions.
        # TODO: Sort things out.
        with NoAutoValidate():
            self._batch_config = BatchConfig()
        self._batch_config.setup(self._distributed_config)
        self._batch_config.validate()
        self._runner = ScheduleRunner(
            multi_stage=self._fast_llm_model, config=self._schedule_config, distributed_config=self._distributed_config
        )
        self._runner.setup(self._fast_llm_model.distributed)
        # TODO: Random state? (Distributed.set_step)
        self._schedule = Schedule(
            multi_stage=self._fast_llm_model,
            batch_config=self._batch_config,
            schedule_config=self._schedule_config,
            distributed_config=self._distributed_config,
            phase=PhaseType.inference,
        )
        with transformers.modeling_utils.no_init_weights():
            self.post_init()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | PretrainedCheckpointConfig,
        *,
        mode: StageMode = StageMode.inference,
        **kwargs,
    ):
        # Pretrained config.
        if not isinstance(pretrained_model_name_or_path, PretrainedConfig):
            pretrained_model_name_or_path = PretrainedCheckpointConfig(
                path=pathlib.Path(pretrained_model_name_or_path),
                format=CheckpointType.state_dict,
            )

        config_updates = {}
        torch_dtype = kwargs.pop("torch_dtype", None)
        if torch_dtype is not None:
            config_updates[("distributed", "training_dtype")] = torch_dtype

        # Create the model
        fast_llm_model = cls.model_class.from_pretrained(
            pretrained_model_name_or_path, config_updates=config_updates, mode=mode
        )
        config = cls.config_class(fast_llm_model.fast_llm_config)

        return cls(config, fast_llm_model, **kwargs)

    def _init_weights(self, module):
        raise NotImplementedError(module)
