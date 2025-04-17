import os
import pathlib
import typing

import torch
import transformers.modeling_outputs
import transformers.generation.utils

from fast_llm.config import NoAutoValidate
from fast_llm.engine.checkpoint.config import CheckpointLoadConfig, FastLLMCheckpointFormat
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.huggingface.config import HuggingfaceModelConfig
from fast_llm.engine.multi_stage.config import StageMode
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.schedule.config import BatchConfig, ScheduleConfig
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.engine.schedule.schedule import Schedule


class HuggingfaceBaseModelForCausalLM(transformers.PreTrainedModel, transformers.generation.utils.GenerationMixin):
    config_class: typing.ClassVar[type[HuggingfaceModelConfig]] = HuggingfaceModelConfig
    model_class: typing.ClassVar[type[FastLLMModel]] = FastLLMModel
    config: HuggingfaceModelConfig
    # base_model_prefix = ""
    # _no_split_modules = None
    # _supports_cache_class = False
    # _tied_weights_keys = []

    def __init__(self, config: HuggingfaceModelConfig, fast_llm_model: FastLLMModel, **kwargs):
        assert self.model_class.config_class is config.model_config_class
        assert config.fast_llm_config is fast_llm_model.config
        assert isinstance(config, self.config_class)
        super().__init__(config, **kwargs)
        self._fast_llm_config = config.fast_llm_config
        self._fast_llm_model = fast_llm_model
        # Transformers needs to be able to inspect the base model.
        self.fast_llm_base_model = self._fast_llm_model.base_model
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
            config=self._schedule_config, multi_stage=self._fast_llm_model, distributed_config=self._distributed_config
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

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | transformers.modeling_outputs.CausalLMOutputWithPast:
        # Meant to be overriden in derived classes
        raise NotImplementedError()

    @classmethod
    def from_fast_llm_model(cls, fast_llm_model: FastLLMModel, **kwargs):
        config = cls.config_class(fast_llm_model.config)
        return cls(config, fast_llm_model, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | CheckpointLoadConfig,
        *,
        mode: StageMode = StageMode.inference,
        **kwargs,
    ) -> typing.Self:
        # Pretrained config.
        if not isinstance(pretrained_model_name_or_path, CheckpointLoadConfig):
            pretrained_model_name_or_path = CheckpointLoadConfig(
                path=pathlib.Path(pretrained_model_name_or_path),
                format=FastLLMCheckpointFormat,
            )

        config_updates = {}
        torch_dtype = kwargs.pop("torch_dtype", None)
        if torch_dtype is not None:
            config_updates[("distributed", "training_dtype")] = torch_dtype

        attn_implementation = kwargs.pop("attn_implementation", None)
        if attn_implementation is not None:
            if attn_implementation == "flash_attention_2":
                config_updates[("base_model", "transformer", "use_flash_attention")] = True
            else:
                config_updates[("base_model", "transformer", "use_flash_attention")] = False

        # Create the model
        fast_llm_model = cls.model_class.from_pretrained(
            pretrained_model_name_or_path, config_updates=config_updates, mode=mode
        )

        return cls.from_fast_llm_model(fast_llm_model, **kwargs)

    def _init_weights(self, module) -> None:
        raise NotImplementedError(module)

    def can_generate(self):
        return True
