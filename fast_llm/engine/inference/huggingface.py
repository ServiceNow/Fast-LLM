import os
import pathlib
import typing

import torch
import transformers.modeling_outputs
import transformers.generation.utils

from fast_llm.engine.checkpoint.config import CheckpointLoadConfig, FastLLMCheckpointFormat
from fast_llm.engine.inference.config import HuggingfaceModelConfig
from fast_llm.engine.inference.runner import InferenceRunner
from fast_llm.engine.multi_stage.config import StageMode
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.schedule.config import BatchConfig, ScheduleConfig
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.engine.schedule.schedule import Schedule
from fast_llm.engine.training.config import TrainerConfig


class HuggingfaceBaseModelForCausalLM(transformers.PreTrainedModel, transformers.generation.utils.GenerationMixin):
    config_class: typing.ClassVar[type[HuggingfaceModelConfig]] = HuggingfaceModelConfig
    runner_class: typing.ClassVar[type[InferenceRunner]] = InferenceRunner
    config: HuggingfaceModelConfig
    # base_model_prefix = ""
    # _no_split_modules = None
    # _supports_cache_class = False
    # _tied_weights_keys = []

    def __init__(
        self,
        config: HuggingfaceModelConfig,
        fast_llm_model: FastLLMModel,
        micro_batch_size: int | None = None,
        runner: ScheduleRunner | None = None,
        **kwargs,
    ):
        assert self.runner_class.model_class.config_class is config.model_config_class
        assert config.fast_llm_config is fast_llm_model.config
        assert isinstance(config, self.config_class)

        # The HF constructor performs a deep copy of the config,
        # but config.fast_llm_config may contain non-picklable items like process groups.
        # Temporarily remove it before the call and restore it afterward.
        fast_llm_config = config.fast_llm_config
        config.fast_llm_config = None
        super().__init__(config, **kwargs)
        config.fast_llm_config = fast_llm_config

        self._inference_runner = self.runner_class(fast_llm_model, micro_batch_size, runner)

        # A model can be created from pretrained which setup it in the current HF wrapper api
        # or set from training loop and also is setup, so, do not accept not setup model
        assert fast_llm_model.is_setup
        # if not fast_llm_model.is_setup:
        #   fast_llm_model.setup(distributed=distributed, mode=StageMode.inference)
        self._inference_runner.setup()

        # Transformers needs to be able to inspect the base model.
        self.fast_llm_base_model = fast_llm_model.base_model
        # # TODO: Support distributed models?
        # assert fast_llm_model.config.distributed.world_size == 1

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
        # Meant to be overridden in derived classes
        raise NotImplementedError()

    @classmethod
    def from_model(
        cls,
        fast_llm_model: FastLLMModel,
        micro_batch_size: int | None = None,
        runner: ScheduleRunner | None = None,
        **kwargs,
    ):
        config = cls.config_class(fast_llm_model.config)
        return cls(
            config,
            fast_llm_model,
            micro_batch_size=micro_batch_size,
            runner=runner,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | CheckpointLoadConfig,
        *updates: dict[str | tuple[str, ...], typing.Any],
        optimizer_state_names: tuple[str, ...] | None = None,
        # setup: bool = True,
        mode: StageMode = StageMode.training,
        use_cpu: bool = False,
        stage_filter: set | None = None,
        **kwargs,
    ) -> typing.Self:
        # Pretrained config.
        if not isinstance(pretrained_model_name_or_path, CheckpointLoadConfig):
            pretrained_model_name_or_path = CheckpointLoadConfig(
                path=pathlib.Path(pretrained_model_name_or_path),
                format=FastLLMCheckpointFormat,
            )

        # Create the model
        # always set up model and crate distributed instance internally for now
        fast_llm_model = cls.runner_class.model_class.from_pretrained(
            pretrained_model_name_or_path,
            *updates,
            optimizer_state_names=optimizer_state_names,
            # setup=setup,
            mode=mode,
            use_cpu=use_cpu,
            stage_filter=stage_filter,
        )

        config = cls.config_class(fast_llm_model.config)
        return cls(config, fast_llm_model, **kwargs)

    def _init_weights(self, module) -> None:
        raise NotImplementedError(module)
