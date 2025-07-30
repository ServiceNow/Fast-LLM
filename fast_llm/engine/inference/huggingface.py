import logging
import os
import pathlib
import typing

import torch
import transformers.generation.utils
import transformers.modeling_outputs

from fast_llm.engine.checkpoint.config import CheckpointLoadConfig, FastLLMCheckpointFormat
from fast_llm.engine.inference.config import HuggingfaceModelConfig
from fast_llm.engine.inference.runner import InferenceRunner
from fast_llm.engine.multi_stage.config import StageMode
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class HuggingfacePreTrainedModel(transformers.PreTrainedModel):
    config_class: typing.ClassVar[type[HuggingfaceModelConfig]] = HuggingfaceModelConfig
    runner_class: typing.ClassVar[type[InferenceRunner]] = InferenceRunner
    config: HuggingfaceModelConfig
    # base_model_prefix = ""
    # _no_split_modules = None
    # _supports_cache_class = False
    # _tied_weights_keys = []

    def __init__(
        self,
        fast_llm_model: FastLLMModel,
        config: HuggingfaceModelConfig | None = None,
        runner: ScheduleRunner | None = None,
        **kwargs,
    ):
        if config is None:
            config = self.config_class(fast_llm_model.config)

        assert self.runner_class.model_class.config_class is config.model_config_class
        assert config.fast_llm_config is fast_llm_model.config
        assert isinstance(config, self.config_class)

        # The HF constructor performs a deep copy of the config,
        # but config.fast_llm_config may contain non-picklable items like process groups.
        # Temporarily remove it before the call and restore it afterward.
        # TODO: Find a clean solution — overriding __deepcopy__ doesn't work here
        # because internally they use copy.deepcopy(self.__dict__).
        fast_llm_config = config.fast_llm_config
        config.fast_llm_config = None
        super().__init__(config, **kwargs)
        config.fast_llm_config = fast_llm_config

        self._inference_runner = self.runner_class(fast_llm_model, runner)

        # A model can be created from pretrained which set it up in the current HF wrapper api
        # or set existing model which  also must be setup, so, do not accept not setup model
        assert fast_llm_model.is_setup

        # We only support data parallel for now
        Assert.eq(fast_llm_model.distributed.config.model_parallel, 1)
        Assert.eq(fast_llm_model.distributed.config.sequence_data_parallel, 1)

        self._inference_runner.setup()

        # Transformers needs to be able to inspect the base model.
        self.fast_llm_base_model = fast_llm_model.base_model

        with transformers.modeling_utils.no_init_weights():
            self.post_init()

        if fast_llm_model.config.multi_stage.zero_stage == 3:
            logger.warning(
                "zero_stage=3 is used for the model; forward and generate will be extremely slow during inference."
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
        # always set up model and create distributed instance internally for now
        fast_llm_model = cls.runner_class.model_class.from_pretrained(
            pretrained_model_name_or_path,
            *updates,
            optimizer_state_names=optimizer_state_names,
            setup=True,
            mode=mode,
            use_cpu=use_cpu,
            stage_filter=stage_filter,
        )

        return cls(fast_llm_model, **kwargs)

    def _init_weights(self, module) -> None:
        raise NotImplementedError(module)


class HuggingfaceBaseModelForCausalLM(HuggingfacePreTrainedModel, transformers.generation.utils.GenerationMixin):
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
