import os
import pathlib
import typing

import transformers.modeling_outputs

from fast_llm.engine.checkpoint.config import CheckpointLoadConfig, FastLLMCheckpointFormat
from fast_llm.engine.inference.config import HuggingfaceModelConfig
from fast_llm.engine.inference.runner import InferenceRunner
from fast_llm.engine.multi_stage.config import StageMode
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel


class HuggingfacePreTrainedModel(transformers.PreTrainedModel):
    config_class: typing.ClassVar[type[HuggingfaceModelConfig]] = HuggingfaceModelConfig
    runner_class: typing.ClassVar[type[InferenceRunner]] = InferenceRunner
    config: HuggingfaceModelConfig
    # base_model_prefix = ""
    # _no_split_modules = None
    # _supports_cache_class = False
    # _tied_weights_keys = []

    def __init__(self, config: HuggingfaceModelConfig, fast_llm_model: FastLLMModel, **kwargs):
        assert self.runner_class.model_class.config_class is config.model_config_class
        assert config.fast_llm_config is fast_llm_model.config
        assert isinstance(config, self.config_class)

        super().__init__(config, **kwargs)

        self._inference_runner = self.runner_class(fast_llm_model)
        if not fast_llm_model.is_setup:
            fast_llm_model.setup(mode=StageMode.inference)
        self._inference_runner.setup()
        # Transformers needs to be able to inspect the base model.
        self.fast_llm_base_model = fast_llm_model.base_model
        # TODO: Support distributed models?
        assert fast_llm_model.config.distributed.world_size == 1

        with transformers.modeling_utils.no_init_weights():
            self.post_init()

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

        # Create the model
        fast_llm_model = cls.runner_class.model_class.from_pretrained(
            pretrained_model_name_or_path, config_updates=config_updates, mode=mode
        )
        config = cls.config_class(fast_llm_model.config)

        return cls(config, fast_llm_model, **kwargs)

    def _init_weights(self, module) -> None:
        raise NotImplementedError(module)
