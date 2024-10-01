import logging
import os
import pathlib
import typing

import transformers

from fast_llm.engine.multi_stage.config import CheckpointType, FastLLMModelConfig, PretrainedConfig

logger = logging.getLogger(__name__)


class HuggingfaceModelConfig(transformers.PretrainedConfig):
    model_type = "fast_llm"
    model_config_class: typing.ClassVar[type[FastLLMModelConfig]] = FastLLMModelConfig

    def __init__(self, fast_llm_config: FastLLMModelConfig | None = None, **kwargs):
        # Needed for `to_diff_dict` (`__repr__`)
        if fast_llm_config is None:
            fast_llm_config = self.model_config_class()
        self.fast_llm_config = fast_llm_config
        self.use_cache = kwargs.pop("use_cache", True)
        super().__init__(**kwargs)
        if self.torch_dtype is not None:
            assert self.torch_dtype == self.fast_llm_config.distributed.training_dtype.torch

    def save_pretrained(self, save_directory: str | os.PathLike, push_to_hub: bool = False, **kwargs):
        # Hack the method to save at the right place.
        # TODO: Implement the rest.
        _backup = transformers.configuration_utils.CONFIG_NAME
        try:
            transformers.configuration_utils.CONFIG_NAME = "metadata.yaml"
            super().save_pretrained(save_directory, push_to_hub, **kwargs)
        finally:
            transformers.configuration_utils.CONFIG_NAME = _backup

    @classmethod
    def _get_config_dict(cls, pretrained_model_name_or_path: str | os.PathLike | PretrainedConfig, **kwargs):
        # TODO: Support download from hub/url

        # Unused arguments, remove to avoid warnings.
        kwargs.pop("cache_dir", None)
        kwargs.pop("force_download", False)
        kwargs.pop("resume_download", False)
        kwargs.pop("proxies", None)
        kwargs.pop("token", None)
        kwargs.pop("local_files_only", False)
        kwargs.pop("revision", None)
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("subfolder", "")
        kwargs.pop("_from_pipeline", None)
        kwargs.pop("_from_auto", False)
        kwargs.pop("_commit_hash", None)
        kwargs.get("gguf_file", None)

        # Get the pretrained config.
        if "pretrained" in kwargs:
            assert isinstance(kwargs["pretrained"], PretrainedConfig)
            assert kwargs["pretrained"].pretrained_checkpoint_path == pretrained_model_name_or_path
            pretrained = kwargs.pop("pretrained")
        elif isinstance(pretrained_model_name_or_path, PretrainedConfig):
            pretrained = pretrained_model_name_or_path
        else:
            pretrained = PretrainedConfig(
                pretrained_checkpoint_path=pathlib.Path(pretrained_model_name_or_path),
                pretrained_checkpoint_type=CheckpointType.state_dict,
            )
        metadata = cls.model_config_class.load_pretrained_metadata(pretrained)
        updates = {}
        torch_dtype = kwargs.pop("torch_dtype", None)
        if torch_dtype is not None:
            updates[("distributed", "training_dtype")] = torch_dtype
        fast_llm_config = cls.model_config_class.from_metadata(
            pretrained, metadata, default=kwargs.pop("fast_llm_config", None), updates=updates
        )

        config_dict = {"fast_llm_config": fast_llm_config}
        if "huggingface_config" in metadata:
            assert "fast_llm_config" not in metadata["huggingface_config"]
            config_dict.update(metadata.pop("huggingface_config"))

        return config_dict, kwargs

    @classmethod
    def from_json_file(cls, json_file: str | os.PathLike):
        raise NotImplementedError()

    def __eq__(self, other):
        # Not sure if elementwise equality is enough, taking a strict approach.
        return other is self

    def to_dict(self) -> dict:
        out = super().to_dict()
        out["fast_llm_config"] = self.fast_llm_config.to_serialized(verbose=None)
        return out

    def to_diff_dict(self) -> dict:
        out = super().to_diff_dict()
        out["fast_llm_config"] = self.fast_llm_config.to_serialized()
        return out

    def to_json_file(self, json_file_path: str | os.PathLike, use_diff: bool = True):
        raise NotImplementedError()

    def update(self, config_dict: dict):
        raise NotImplementedError()

    def update_from_string(self, update_str: str):
        raise NotImplementedError()
