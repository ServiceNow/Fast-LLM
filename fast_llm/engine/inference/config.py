import copy
import logging
import os
import pathlib
import typing

import transformers

from fast_llm.config import FieldVerboseLevel
from fast_llm.engine.checkpoint.config import CheckpointLoadMetadataConfig, FastLLMCheckpointFormat
from fast_llm.engine.multi_stage.config import FastLLMModelConfig

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
            assert self.torch_dtype == self.fast_llm_config.distributed.compute_dtype.torch

    def save_pretrained(self, save_directory: str | os.PathLike, push_to_hub: bool = False, **kwargs) -> None:
        # Hack the method to save at the right place.
        # TODO: Implement the rest.
        _backup = transformers.configuration_utils.CONFIG_NAME
        try:
            transformers.configuration_utils.CONFIG_NAME = "metadata.yaml"
            super().save_pretrained(save_directory, push_to_hub, **kwargs)
        finally:
            transformers.configuration_utils.CONFIG_NAME = _backup

    def __deepcopy__(self, memo):
        # Hugging Face's PretrainedModel will deep copy the config
        # when `generate` is enabled. However, `fast_llm_config`
        # cannot be deep copied if the world size is greater than 1,
        # as it will contain references to process groups.
        # Therefore, we copy it by reference instead.
        cls = self.__class__
        copied = cls.__new__(cls)
        memo[id(self)] = copied
        for k, v in self.__dict__.items():
            if k == "fast_llm_config":
                setattr(copied, k, v)  # Keep the same reference
            else:
                setattr(copied, k, copy.deepcopy(v, memo))
        return copied

    @classmethod
    def _get_config_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike | CheckpointLoadMetadataConfig, **kwargs
    ):
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
            assert isinstance(kwargs["pretrained"], CheckpointLoadMetadataConfig)
            assert kwargs["pretrained"].path == pretrained_model_name_or_path
            pretrained = kwargs.pop("pretrained")
        elif isinstance(pretrained_model_name_or_path, CheckpointLoadMetadataConfig):
            pretrained = pretrained_model_name_or_path
        else:
            pretrained = CheckpointLoadMetadataConfig(
                path=pathlib.Path(pretrained_model_name_or_path),
                format=FastLLMCheckpointFormat,
            )
        metadata = cls.model_config_class.load_metadata(pretrained)
        updates = {}
        torch_dtype = kwargs.pop("torch_dtype", None)
        if torch_dtype is not None:
            updates[("distributed", "compute_dtype")] = torch_dtype
        fast_llm_config = cls.model_config_class.from_metadata(
            pretrained, metadata, default=kwargs.pop("fast_llm_config", None), updates=updates
        )

        config_dict = {"fast_llm_config": fast_llm_config}
        return config_dict, kwargs

    @classmethod
    def from_json_file(cls, json_file: str | os.PathLike) -> typing.Self:
        raise NotImplementedError()

    def __eq__(self, other) -> bool:
        # Not sure if elementwise equality is enough, taking a strict approach.
        return other is self

    def to_dict(self) -> dict[str, typing.Any]:
        out = super().to_dict()
        if self.fast_llm_config is not None:
            out["fast_llm_config"] = self.fast_llm_config.to_dict(verbose=FieldVerboseLevel.everything)
        return out

    def to_diff_dict(self) -> dict[str, typing.Any]:
        out = super().to_diff_dict()
        out["fast_llm_config"] = self.fast_llm_config.to_dict(verbose=FieldVerboseLevel.explicit)
        return out

    def to_json_file(self, json_file_path: str | os.PathLike, use_diff: bool = True) -> None:
        raise NotImplementedError()

    def update(self, config_dict: dict):
        raise NotImplementedError()

    def update_from_string(self, update_str: str):
        raise NotImplementedError()
