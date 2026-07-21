import abc
import functools
import json
import pathlib
import shutil
import typing

import safetensors
import torch

from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.checkpoint.config import CheckpointLoadConfig, CheckpointSaveConfig, CheckpointSaveMetadataConfig
from fast_llm.engine.checkpoint.external import (
    ConfigSectionConverter,
    ExternalStateDictCheckpointHandler,
    WeightConverter,
    logger,
)
from fast_llm.engine.multi_stage.config import CheckpointMetadata, FastLLMModelConfig
from fast_llm.tensor import SafeTensorSlice
from fast_llm.utils import Assert, safe_merge_dicts

if typing.TYPE_CHECKING:
    import transformers


class HuggingFaceBaseModelConverter(ConfigSectionConverter):
    """Section converter for a full HF model root. Inherits the declarative config-side machinery from
    :class:`ConfigSectionConverter` (``import_config`` / ``export_config`` driven by
    ``_create_config_converters``) and adds the weight-side ``get_converters`` aggregation that the
    enclosing :class:`HuggingfaceStateDictCheckpointHandler` invokes.
    """

    @classmethod
    def get_converters(cls, config: BaseModelConfig) -> list[WeightConverter]:
        """Default: walk the section's weight declarations from the root.

        Subclasses override when a section needs cross-section state from the full base-model config
        (typically when an extension point on the head must read from a sibling section).
        """
        return cls.emit_weight_converters(config, "", "")


class HuggingfaceStateDictCheckpointHandler(ExternalStateDictCheckpointHandler, abc.ABC):
    architecture: typing.ClassVar[str]
    base_model_converter_class: typing.ClassVar[type[HuggingFaceBaseModelConverter]]

    @classmethod
    @abc.abstractmethod
    def get_transformers_configuration_class(cls) -> type["transformers.PretrainedConfig"]:
        pass

    @classmethod
    def get_model_files(cls) -> tuple[str | None, str | None, str | None]:
        return None, None, None

    @classmethod
    def _save_serialized_metadata(cls, config: CheckpointSaveMetadataConfig, metadata: dict, index: dict) -> None:
        config.path.mkdir(parents=True, exist_ok=True)
        path = config.path / f"{cls.base_file_name}.safetensors.index.json"
        logger.info(f"Saving index to {path}")
        # Save the index.
        path.write_text(json.dumps({"metadata": metadata, "weight_map": index}, indent=4))

    def _serialize_metadata(self, config: CheckpointSaveMetadataConfig, metadata: CheckpointMetadata) -> dict:
        huggingface_config = self._export_config(self._model.config)
        self._save_config(config.path, huggingface_config)
        return {
            "fast_llm_metadata": metadata.to_dict(),
            "model_config": huggingface_config,
            "format": "pt",
        }

    def load(self, config: CheckpointLoadConfig) -> dict[str, typing.Any] | None:
        assert not config.optimizer_state
        metadata = self._model.config.load_metadata(config)
        self._model.config.base_model.compare_architecture(metadata.config.base_model, logger.warning)
        super().load(config)

    def save(self, config: CheckpointSaveConfig, metadata: CheckpointMetadata) -> None:
        super().save(config, metadata)
        # Copy the modeling files to the output directory
        modeling_file, configuration_file, generation_utils_file = self.get_model_files()
        if configuration_file is not None:
            shutil.copy(configuration_file, config.path)
        if modeling_file is not None:
            shutil.copy(modeling_file, config.path)
        if generation_utils_file is not None:
            shutil.copy(generation_utils_file, config.path)
            gen_config = pathlib.Path(generation_utils_file).parent / "generation_config.json"
            if gen_config.exists():
                shutil.copy(gen_config, config.path)

    @classmethod
    def get_huggingface_model_type(self) -> str:
        # We assume the two names match, but derived classes can make it different.
        return self.format.name

    @classmethod
    def _get_key(cls, parameter_name: str, shard_name: str) -> str:
        Assert.eq(shard_name, "weights")
        return parameter_name

    @classmethod
    def _resolve_path(cls, path: pathlib.Path) -> pathlib.Path:
        """Resolve a local directory or HF Hub model id (e.g. ``meta-llama/Llama-3.2-1B``) to a
        local snapshot directory. Local directories pass through unchanged; everything else is
        materialized via :func:`huggingface_hub.snapshot_download` (cached on subsequent calls).
        """
        if path.is_dir():
            return path
        import huggingface_hub

        return pathlib.Path(huggingface_hub.snapshot_download(str(path)))

    # Use custom config instead of relying on the transformers library
    @classmethod
    def _load_config(cls, directory: pathlib.Path | str) -> dict:
        config = cls.get_transformers_configuration_class().from_pretrained(directory).to_dict()
        Assert.eq(config["model_type"], cls.get_huggingface_model_type())
        return config

    @classmethod
    def _save_config(cls, directory: pathlib.Path | str, config: dict[str, typing.Any]) -> None:
        cls.get_transformers_configuration_class().from_dict(config).save_pretrained(directory)

    @classmethod
    def _export_config(cls, config: FastLLMModelConfig) -> dict[str, typing.Any]:
        return safe_merge_dicts(
            cls.base_model_converter_class.export_config(config.base_model),
            {
                "model_type": cls.get_huggingface_model_type(),
                "architectures": [cls.architecture],
            },
        )

    # HF metadata keys that are always permitted, regardless of the converter tree. The generic
    # ``PretrainedConfig`` fields are added dynamically (see :meth:`_hf_metadata_allowlist`) because the
    # exact set drifts across the supported transformers range — e.g. the generation kwargs and
    # ``torchscript`` that v4 dumps into ``to_dict()`` were moved out to ``GenerationConfig`` in v5. This
    # static set covers the widely-shared metadata that Fast-LLM intentionally does not store but that a
    # bare ``PretrainedConfig`` does not carry (model-specific defaults like ``max_position_embeddings``).
    _HF_METADATA_ALLOWLIST: typing.ClassVar[frozenset[str]] = frozenset(
        {
            # transformers metadata Fast-LLM does not store that a bare ``PretrainedConfig().to_dict()``
            # omits across the supported range (so the dynamic union would miss them).
            "auto_map",
            "torch_dtype",
            "use_cache",
            # Architecture-family marker some transformers v4 configs carry (e.g. LlamaConfig); dropped
            # in v5, not consumed by Fast-LLM, and absent from a bare ``PretrainedConfig``.
            "is_llama_config",
            # Token ids — generation/inference, not architecture (a bare v5 config omits these).
            "bos_token_id",
            "decoder_start_token_id",
            "eos_token_id",
            "mask_token_id",
            "pad_token_id",
            "sep_token_id",
            # Generation defaults — never architecture.
            "bad_words_ids",
            "begin_suppress_tokens",
            "diversity_penalty",
            "do_sample",
            "early_stopping",
            "encoder_no_repeat_ngram_size",
            "exponential_decay_length_penalty",
            "forced_bos_token_id",
            "forced_eos_token_id",
            "length_penalty",
            "max_length",
            "min_length",
            "no_repeat_ngram_size",
            "num_beam_groups",
            "num_beams",
            "num_return_sequences",
            "output_scores",
            "remove_invalid_values",
            "repetition_penalty",
            "return_dict_in_generate",
            "suppress_tokens",
            "temperature",
            "top_k",
            "top_p",
            "typical_p",
            # Initialization / pretraining metadata Fast-LLM does not consume.
            "initializer_range",
            "max_position_embeddings",
            "pretraining_tp",
            # Family markers / default-valued knobs serialized by recent transformers versions.
            "is_llama_config",
            "rope_interleaved",
        }
    )

    @classmethod
    @functools.cache
    def _hf_metadata_allowlist(cls) -> frozenset[str]:
        """Static allowlist unioned with the live ``PretrainedConfig`` field set.

        Every key a bare ``PretrainedConfig`` carries is generic transformers metadata, never
        architecture, so deriving them from the installed transformers keeps the coverage check correct
        across the supported version range instead of hard-coding a version-specific set.
        """
        import transformers

        return cls._HF_METADATA_ALLOWLIST | frozenset(transformers.PretrainedConfig().to_dict())

    @classmethod
    def _check_hf_coverage(cls, config: dict[str, typing.Any]) -> None:
        """Run the HF-side coverage check at the import boundary.

        Subclasses that override :meth:`_import_config` should call this explicitly to keep the check
        active.
        """
        cls.base_model_converter_class.check_hf_coverage(config, allowlist=cls._hf_metadata_allowlist())

    @classmethod
    def _import_config(cls, config: dict[str, typing.Any]) -> FastLLMModelConfig:
        Assert.eq(config["model_type"], cls.get_huggingface_model_type())
        Assert.eq(config["architectures"], [cls.architecture])
        cls._check_hf_coverage(config)
        return cls._model_class.from_dict({"base_model": cls.base_model_converter_class.import_config(config)})

    def _create_weight_converters(self) -> list[WeightConverter]:
        return self.base_model_converter_class.get_converters(self._model.config.base_model)

    def _load_weights(
        self, config: CheckpointLoadConfig, device
    ) -> typing.Iterator[tuple[str, str, torch.Tensor | SafeTensorSlice]]:
        import transformers

        Assert.eq(self.get_shard_names(config), ("weights",))
        directory = self._resolve_path(config.path)
        if (directory / transformers.utils.SAFE_WEIGHTS_NAME).is_file():
            paths = {directory / transformers.utils.SAFE_WEIGHTS_NAME}
        elif (directory / transformers.utils.SAFE_WEIGHTS_INDEX_NAME).is_file():
            logger.info(f"Loading index from {directory / transformers.utils.SAFE_WEIGHTS_INDEX_NAME}")
            paths = {
                directory / path
                for path in json.loads((directory / transformers.utils.SAFE_WEIGHTS_INDEX_NAME).read_text())[
                    "weight_map"
                ].values()
            }
        elif (directory / transformers.utils.WEIGHTS_NAME).is_file():
            paths = {directory / transformers.utils.WEIGHTS_NAME}
        elif (directory / transformers.utils.WEIGHTS_INDEX_NAME).is_file():
            logger.info(f"Loading index from {directory / transformers.utils.WEIGHTS_INDEX_NAME}")
            paths = {
                directory / path
                for path in json.loads((directory / transformers.utils.WEIGHTS_INDEX_NAME).read_text())[
                    "weight_map"
                ].values()
            }
        else:
            raise FileNotFoundError(f"No compatible checkpoint found in {directory}")

        for path in paths:
            logger.info(f"Loading from {path}")
            if path.suffix == ".safetensors":
                with safetensors.safe_open(path, framework="pt", device=str(device)) as f:
                    for key in f.keys():
                        yield key, "weights", f.get_slice(key)
            elif path.suffix == ".bin":
                for key, tensor in torch.load(path, weights_only=True).items():
                    yield key, "weights", tensor
            else:
                raise NotImplementedError(f"Unknown file format for {path}")
