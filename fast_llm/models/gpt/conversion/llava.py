import typing

from fast_llm import __version__
from fast_llm.config import MISSING, get_nested_dict_value, set_nested_dict_value
from fast_llm.engine.base_model.config import BaseModelConfig
from fast_llm.engine.checkpoint.config import CheckpointFormat, CheckpointLoadMetadataConfig
from fast_llm.engine.checkpoint.external import ExternalStateDictCheckpointHandler
from fast_llm.engine.checkpoint.huggingface import HuggingfaceStateDictCheckpointHandler
from fast_llm.engine.multi_stage.config import CheckpointMetadata, FastLLMModelConfig
from fast_llm.functional.config import ActivationType
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTModelConfig
from fast_llm.models.gpt.conversion.auto import AutoGPTHuggingfaceCheckpointHandler
from tests.utils.model_configs import LlavaGPTHuggingfaceCheckpointFormat


class LlavaHuggingfaceCheckpointHandler(HuggingfaceStateDictCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = LlavaGPTHuggingfaceCheckpointFormat
    architecture: typing.ClassVar[str] = "LlavaForConditionalGeneration"
    _model_class: typing.ClassVar[FastLLMModelConfig] = GPTModelConfig

    @classmethod
    def get_vision_handler_class(cls) -> type[ExternalStateDictCheckpointHandler]:
        return AutoGPTHuggingfaceCheckpointHandler.get_handler_class(cls.format.vision_name)

    @classmethod
    def get_text_handler_class(cls) -> type[ExternalStateDictCheckpointHandler]:
        return AutoGPTHuggingfaceCheckpointHandler.get_handler_class(cls.format.text_name)

    @classmethod
    def _load_metadata(cls, config: CheckpointLoadMetadataConfig) -> CheckpointMetadata:
        vision_handler_cls = cls.get_vision_handler_class()
        text_handler_cls = cls.get_text_handler_class()
        cfg_dict = cls._load_config(config.path)
        kwargs = {}
        if "text_config" in cfg_dict:
            text_kwargs = text_handler_cls._import_config_dict(cfg_dict["text_config"])
            kwargs.update(text_kwargs)
        if "vision_config" in cfg_dict:
            vision_kwargs = vision_handler_cls._import_config_dict(cfg_dict["vision_config"])
            vision_kwargs = {tuple(["vision_encoder"] + list(key)): value for key, value in vision_kwargs.items()}
            kwargs.update(vision_kwargs)
        kwargs.update(
            cls._import_config(
                {key: value for key, value in cfg_dict.items() if key not in ("text_config", "vision_config")}
            )
        )
        imported_model_config = cls._model_class.get_base_model_config_class().from_dict({}, kwargs)
        return CheckpointMetadata(
            fast_llm_version=__version__,
            model=cls._model_class,
            format=config.format,
            config=cls._model_class.from_dict({"base_model": imported_model_config.to_dict()}),
            shards=["weights"],
        )

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantExportParamConverter(export_names=(("architectures",),), export_value=[cls.architecture]),
            MappedConfigParamConverter(
                fast_llm_names=(("vision_encoder", "adapter_activation_type"),),
                export_names=(("projector_hidden_act",),),
                fast_llm_value=ActivationType.from_hf_name,
                export_value=lambda activation_type: activation_type.hf_name,
            ),
            RenameParamConverter(
                fast_llm_names=(("vision_encoder", "adapter_size"),),
                export_names=(("projector_intermediate_size",),),
            ),
        ]

    @classmethod
    def _import_config(cls, config: dict[str, typing.Any]) -> GPTBaseModelConfig:
        # handler_cls = AutoGPTHuggingfaceCheckpointHandler.get_handler_class(config["model_type"])
        kwargs = {}
        for converter in cls._create_config_converters():
            try:
                values = ()
                for export_name in converter.export_names:
                    try:
                        value = get_nested_dict_value(config, export_name)
                    except KeyError:
                        value = MISSING
                    values = values + (value,)
                values = converter.import_params(values)
                for fast_llm_name, value in zip(converter.fast_llm_names, values, strict=True):
                    if value is MISSING:
                        raise ValueError(f"Missing converted value for fast-llm parameter {fast_llm_name}")
                    if fast_llm_name in kwargs:
                        raise ValueError(f"Duplicate converted value for fast-llm parameter {fast_llm_name}")
                    kwargs[fast_llm_name] = value
            except Exception as e:
                raise RuntimeError(f"Config conversion failed for converter {converter}", *e.args)

        return kwargs

    @classmethod
    def _export_config(cls, config: BaseModelConfig) -> dict[str, typing.Any]:
        exported_config = {}
        vision_handler_cls = cls.get_vision_handler_class()
        text_handler_cls = cls.get_text_handler_class()
        for converter in vision_handler_cls._create_config_converters():
            try:
                values = converter.export_params(
                    tuple(
                        cls._get_fast_llm_attribute(config, ("vision_encoder",) + fast_llm_name)
                        for fast_llm_name in converter.fast_llm_names
                    )
                )
                for export_name, value in zip(converter.export_names, values, strict=True):
                    if value is not MISSING:
                        set_nested_dict_value(exported_config, ("vision_config",) + export_name, value)
            except Exception as e:
                raise RuntimeError(f"Config conversion failed for converter {converter}", *e.args)

        for converter in text_handler_cls._create_config_converters():
            try:
                values = converter.export_params(
                    tuple(
                        cls._get_fast_llm_attribute(config, fast_llm_name)
                        for fast_llm_name in converter.fast_llm_names
                    )
                )
                for export_name, value in zip(converter.export_names, values, strict=True):
                    if value is not MISSING:
                        set_nested_dict_value(exported_config, ("text_config",) + export_name, value)
            except Exception as e:
                raise RuntimeError(f"Config conversion failed for converter {converter}", *e.args)

        for converter in cls._create_config_converters():
            try:
                values = converter.export_params(
                    tuple(
                        cls._get_fast_llm_attribute(config, fast_llm_name)
                        for fast_llm_name in converter.fast_llm_names
                    )
                )
                for export_name, value in zip(converter.export_names, values, strict=True):
                    if value is not MISSING:
                        set_nested_dict_value(exported_config, export_name, value)
            except Exception as e:
                raise RuntimeError(f"Config conversion failed for converter {converter}", *e.args)

        return exported_config

    def _create_weight_converters(self):
        vision_handler_cls = self.get_vision_handler_class()
        vision_handler = vision_handler_cls(self._model)
        converters = vision_handler._create_weight_converters(hf_base_prefix="vision_tower.", offset=0)
        text_handler_cls = self.get_text_handler_class()
        text_handler = text_handler_cls(self._model)
        converters.extend(
            text_handler._create_weight_converters(hf_base_prefix="language_model.", offset=vision_handler.num_layers)
        )
        return converters
