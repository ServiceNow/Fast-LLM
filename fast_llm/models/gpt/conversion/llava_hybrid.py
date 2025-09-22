import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import ExternalStateDictCheckpointHandler
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.models.gpt.conversion.config import LlavaHybridCheckpointFormat
from fast_llm.models.gpt.conversion.llava import LlavaHuggingfaceCheckpointHandler


class LlavaHybridHuggingfaceCheckpointHandler(CustomModelingExportMixin, LlavaHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = LlavaHybridCheckpointFormat
    architecture: typing.ClassVar[str] = "LlavaHybridForConditionalGeneration"
    modeling_file = modeling_llava_hybrid.__file__
    configuration_file = configuration_llava_hybrid.__file__
    configuration_cls: typing.ClassVar[type[PretrainedConfig]] = configuration_llava_hybrid.LlavaHybridConfig
    _model_class: typing.ClassVar[FastLLMModelConfig] = HybridSSMModelConfig
    additional_files = [
        modeling_ssm_hybrid_apriel15b.__file__,
        configuration_ssm_hybrid_apriel15b.__file__,
    ]

    @classmethod
    def get_text_handler_class(cls) -> type[ExternalStateDictCheckpointHandler]:
        from fast_llm.models.ssm.conversion import AprielThinkerSSMHHybridHuggingfaceCheckpointHandler

        return AprielThinkerSSMHHybridHuggingfaceCheckpointHandler

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            ConstantExportParamConverter(
                export_names=(("auto_map",),),
                export_value={
                    "AutoConfig": "configuration_llava_hybrid.LlavaHybridConfig",
                    "AutoModel": "modeling_llava_hybrid.LlavaHybridModel",
                    "AutoModelForVision2Seq": "modeling_llava_hybrid.LlavaHybridForConditionalGeneration",
                    "AutoModelForCausalLM": "modeling_llava_hybrid.LlavaHybridForConditionalGeneration",
                },
            ),
        ]
