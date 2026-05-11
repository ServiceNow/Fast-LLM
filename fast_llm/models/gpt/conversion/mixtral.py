import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import (
    ConstantImportConfigConverter,
    IgnoredConfigConverter,
    RenameConfigConverter,
    SplitWeightConverter,
    WeightConverter,
)
from fast_llm.layers.decoder.mlp.config import MoEMLPConfig, RoutingType
from fast_llm.models.gpt.conversion.config import MixtralCheckpointFormat
from fast_llm.models.gpt.conversion.llama import LlamaMLPConverter, MLPLayer2Converter, get_weight_and_bias_converters
from fast_llm.models.gpt.conversion.mistral import (
    MistralBaseModelConverter,
    MistralBlockConverter,
    MistralDecoderConverter,
    MistralHeadConverter,
    MistralHuggingfaceCheckpointHandler,
)


class MixtralMLPConverter(LlamaMLPConverter):
    fast_llm_config_class = MoEMLPConfig

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            **super()._create_config_converters(),
            # Mixtral has no `mlp_bias` HF field; biases are always disabled.
            "add_linear_biases": ConstantImportConfigConverter(("add_linear_biases",), False),
            "experts": RenameConfigConverter(("experts",), ("num_local_experts",)),
            "experts_per_token": RenameConfigConverter(("experts_per_token",), ("num_experts_per_tok",)),
            # Mixtral has no shared experts and uses the topk default; assert on export, inject defaults on import.
            "shared_experts": ConstantImportConfigConverter(("shared_experts",), 0),
            "routing": ConstantImportConfigConverter(("routing",), RoutingType.topk),
            # Mixtral's gate is a default LinearConfig (no bias); blanket-consume so coverage passes.
            "router": IgnoredConfigConverter(("router",)),
            # Router / inference toggles surfaced by HF but not consumed by Fast-LLM's MoEMLPConfig
            # (auxiliary_loss_coefficient and jitter_eps are FieldHint.feature, not architecture).
            "router_runtime_unsupported": IgnoredConfigConverter(
                hf_paths=(("router_aux_loss_coef",), ("router_jitter_noise",), ("output_router_logits",)),
            ),
        }

    @classmethod
    def import_config(cls, hf_dict: dict) -> dict:
        # Inject the Fast-LLM dynamic-type discriminator so `from_dict` instantiates `MoEMLPConfig`
        # rather than the default `MLPConfig`. The MLP is wrapped via `NestedConfigConverter`, so
        # there's no surrounding `DispatchConfigConverter` to inject this for us.
        return {"type": "moe", **super().import_config(hf_dict)}

    @classmethod
    def get_converters(
        cls,
        config: MoEMLPConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.router",
                f"{hf_prefix}.gate",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_1",
                tuple(f"{hf_prefix}.experts.{i}.{w}" for i in range(config.experts) for w in ("w1", "w3")),
                False,
                SplitWeightConverter,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_2",
                tuple(f"{hf_prefix}.experts.{i}.w2" for i in range(config.experts)),
                False,
                MLPLayer2Converter,
                drop_on_export=drop_on_export,
            ),
        ]


class MixtralBlockConverter(MistralBlockConverter):
    hf_mlp_name: typing.ClassVar[str] = "block_sparse_moe"
    mlp_converter_class: typing.ClassVar[type[MixtralMLPConverter]] = MixtralMLPConverter


class MixtralDecoderConverter(MistralDecoderConverter):
    block_converter_class: typing.ClassVar[type[MixtralBlockConverter]] = MixtralBlockConverter


class MixtralHeadConverter(MistralHeadConverter):
    block_converter_class: typing.ClassVar[type[MixtralBlockConverter]] = MixtralBlockConverter


class MixtralBaseModelConverter(MistralBaseModelConverter):
    decoder_converter_class: typing.ClassVar[type[MixtralDecoderConverter]] = MixtralDecoderConverter
    head_converter_class: typing.ClassVar[type[MixtralHeadConverter]] = MixtralHeadConverter


class MixtralHuggingfaceCheckpointHandler(MistralHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = MixtralCheckpointFormat
    architecture: typing.ClassVar[str] = "MixtralForCausalLM"
    base_model_converter_class: typing.ClassVar[type[MixtralBaseModelConverter]] = MixtralBaseModelConverter

    @classmethod
    def get_transformers_configuration_class(cls):
        import transformers

        return transformers.MixtralConfig
