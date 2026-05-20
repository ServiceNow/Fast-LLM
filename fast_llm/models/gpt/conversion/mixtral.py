import functools
import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import (
    ConstantImportConfigConverter,
    IgnoredConfigConverter,
    LinearWeightConverter,
    RenameConfigConverter,
    SplitWeightConverter,
    TransposeSplitWeightConverter,
    WeightConverter,
)
from fast_llm.layers.decoder.mlp.config import MoEMLPConfig, RoutingType
from fast_llm.models.gpt.conversion.config import MixtralCheckpointFormat
from fast_llm.models.gpt.conversion.llama import LlamaMLPConverter
from fast_llm.models.gpt.conversion.mistral import (
    MistralBaseModelConverter,
    MistralBlockConverter,
    MistralHeadConverter,
    MistralHuggingfaceCheckpointHandler,
)
from fast_llm.utils import Assert


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
            # Mixtral has no HF representation for the router sub-config. The blanket consume satisfies
            # architecture coverage; non-architecture fields (lr_scale, apply_peft, weight.initialization,
            # weight.lr_scale) cannot round-trip through the HF format by design — Fast-LLM keeps them on
            # the in-memory config independently. The only architecture-hint sub-field is ``router.weight``,
            # a ParameterConfig with no architecture sub-fields, so the blanket carries no real risk.
            "router": IgnoredConfigConverter(("router",)),
            "router_normalization": ConstantImportConfigConverter(("router_normalization",), None),
            "router_scale": IgnoredConfigConverter(("router_scale",)),
            "router_input_scale": ConstantImportConfigConverter(("router_input_scale",), 1.0),
            "router_per_expert_scale": IgnoredConfigConverter(("router_per_expert_scale",)),
            # Router / inference toggles surfaced by HF but not consumed by Fast-LLM's MoEMLPConfig
            # (auxiliary_loss_coefficient and jitter_eps are FieldHint.feature, not architecture).
            "router_runtime_unsupported": IgnoredConfigConverter(
                hf_paths=(("router_aux_loss_coef",), ("router_jitter_noise",), ("output_router_logits",)),
            ),
        }

    @classmethod
    def _validate_export(cls, config: MoEMLPConfig) -> None:
        super()._validate_export(config)
        Assert.custom(lambda v: not v, config.router_scale.enabled)
        Assert.custom(lambda v: not v, config.router_per_expert_scale.enabled)

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {
            "router": LinearWeightConverter("router", "gate"),
            "layer_1": LinearWeightConverter(
                "layer_1",
                lambda c: tuple(f"experts.{i}.{w}" for i in range(c.experts) for w in ("w1", "w3")),
                transform=SplitWeightConverter,
            ),
            "layer_2": LinearWeightConverter(
                "layer_2",
                lambda c: tuple(f"experts.{i}.w2" for i in range(c.experts)),
                transform=TransposeSplitWeightConverter,
            ),
        }


class MixtralBlockConverter(MistralBlockConverter):
    hf_mlp_name: typing.ClassVar[str] = "block_sparse_moe"
    mlp_converter_class: typing.ClassVar[type[MixtralMLPConverter]] = MixtralMLPConverter


class MixtralHeadConverter(MistralHeadConverter):
    block_converter_class: typing.ClassVar[type[MixtralBlockConverter]] = MixtralBlockConverter


class MixtralBaseModelConverter(MistralBaseModelConverter):
    block_converter_class: typing.ClassVar[type[MixtralBlockConverter]] = MixtralBlockConverter
    head_converter_class: typing.ClassVar[type[MixtralHeadConverter]] = MixtralHeadConverter


class MixtralHuggingfaceCheckpointHandler(MistralHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = MixtralCheckpointFormat
    architecture: typing.ClassVar[str] = "MixtralForCausalLM"
    base_model_converter_class: typing.ClassVar[type[MixtralBaseModelConverter]] = MixtralBaseModelConverter

    @classmethod
    def get_transformers_configuration_class(cls):
        import transformers

        return transformers.MixtralConfig
