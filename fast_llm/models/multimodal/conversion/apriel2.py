"""Apriel2 multimodal checkpoint format converter."""

import functools
import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import (
    BlockSequenceWeightConverter,
    ConfigSectionConverter,
    ConstantExportConfigConverter,
    ConstantImportConfigConverter,
    CustomConfigConverter,
    IgnoredConfigConverter,
    LinearWeightConverter,
    NestedConfigConverter,
    NestedWeightConverter,
    OptionalConfigConverter,
    OutputProjectionWeightConverter,
    PatchEmbeddingWeightConverter,
    RenameConfigConverter,
    SelfBlockSequenceWeightConverter,
    TransposeSplitWeightConverter,
    WeightConverter,
)
from fast_llm.engine.checkpoint.huggingface import HuggingFaceBaseModelConverter, HuggingfaceStateDictCheckpointHandler
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.attention.rotary.config import DefaultRotaryConfig, Rotary2DConfig
from fast_llm.layers.block.config import FixedBlockSequenceConfig
from fast_llm.layers.common.normalization.config import RMSNormalizationConfig
from fast_llm.layers.decoder.config import DecoderBlockConfig
from fast_llm.layers.decoder.mlp.config import MLPConfig
from fast_llm.layers.vision.config import PatchEmbeddingsConfig, VisionEncoderConfig
from fast_llm.models.gpt.conversion.apriel2 import (
    Apriel2BaseModelConverter,
    Apriel2BlockConverter,
    Apriel2HeadConverter,
    Apriel2MLPConverter,
    Apriel2RMSNormConverter,
)
from fast_llm.models.gpt.conversion.llama import LlamaEmbeddingsConverter, LlamaNormalizationConverter
from fast_llm.models.multimodal.config import MultiModalBaseModelConfig, MultiModalModelConfig
from fast_llm.models.multimodal.conversion.config import Apriel2CheckpointFormat
from fast_llm.models.multimodal.conversion.llava import PixtralAttentionConverter
from fast_llm.models.multimodal.model import MultiModalModel
from fast_llm.utils import Assert


def _apriel2_vision_attention_rotary_export(config: AttentionConfig) -> dict:
    """Emit the Apriel2-vision rotary subdict. Two rotary types are supported:
    :class:`Rotary2DConfig` (HF ``pixtral_2d``) and :class:`DefaultRotaryConfig` (HF ``mistral_1d``).
    ``patch_size``/``max_image_size`` HF metadata is injected by the parent vision-model converter
    (it derives from ``embeddings.patch_height``, outside this scope)."""
    rotary = config.rotary
    if type(rotary) is Rotary2DConfig:
        return {("rotary",): {"type": "pixtral_2d", "theta": rotary.theta}}
    if type(rotary) is DefaultRotaryConfig:
        return {("rotary",): {"type": "mistral_1d", "theta": rotary.theta}}
    raise NotImplementedError(f"Unsupported rotary type: {type(rotary).__name__}")


def _apriel2_vision_attention_rotary_import(hf_dict: dict) -> dict:
    rotary = dict(hf_dict["rotary"])
    if rotary.get("type") == "pixtral_2d":
        rotary["type"] = "default_2d"
    elif rotary.get("type") == "mistral_1d":
        rotary["type"] = "default"
    rotary.pop("patch_size", None)
    rotary.pop("max_image_size", None)
    return {("rotary",): rotary}


class Apriel2VisionAttentionConverter(PixtralAttentionConverter):
    """Converts :class:`AttentionConfig` ↔ Apriel2 vision attention HF subdict (typed ``"attention"``).

    Apriel2's vision attention shape uses Apriel2-native field names (``heads``, ``head_groups``, ``head_size``,
    ``add_linear_biases``, ``causal``) plus an explicit ``cross_document_attention=False`` flag and a nested
    typed ``rotary`` block. Differs from the text :class:`Apriel2AttentionConverter` mainly in the rotary type
    set (``pixtral_2d``/``mistral_1d`` instead of ``mistral_1d``/``llama3``/``yarn``) and the lack of
    per-layer-bias and ``window_size`` representations.

    Inherits :meth:`get_converters` from :class:`PixtralAttentionConverter` (Llama-style q/k/v/o weight layout).
    """

    hf_type_name = "attention"

    @classmethod
    def _create_config_converters(cls) -> dict:
        # Replace Pixtral's declarations wholesale: Apriel2 vision uses Apriel2-native field names, allows GQA
        # and both Rotary2D + DefaultRotary, and has no HF representation for per-layer biases or window_size.
        return {
            "heads": RenameConfigConverter(("heads",), ("heads",)),
            "head_groups": RenameConfigConverter(("head_groups",), ("head_groups",)),
            "head_size": RenameConfigConverter(("head_size",), ("head_size",)),
            "add_linear_biases": RenameConfigConverter(("add_linear_biases",), ("add_linear_biases",)),
            "causal": RenameConfigConverter(("causal",), ("causal",)),
            "cross_document_attention": ConstantExportConfigConverter(("cross_document_attention",), False),
            "rotary": CustomConfigConverter(
                fast_llm_paths=(("rotary",),),
                hf_paths=(("rotary",),),
                export_fn=_apriel2_vision_attention_rotary_export,
                import_fn=_apriel2_vision_attention_rotary_import,
                fast_llm_recurses=True,
            ),
            # Apriel2 vision attention has no per-layer bias representation; the Fast-LLM defaults round-trip.
            "linear_layers": IgnoredConfigConverter(
                ("query_layer",), ("key_layer",), ("value_layer",), ("dense_layer",)
            ),
            "softmax_scale_power": IgnoredConfigConverter(("softmax_scale_power",)),
            "query_norm": ConstantImportConfigConverter(("query_norm",), None),
            "key_norm": ConstantImportConfigConverter(("key_norm",), None),
            "value_norm": ConstantImportConfigConverter(("value_norm",), None),
            "shared_key_value": ConstantImportConfigConverter(("shared_key_value",), False),
        }

    @classmethod
    def _validate_export(cls, config: AttentionConfig) -> None:
        # Replace Pixtral's Rotary2D-only + head_groups==heads checks (Apriel2 vision allows both rotary types
        # and supports GQA). Keep the per-layer bias consistency check from the Llama base.
        Assert.incl(config.query_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.key_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.value_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.dense_layer.bias.enabled, (None, config.add_linear_biases))


class Apriel2VisionMLPConverter(Apriel2MLPConverter):
    """The vision-side MLP shape ``{type: mlp, intermediate_size, activation, gated, add_linear_biases}``.

    Distinct from the text :class:`Apriel2MLPConverter` only in lacking the per-layer-bias declaration: the
    Apriel2 vision MLP HF shape has no representation for per-layer ``bias.enabled`` overrides, so the
    Fast-LLM defaults are dropped on export (declared :class:`IgnoredConfigConverter`) and re-defaulted on
    import. Weight-side ``get_converters`` is inherited from the text MLP.
    """

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            **super()._create_config_converters(),
            # Apriel2 vision MLP has no per-layer ``bias.enabled`` representation; the Fast-LLM defaults
            # round-trip. Replaces the text MLP's ``_per_layer_bias_converter`` claim.
            "layers": IgnoredConfigConverter(("layer_1",), ("layer_2",)),
        }

    @classmethod
    def _validate_export(cls, config: MLPConfig) -> None:
        # The inherited weight side reads ``effective_bias(config.layer_X, config.add_linear_biases)``; if
        # a per-layer override diverges from ``add_linear_biases`` the bias would silently include/exclude
        # weight tensors that the HF format cannot describe (since per-layer bias is Ignored on the config
        # side).
        Assert.incl(config.layer_1.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.layer_2.bias.enabled, (None, config.add_linear_biases))


class Apriel2VisionBlockConverter(ConfigSectionConverter):
    """Converts a vision :class:`DecoderBlockConfig` ↔ Apriel2's nested ``{mixer, mlp, normalization}`` block.

    Distinct from :class:`PixtralBlockConverter` (which flat-merges its children into the parent's HF dict)
    because the Apriel2 vision format nests each sub-section under a typed sub-key, matching the Apriel2 text
    decoder shape.
    """

    fast_llm_config_class = DecoderBlockConfig

    mixer_converter_class: typing.ClassVar[type[Apriel2VisionAttentionConverter]] = Apriel2VisionAttentionConverter
    mlp_converter_class: typing.ClassVar[type[Apriel2VisionMLPConverter]] = Apriel2VisionMLPConverter
    # Config-side: the Apriel2 HF format nests normalization as ``{"type": "rms_norm", "epsilon": ...}``;
    # ``Apriel2RMSNormConverter`` handles the typed shape. Weight side uses LlamaNormalizationConverter
    # directly (flat parameter names — independent of how the surrounding HF config is structured).
    normalization_converter_class: typing.ClassVar[type[Apriel2RMSNormConverter]] = Apriel2RMSNormConverter

    hf_mixer_name: typing.ClassVar[str] = "mixer"
    hf_mlp_name: typing.ClassVar[str] = "mlp"
    hf_norm_1_name: typing.ClassVar[str] = "input_layernorm"
    hf_norm_2_name: typing.ClassVar[str] = "post_attention_layernorm"

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "mixer": NestedConfigConverter(("mixer",), cls.mixer_converter_class, hf_path=("mixer",)),
            "mlp": NestedConfigConverter(("mlp",), cls.mlp_converter_class, hf_path=("mlp",)),
            "normalization": NestedConfigConverter(
                ("normalization",), cls.normalization_converter_class, hf_path=("normalization",)
            ),
            "pre_mixer_normalization": ConstantImportConfigConverter(("pre_mixer_normalization",), None),
            "pre_mlp_normalization": ConstantImportConfigConverter(("pre_mlp_normalization",), None),
            "post_mixer_normalization": ConstantImportConfigConverter(("post_mixer_normalization",), None),
            "post_mlp_normalization": ConstantImportConfigConverter(("post_mlp_normalization",), None),
            "output_scale": IgnoredConfigConverter(("output_scale",)),
        }

    @classmethod
    def _validate_export(cls, config: DecoderBlockConfig) -> None:
        # Config side binds Apriel2RMSNormConverter via plain Nested (RMS-only), so this is currently
        # unreachable in practice. Mirror the text-side assertion so a future widening of the config
        # dispatch doesn't silently produce phantom norm_1.weight/norm_2.weight converters.
        Assert.is_(type(config.normalization), RMSNormalizationConfig)
        Assert.custom(lambda v: not v, config.output_scale.enabled)

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {
            "mixer": NestedWeightConverter("mixer", cls.hf_mixer_name, cls.mixer_converter_class),
            "mlp": NestedWeightConverter("mlp", cls.hf_mlp_name, cls.mlp_converter_class),
            "norm_1": NestedWeightConverter(
                "norm_1", cls.hf_norm_1_name, LlamaNormalizationConverter, config_attr="normalization"
            ),
            "norm_2": NestedWeightConverter(
                "norm_2", cls.hf_norm_2_name, LlamaNormalizationConverter, config_attr="normalization"
            ),
        }


class Apriel2VisionEncoderConverter(ConfigSectionConverter):
    """Converts a :class:`FixedBlockSequenceConfig` (vision encoder) ↔ Apriel2 HF ``encoder`` subdict + the
    flat ``num_hidden_layers`` mirror that the HF format also requires at the surrounding vision_config level.

    No ``hf_type_name`` is set: the ``type: "fixed"`` discriminator lives *inside* the ``encoder`` subdict
    (emitted by the Custom's export_fn), not at the parent vision_config level. The Fast-LLM-side ``type``
    is auto-injected by :meth:`ConfigSectionConverter.import_config` via ``fast_llm_config_class.dynamic_type_name``.
    """

    fast_llm_config_class = FixedBlockSequenceConfig

    block_converter_class: typing.ClassVar[type[Apriel2VisionBlockConverter]] = Apriel2VisionBlockConverter

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "encoder": CustomConfigConverter(
                fast_llm_paths=(("num_blocks",), ("block",)),
                hf_paths=(("encoder",),),
                export_fn=lambda c: {
                    ("encoder",): {
                        "type": "fixed",
                        "num_blocks": c.num_blocks,
                        "block": cls.block_converter_class.export_config(c.block),
                    },
                },
                import_fn=lambda hf: {
                    ("num_blocks",): hf["encoder"]["num_blocks"],
                    ("block",): cls.block_converter_class.import_config(hf["encoder"]["block"]),
                },
                fast_llm_recurses=True,
            ),
            "num_hidden_layers_mirror": CustomConfigConverter(
                fast_llm_paths=(),
                hf_paths=(("num_hidden_layers",),),
                export_fn=lambda c: {("num_hidden_layers",): c.num_blocks},
                import_fn=lambda hf: {},
            ),
        }

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {
            "blocks": SelfBlockSequenceWeightConverter(cls.block_converter_class),
        }


class Apriel2EmbeddingsConverter(ConfigSectionConverter):
    """Converts :class:`PatchEmbeddingsConfig` ↔ Apriel2 HF ``embeddings`` subdict, with top-level
    ``patch_size``/``num_channels`` mirrors that the Apriel2 vision_config also requires."""

    fast_llm_config_class = PatchEmbeddingsConfig

    normalization_converter_class: typing.ClassVar[type[Apriel2RMSNormConverter]] = Apriel2RMSNormConverter

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "patch_height": RenameConfigConverter(("patch_height",), ("embeddings", "patch_height")),
            "patch_width": RenameConfigConverter(("patch_width",), ("embeddings", "patch_width")),
            "normalization": NestedConfigConverter(
                ("normalization",),
                cls.normalization_converter_class,
                hf_path=("embeddings", "normalization"),
            ),
            # ``patch_embeddings`` (AffineLinearConfig) carries no HF architecture info; bias presence validated below.
            "patch_embeddings": IgnoredConfigConverter(("patch_embeddings",)),
            # ``input_channels`` is a cached_property pinned to 3 on the Fast-LLM side; HF emits it under
            # ``embeddings`` and again as a top-level ``num_channels`` mirror.
            "embeddings_input_channels": ConstantExportConfigConverter(("embeddings", "input_channels"), 3),
            "num_channels_mirror": ConstantExportConfigConverter(("num_channels",), 3),
            # ``patch_size`` HF top-level mirror of ``embeddings.patch_height`` — emit on export, ignored on
            # import (the under-``embeddings`` path is the authoritative source).
            "patch_size_mirror": CustomConfigConverter(
                fast_llm_paths=(),
                hf_paths=(("patch_size",),),
                export_fn=lambda c: {("patch_size",): c.patch_height},
                import_fn=lambda hf: {},
            ),
        }

    @classmethod
    def _validate_export(cls, config: PatchEmbeddingsConfig) -> None:
        Assert.eq(config.patch_height, config.patch_width)
        Assert.incl(config.patch_embeddings.bias.enabled, (None, False))

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {
            "patch_embeddings": LinearWeightConverter(
                "patch_embeddings",
                "patch_embeddings",
                transform=PatchEmbeddingWeightConverter,
                bias_fn=False,
            ),
            "normalization": NestedWeightConverter(
                "normalization", "normalization", LlamaNormalizationConverter, config_attr="normalization"
            ),
        }


class Apriel2VisionAdapterConverter(Apriel2VisionMLPConverter):
    """Converts :class:`MLPConfig` (adapter) ↔ Apriel2 HF ``adapter`` subdict.

    Config side: shares the vision MLP declaration (typed ``{type: mlp, ...}`` shape with no per-layer-bias
    representation). Weight side: uses ``linear_1`` / ``linear_2`` weight names matching the Llava-style
    adapter (distinct from the Apriel2 MLP ``gate_proj``/``up_proj``/``down_proj`` layout).
    """

    @classmethod
    def _validate_export(cls, config: MLPConfig) -> None:
        Assert.incl(config.layer_1.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.layer_2.bias.enabled, (None, config.add_linear_biases))

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {
            "layer_1": LinearWeightConverter("layer_1", "linear_1"),
            "layer_2": LinearWeightConverter("layer_2", "linear_2", transform=TransposeSplitWeightConverter),
        }


class Apriel2VisionModelConverter(ConfigSectionConverter):
    """Top-level vision-encoder converter. The HF representation lives under a single ``vision_encoder`` key,
    so declarations are written relative to that nested subdict.

    ``patch_size``/``max_image_size`` rotary metadata is injected here (cross-section reference to
    ``embeddings.patch_height``) — the attention converter cannot see it from its own scope.
    """

    fast_llm_config_class = VisionEncoderConfig

    embeddings_converter_class: typing.ClassVar[type[Apriel2EmbeddingsConverter]] = Apriel2EmbeddingsConverter
    encoder_converter_class: typing.ClassVar[type[Apriel2VisionEncoderConverter]] = Apriel2VisionEncoderConverter
    vision_adapter_converter_class: typing.ClassVar[type[Apriel2VisionAdapterConverter]] = (
        Apriel2VisionAdapterConverter
    )

    hf_embeddings_prefix: typing.ClassVar[str] = "model.vision_encoder.embeddings"
    hf_encoder_prefix: typing.ClassVar[str] = "model.vision_encoder.encoder.blocks"
    hf_adapter_prefix: typing.ClassVar[str] = "model.vision_encoder.adapter"

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "embeddings": NestedConfigConverter(("embeddings",), cls.embeddings_converter_class),
            "encoder": NestedConfigConverter(("encoder",), cls.encoder_converter_class),
            "adapter": NestedConfigConverter(("adapter",), cls.vision_adapter_converter_class, hf_path=("adapter",)),
            "hidden_size": RenameConfigConverter(("hidden_size",), ("hidden_size",)),
            # Cross-section rotary metadata: the Apriel2 HF format requires patch_size + max_image_size inside
            # ``encoder.block.mixer.rotary`` (for ``pixtral_2d``), derived from embeddings.patch_height plus a
            # constant 1024. Written here because this converter is the smallest scope that sees both.
            # No fast_llm_paths/hf_paths claims — the encoder's recursive rotary claim covers HF coverage; the
            # values land on import via the same recursive claim and are stripped by the attention import_fn.
            "rotary_metadata": CustomConfigConverter(
                fast_llm_paths=(),
                hf_paths=(),
                export_fn=cls._inject_rotary_metadata,
                import_fn=lambda hf: {},
            ),
        }

    @staticmethod
    def _inject_rotary_metadata(config: VisionEncoderConfig) -> dict:
        rotary = config.encoder.block.mixer.rotary
        if type(rotary) is Rotary2DConfig:
            return {
                ("encoder", "block", "mixer", "rotary", "patch_size"): config.embeddings.patch_height,
                ("encoder", "block", "mixer", "rotary", "max_image_size"): 1024,
            }
        return {}

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        # The vision converter normally lives under a single ``vision_encoder`` HF prefix, but Apriel2's
        # state-dict places each piece at distinct absolute paths (``model.vision_encoder.{embeddings,
        # encoder.blocks, adapter}``). Use ``NestedWeightConverter`` with the absolute prefix on each entry;
        # the parent's :class:`NestedWeightConverter("vision_encoder", "", ...)` passes ``hf_prefix=""``,
        # so the absolute prefix lands as-is.
        return {
            "embeddings": NestedWeightConverter(
                "embeddings", cls.hf_embeddings_prefix, cls.embeddings_converter_class
            ),
            "encoder": NestedWeightConverter("encoder", cls.hf_encoder_prefix, cls.encoder_converter_class),
            "adapter": NestedWeightConverter("adapter", cls.hf_adapter_prefix, cls.vision_adapter_converter_class),
        }


class Apriel2MultimodalHeadConverter(Apriel2HeadConverter):
    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {
            "final_norm": NestedWeightConverter(
                "final_norm", "model.norm", cls.normalization_converter_class, config_attr="normalization"
            ),
            "output_weights": OutputProjectionWeightConverter("output_weights", "lm_head.weight"),
        }


class Apriel2MultimodalBaseModelConverter(HuggingFaceBaseModelConverter):
    """Top-level converter for Apriel2 multimodal. Composes the Apriel2 text base (flat-merged into the HF
    top-level dict) with an optional vision encoder (under HF key ``vision_encoder``) and an optional
    ``image_token_index`` field.

    Architecturally the Fast-LLM config (:class:`MultiModalBaseModelConfig`) multi-inherits from both
    :class:`GPTBaseModelConfig` (text) and :class:`VisionMultiModalModelConfig` (vision/image_token_index),
    so a single declaration set drives both halves.
    """

    fast_llm_config_class = MultiModalBaseModelConfig

    text_base_converter_class: typing.ClassVar[type[ConfigSectionConverter]] = Apriel2BaseModelConverter
    vision_model_converter_class: typing.ClassVar[type[Apriel2VisionModelConverter]] = Apriel2VisionModelConverter
    embeddings_converter_class: typing.ClassVar[type[LlamaEmbeddingsConverter]] = LlamaEmbeddingsConverter
    block_converter_class: typing.ClassVar[type[ConfigSectionConverter]] = Apriel2BlockConverter
    head_converter_class: typing.ClassVar[type[Apriel2MultimodalHeadConverter]] = Apriel2MultimodalHeadConverter

    @classmethod
    def _create_config_converters(cls) -> dict:
        text_base_cls = cls.text_base_converter_class
        vision_cls = cls.vision_model_converter_class

        def _vision_export(config: MultiModalBaseModelConfig) -> dict:
            if config.vision_encoder is None:
                return {}
            return {("vision_encoder",): vision_cls.export_config(config.vision_encoder)}

        def _vision_import(hf_dict: dict) -> dict:
            if "vision_encoder" not in hf_dict:
                return {}
            return {("vision_encoder",): vision_cls.import_config(hf_dict["vision_encoder"])}

        return {
            # Flat-merge the Apriel2 text base into the top-level HF dict. The text base claims every
            # GPTBaseModelConfig architecture leaf via its own declarations; we mark them recursively
            # consumed here and forward HF coverage via the text base's ``_consumed_hf_paths``.
            "text_base": CustomConfigConverter(
                fast_llm_paths=(
                    ("embeddings",),
                    ("decoder",),
                    ("head",),
                    ("hidden_size",),
                    ("tied_embedding_weight",),
                    ("peft",),
                ),
                hf_paths=tuple(text_base_cls._consumed_hf_paths()),
                export_fn=lambda c: {(k,): v for k, v in text_base_cls.export_config(c).items()},
                import_fn=lambda hf: {(k,): v for k, v in text_base_cls.import_config(hf).items()},
                fast_llm_recurses=True,
            ),
            # Optional vision encoder. The Fast-LLM ``vision_encoder`` field is architecture-hint and
            # ``None`` by default; the HF ``vision_encoder`` key is absent for text-only models.
            "vision_encoder": CustomConfigConverter(
                fast_llm_paths=(("vision_encoder",),),
                hf_paths=(("vision_encoder",),),
                export_fn=_vision_export,
                import_fn=_vision_import,
                fast_llm_recurses=True,
            ),
            # ``image_token_index`` is FieldHint.optional so it's not in the architecture-coverage set,
            # but it does live on the HF dict for vision-enabled checkpoints.
            "image_token_index": OptionalConfigConverter(("image_token_index",), ("image_token_index",)),
        }

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {
            # Vision encoder is optional — ``optional=True`` skips the recursion when
            # ``config.vision_encoder is None`` (text-only checkpoints).
            "vision_encoder": NestedWeightConverter(
                "vision_encoder", "", cls.vision_model_converter_class, optional=True
            ),
            # ``embeddings`` flat-merges into ``model``; vision_encoder writes to its own absolute prefix.
            "embeddings": NestedWeightConverter("embeddings", "model", cls.embeddings_converter_class),
            "decoder": BlockSequenceWeightConverter("decoder", "model.decoder.blocks", cls.block_converter_class),
            "head": NestedWeightConverter("head", "", cls.head_converter_class),
        }


class Apriel2HuggingfaceCheckpointHandler(HuggingfaceStateDictCheckpointHandler):
    _model: MultiModalModel
    _model_class: typing.ClassVar[FastLLMModelConfig] = MultiModalModelConfig
    format: typing.ClassVar[type[CheckpointFormat]] = Apriel2CheckpointFormat
    architecture: typing.ClassVar[str] = "Apriel2ForConditionalGeneration"
    base_model_converter_class: typing.ClassVar[type[Apriel2MultimodalBaseModelConverter]] = (
        Apriel2MultimodalBaseModelConverter
    )

    @classmethod
    def get_huggingface_model_type(cls) -> str:
        return "apriel2"

    @classmethod
    def get_transformers_configuration_class(cls):
        from fast_llm_external_models.apriel2.configuration_apriel2 import Apriel2Config

        return Apriel2Config

    @classmethod
    def get_model_files(cls) -> tuple[str, str, str | None]:
        from fast_llm_external_models.apriel2 import configuration_apriel2, modeling_apriel2

        return configuration_apriel2.__file__, modeling_apriel2.__file__, None

    @classmethod
    def _export_config(cls, config: MultiModalModelConfig) -> dict[str, typing.Any]:
        return {
            **cls.base_model_converter_class.export_config(config.base_model),
            "architectures": [cls.architecture],
            "model_type": cls.get_huggingface_model_type(),
            "auto_map": {
                "AutoConfig": "configuration_apriel2.Apriel2Config",
                "AutoModel": "modeling_apriel2.Apriel2Model",
                "AutoModelForCausalLM": "modeling_apriel2.Apriel2ForConditionalGeneration",
                "AutoModelForImageTextToText": "modeling_apriel2.Apriel2ForConditionalGeneration",
            },
        }

    @classmethod
    def _import_config(cls, config: dict[str, typing.Any]) -> dict[str, typing.Any]:
        cls._check_hf_coverage(config)
        return {"base_model": cls.base_model_converter_class.import_config(config)}
