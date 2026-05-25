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
    ImportOnlyConfigConverter,
    LinearWeightConverter,
    NestedConfigConverter,
    NestedWeightConverter,
    PatchEmbeddingWeightConverter,
    RenameConfigConverter,
    TransposeSplitWeightConverter,
    WeightConverter,
)
from fast_llm.engine.checkpoint.huggingface import HuggingFaceBaseModelConverter, HuggingfaceStateDictCheckpointHandler
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.functional.config import ActivationType
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.attention.rotary.config import Rotary2DConfig
from fast_llm.layers.decoder.mlp.config import MLPConfig
from fast_llm.layers.vision.config import PatchEmbeddingsConfig, VisionEncoderConfig
from fast_llm.models.gpt.conversion.llama import (
    _TRANSFORMERS_V4,
    LlamaAttentionConverter,
    LlamaBlockConverter,
    LlamaDecoderConverter,
    LlamaHeadConverter,
    LlamaNormalizationConverter,
)
from fast_llm.models.gpt.conversion.mistral import MistralBaseModelConverter, MistralMLPConverter
from fast_llm.models.multimodal.config import MultiModalBaseModelConfig, MultiModalModelConfig
from fast_llm.models.multimodal.conversion.config import LlavaCheckpointFormat
from fast_llm.models.multimodal.model import MultiModalModel
from fast_llm.utils import Assert, div


class PixtralNormalizationConverter(LlamaNormalizationConverter):
    """RMS norm with HF-side hardcoded epsilon=1e-5 (Pixtral's HF format omits the field)."""

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            **super()._create_config_converters(),
            # Pin epsilon to 1e-5: assert on export, inject on import. No HF write/read.
            "epsilon": ConstantImportConfigConverter(("epsilon",), 1e-5),
        }


def _pixtral_rotary_export(config: AttentionConfig) -> dict:
    if _TRANSFORMERS_V4:
        return {("rope_theta",): config.rotary.theta}
    return {("rope_parameters",): {"rope_theta": config.rotary.theta, "rope_type": "default"}}


def _pixtral_rotary_import(hf_dict: dict) -> dict:
    if "rope_parameters" in hf_dict:
        theta = hf_dict["rope_parameters"]["rope_theta"]
    else:
        theta = hf_dict["rope_theta"]
    return {("rotary",): {"type": "default_2d", "theta": theta}}


class PixtralAttentionConverter(LlamaAttentionConverter):
    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            **super()._create_config_converters(),
            # PixtralConfig hardcodes Q/K/V/O biases off and does not surface ``attention_bias``.
            "add_linear_biases": ConstantImportConfigConverter(("add_linear_biases",), False),
            # Pixtral attention is non-causal (vision encoder).
            "causal": ConstantImportConfigConverter(("causal",), False),
            # No GQA in Pixtral; ``head_groups`` derives from ``num_attention_heads`` on import and is redundant
            # on export (``_validate_export`` enforces equality with ``heads``).
            "head_groups": ImportOnlyConfigConverter(
                fast_llm_paths=(("head_groups",),),
                import_fn=lambda hf: {("head_groups",): hf["num_attention_heads"]},
            ),
            # Llava's PixtralVisionConfig has no ``head_dim`` field — it is derived as ``hidden_size //
            # num_attention_heads``. Don't emit head_dim on export (would otherwise need to be popped
            # downstream); on import, derive head_size from the same expression. Invariant validated by
            # :class:`LlavaVisionModelConverter._validate_export`, which has access to the parent's
            # ``hidden_size``.
            "head_size": ImportOnlyConfigConverter(
                fast_llm_paths=(("head_size",),),
                import_fn=lambda hf: {("head_size",): div(hf["hidden_size"], hf["num_attention_heads"])},
            ),
            # Pixtral always uses 2D rotary; only ``theta`` round-trips. The flat (v4) vs ``rope_parameters`` (v5)
            # layout follows the active transformers major version, mirroring the Llama parent.
            "rotary": CustomConfigConverter(
                fast_llm_paths=(("rotary",),),
                hf_paths=(("rope_theta",), ("rope_parameters",)),
                export_fn=_pixtral_rotary_export,
                import_fn=_pixtral_rotary_import,
                fast_llm_recurses=True,
            ),
        }

    @classmethod
    def _validate_export(cls, config: AttentionConfig) -> None:
        super()._validate_export(config)
        Assert.is_(type(config.rotary), Rotary2DConfig)
        Assert.eq(config.head_groups, config.heads)


class PixtralBlockConverter(LlamaBlockConverter):
    mixer_converter_class: typing.ClassVar[type[PixtralAttentionConverter]] = PixtralAttentionConverter
    mlp_converter_class: typing.ClassVar[type[MistralMLPConverter]] = MistralMLPConverter
    normalization_converter_class: typing.ClassVar[type[PixtralNormalizationConverter]] = PixtralNormalizationConverter
    hf_mixer_name: typing.ClassVar[str] = "attention"
    hf_mlp_name: typing.ClassVar[str] = "feed_forward"
    hf_norm_1_name: typing.ClassVar[str] = "attention_norm"
    hf_norm_2_name: typing.ClassVar[str] = "ffn_norm"


class PixtralEncoderConverter(LlamaDecoderConverter):
    block_converter_class: typing.ClassVar[type[PixtralBlockConverter]] = PixtralBlockConverter


class PixtralEmbeddingsConverter(ConfigSectionConverter):
    """Converts ``PatchEmbeddingsConfig`` ↔ Pixtral HF flat fields (``patch_size`` / ``num_channels``).

    Pixtral's HF ``vision_config`` carries a single ``patch_size`` field (height == width); the converter
    expands it to both Fast-LLM dimensions on import and validates equality on export.
    """

    fast_llm_config_class = PatchEmbeddingsConfig
    normalization_converter_class: typing.ClassVar[type[PixtralNormalizationConverter]] = PixtralNormalizationConverter

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "patch_height": RenameConfigConverter(("patch_height",), ("patch_size",)),
            # Pixtral has one `patch_size`; mirror it to `patch_width` on import and validate equality on export.
            "patch_width": ImportOnlyConfigConverter(
                fast_llm_paths=(("patch_width",),),
                import_fn=lambda hf: {("patch_width",): hf["patch_size"]},
            ),
            # `input_channels` is a derived cached_property pinned to 3; assert on import, emit on export.
            "num_channels": ConstantExportConfigConverter(("num_channels",), 3),
            # PixtralNormalizationConverter exports {} (epsilon pinned), so flat-merge is a no-op on export.
            "normalization": NestedConfigConverter(("normalization",), cls.normalization_converter_class),
            # patch_embeddings (the AffineLinearConfig) has no HF representation; bias presence validated below.
            "patch_embeddings": IgnoredConfigConverter(("patch_embeddings",)),
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
                "patch_conv",
                transform=PatchEmbeddingWeightConverter,
                bias_fn=False,
            ),
            # ``PixtralEmbeddingsConverter``'s section config IS the ``PatchEmbeddingsConfig`` (carries the
            # ``normalization`` sub-config directly), so the nested ``LlamaNormalizationConverter`` reads
            # ``getattr(section_config, "normalization")``.
            "normalization": NestedWeightConverter(
                "normalization", "ln_pre", cls.normalization_converter_class, config_attr="normalization"
            ),
        }


class LlavaVisionAdapterConverter(ConfigSectionConverter):
    """Converts the vision adapter :class:`MLPConfig` ↔ Llava's flat top-level adapter fields
    (``projector_hidden_act``, ``multimodal_projector_bias``).

    Wrinkle: the adapter's ``intermediate_size`` derives from the **text** half of the model
    (``text_config["hidden_size"]``). The cross-section reference is reachable because this converter is
    flat-merged at the :class:`LlavaBaseModelConverter` scope, where ``text_config`` lives as a sibling
    HF top-level key.
    """

    fast_llm_config_class = MLPConfig
    hf_type_name = "mlp"

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            # Cross-section: imported from text_config.hidden_size. No HF claim — text_config is claimed
            # by the language model converter at the base level.
            "intermediate_size": ImportOnlyConfigConverter(
                fast_llm_paths=(("intermediate_size",),),
                import_fn=lambda hf: {("intermediate_size",): hf["text_config"]["hidden_size"]},
            ),
            "add_linear_biases": RenameConfigConverter(("add_linear_biases",), ("multimodal_projector_bias",)),
            "gated": ConstantImportConfigConverter(("gated",), False),
            "activation": CustomConfigConverter(
                fast_llm_paths=(("activation",),),
                hf_paths=(("projector_hidden_act",),),
                export_fn=lambda c: {("projector_hidden_act",): c.activation.hf_name},
                import_fn=lambda hf: {("activation",): ActivationType.from_hf_name(hf["projector_hidden_act"])},
            ),
            # Per-layer ``bias.enabled`` has no HF representation; defaults round-trip. Validated below.
            "linear_layers": IgnoredConfigConverter(("layer_1",), ("layer_2",)),
            "pre_norm": ConstantImportConfigConverter(("pre_norm",), None),
            "post_norm": ConstantImportConfigConverter(("post_norm",), None),
        }

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


class LlavaVisionModelConverter(ConfigSectionConverter):
    """Converts :class:`VisionEncoderConfig` ↔ Llava's ``vision_config`` HF subdict.

    Declarations operate relative to ``vision_config`` (parent nests this converter via
    ``NestedConfigConverter(hf_path=("vision_config",))``). The adapter is *not* declared here — it
    lives at the base level because its Fast-LLM intermediate_size derives from text_config.hidden_size,
    a cross-section reference only visible at the top of the HF dict.
    """

    fast_llm_config_class = VisionEncoderConfig

    embeddings_converter_class: typing.ClassVar[type[PixtralEmbeddingsConverter]] = PixtralEmbeddingsConverter
    encoder_converter_class: typing.ClassVar[type[PixtralEncoderConverter]] = PixtralEncoderConverter
    model_type: typing.ClassVar[str] = "pixtral"

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            # Flat-merged into vision_config: embeddings (PatchEmbeddingsConverter writes patch_size/etc),
            # encoder (LlamaDecoderConverter, declarative — flat-merged Fixed sequence shape).
            "embeddings": NestedConfigConverter(("embeddings",), cls.embeddings_converter_class),
            "encoder": NestedConfigConverter(("encoder",), cls.encoder_converter_class),
            "hidden_size": RenameConfigConverter(("hidden_size",), ("hidden_size",)),
            # Llava's vision_config carries a literal ``model_type: "pixtral"``;
            # ``ConstantExportConfigConverter`` emits on export and asserts equality on import.
            "model_type": ConstantExportConfigConverter(("model_type",), cls.model_type),
            # ``transformers.LlavaConfig.from_dict(...).save_pretrained(...)`` round-trips the
            # vision_config through :class:`PixtralVisionConfig`, which fills in these default fields.
            # Fast-LLM does not consume them; mark them ignored so the recursive coverage check accepts
            # round-tripped saves. (``head_dim`` is normally not emitted because we override head_size to
            # ImportOnly, but transformers fills it from ``hidden_size // num_attention_heads`` on load.)
            "pixtral_hf_defaults": IgnoredConfigConverter(
                hf_paths=(
                    ("head_dim",),
                    ("image_size",),
                    ("initializer_factor",),
                    ("layer_norm_eps",),
                    ("projection_dim",),
                    ("vocab_size",),
                ),
            ),
            # Adapter is handled at LlavaBaseModelConverter scope (sees text_config). Mark recursively
            # consumed here so the architecture walker sees the sub-tree as claimed at this level too.
            "adapter": IgnoredConfigConverter(("adapter",)),
        }

    @classmethod
    def _validate_export(cls, config: VisionEncoderConfig) -> None:
        # Llava's PixtralVisionConfig does not carry head_dim — it is derived as ``hidden_size //
        # num_attention_heads``. Validate the Fast-LLM head_size satisfies this invariant.
        mixer = config.encoder.block.mixer
        if isinstance(mixer, AttentionConfig):
            Assert.eq(mixer.head_size * mixer.heads, config.hidden_size)

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {
            "embeddings": NestedWeightConverter("embeddings", "vision_tower", cls.embeddings_converter_class),
            # The encoder section IS a FixedBlockSequenceConfig — fan out blocks via the dedicated primitive.
            "encoder": NestedWeightConverter(
                "encoder", "vision_tower.transformer.layers", cls.encoder_converter_class
            ),
            "adapter": NestedWeightConverter("adapter", "multi_modal_projector", LlavaVisionAdapterConverter),
        }


class LlavaHeadConverter(LlamaHeadConverter):
    # Llava always emits a separate ``language_model.lm_head.weight`` declaration even when
    # ``tied_embedding_weight=True``, so the head uses a plain rename instead of
    # :class:`OutputProjectionWeightConverter` (which drops on export under the tied flag).
    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        return {
            "final_norm": NestedWeightConverter(
                "final_norm",
                "language_model.model.norm",
                cls.normalization_converter_class,
                config_attr="normalization",
            ),
            "output_weights": WeightConverter("output_weights", "language_model.lm_head.weight"),
        }


class LlavaLanguageModelConverter(MistralBaseModelConverter):
    head_converter_class: typing.ClassVar[type[LlavaHeadConverter]] = LlavaHeadConverter


class LlavaBaseModelConverter(ConfigSectionConverter, HuggingFaceBaseModelConverter):
    """Top-level converter for Llava. Composes:

    * ``text_config`` HF subdict ← :class:`LlavaLanguageModelConverter` (Mistral text base).
    * ``vision_config`` HF subdict ← :class:`LlavaVisionModelConverter` (Pixtral vision encoder).
    * Top-level adapter fields (``projector_hidden_act``, ``multimodal_projector_bias``) ←
      :class:`LlavaVisionAdapterConverter`, flat-merged because the adapter's ``intermediate_size``
      derives from ``text_config.hidden_size``.
    * Top-level multimodal metadata (``image_token_index``, ``vision_feature_select_strategy``,
      ``vision_feature_layer``).
    """

    fast_llm_config_class = MultiModalBaseModelConfig

    vision_model_converter_class: typing.ClassVar[type[LlavaVisionModelConverter]] = LlavaVisionModelConverter
    vision_adapter_converter_class: typing.ClassVar[type[LlavaVisionAdapterConverter]] = LlavaVisionAdapterConverter
    # TODO: Make it flexible?
    language_model_converter_class: typing.ClassVar[type[LlavaLanguageModelConverter]] = LlavaLanguageModelConverter
    # TODO: Is tie_word_embeddings supported?

    @classmethod
    def _create_config_converters(cls) -> dict:
        text_base_cls = cls.language_model_converter_class
        vision_cls = cls.vision_model_converter_class
        adapter_cls = cls.vision_adapter_converter_class

        # The Fast-LLM ``MultiModalBaseModelConfig`` IS-A ``GPTBaseModelConfig`` (multi-inherits via
        # ``VisionMultiModalModelConfig``), so ``text_base_cls.export_config(config)`` works directly on
        # the multimodal config: its declarations only touch GPTBaseModelConfig fields, which exist here.
        def _text_export(config: MultiModalBaseModelConfig) -> dict:
            return {("text_config",): text_base_cls.export_config(config)}

        def _text_import(hf_dict: dict) -> dict:
            return {(k,): v for k, v in text_base_cls.import_config(hf_dict["text_config"]).items()}

        return {
            "text_base": CustomConfigConverter(
                fast_llm_paths=(
                    ("embeddings",),
                    ("decoder",),
                    ("head",),
                    ("hidden_size",),
                    ("tied_embedding_weight",),
                    ("peft",),
                ),
                hf_paths=(("text_config",),),
                export_fn=_text_export,
                import_fn=_text_import,
                fast_llm_recurses=True,
            ),
            "vision_encoder": NestedConfigConverter(("vision_encoder",), vision_cls, hf_path=("vision_config",)),
            # Adapter flat-merged at top level: its import sees text_config.hidden_size as a sibling key.
            "adapter": NestedConfigConverter(("vision_encoder", "adapter"), adapter_cls, hf_path=None),
            "image_token_index": RenameConfigConverter(("image_token_index",), ("image_token_index",)),
            "vision_feature_select_strategy": ConstantExportConfigConverter(
                ("vision_feature_select_strategy",), "full"
            ),
            "vision_feature_layer": ConstantExportConfigConverter(("vision_feature_layer",), -1),
            # ``transformers.LlavaConfig.save_pretrained(...)`` round-trips the top-level config through
            # :class:`transformers.LlavaConfig`, which fills these defaults. Fast-LLM tracks
            # ``tie_word_embeddings`` *inside* text_config (Llama's tied_embedding_weight), not at the
            # Llava level; ``image_seq_length`` is a runtime/inference field, not architecture.
            "llava_hf_defaults": IgnoredConfigConverter(
                hf_paths=(("image_seq_length",), ("tie_word_embeddings",)),
            ),
        }

    @classmethod
    def _validate_export(cls, config: MultiModalBaseModelConfig) -> None:
        # Llava requires both a vision encoder and an image_token_index to be set.
        assert config.vision_encoder is not None, "Llava requires a vision encoder"
        assert config.image_token_index is not None, "Llava requires an image_token_index"

    @classmethod
    @functools.cache
    def _create_weight_converters(cls) -> dict[str, WeightConverter]:
        text_base_cls = cls.language_model_converter_class
        return {
            "vision_encoder": NestedWeightConverter("vision_encoder", "", cls.vision_model_converter_class),
            "embeddings": NestedWeightConverter(
                "embeddings", "language_model.model", text_base_cls.embeddings_converter_class
            ),
            "decoder": BlockSequenceWeightConverter(
                "decoder", "language_model.model.layers", text_base_cls.block_converter_class
            ),
            # ``LlavaHeadConverter``'s leaf converters use absolute HF paths (``language_model.lm_head.weight``,
            # ``language_model.model.norm``), so an empty ``hf_prefix`` lets them land verbatim.
            "head": NestedWeightConverter("head", "", text_base_cls.head_converter_class),
        }


class LlavaHuggingfaceCheckpointHandler(HuggingfaceStateDictCheckpointHandler):
    _model: MultiModalModel
    _model_class: typing.ClassVar[FastLLMModelConfig] = MultiModalModelConfig
    format: typing.ClassVar[type[CheckpointFormat]] = LlavaCheckpointFormat
    architecture: typing.ClassVar[str] = "LlavaForConditionalGeneration"
    base_model_converter_class: typing.ClassVar[type[LlavaBaseModelConverter]] = LlavaBaseModelConverter

    @classmethod
    def get_transformers_configuration_class(cls):
        import transformers

        return transformers.LlavaConfig
