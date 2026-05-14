import typing

import torch

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import (
    ConfigSectionConverter,
    ConstantExportConfigConverter,
    ConstantImportConfigConverter,
    CustomConfigConverter,
    IgnoredConfigConverter,
    ImportOnlyConfigConverter,
    NestedConfigConverter,
    RenameConfigConverter,
    WeightConverter,
)
from fast_llm.engine.checkpoint.huggingface import HuggingFaceBaseModelConverter, HuggingfaceStateDictCheckpointHandler
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.functional.config import ActivationType
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.attention.rotary.config import Rotary2DConfig
from fast_llm.layers.block.config import FixedBlockSequenceConfig
from fast_llm.layers.decoder.mlp.config import MLPConfig
from fast_llm.layers.language_model.config import LanguageModelConfig
from fast_llm.layers.vision.config import PatchEmbeddingsConfig, VisionEncoderConfig
from fast_llm.models.gpt.conversion.llama import (
    _TRANSFORMERS_V4,
    LlamaAttentionConverter,
    LlamaBlockConverter,
    LlamaDecoderConverter,
    LlamaNormalizationConverter,
    MLPLayer2Converter,
    get_parameter_converter,
    get_weight_and_bias_converters,
)
from fast_llm.models.gpt.conversion.mistral import MistralBaseModelConverter, MistralHeadConverter, MistralMLPConverter
from fast_llm.models.multimodal.config import MultiModalBaseModelConfig, MultiModalModelConfig
from fast_llm.models.multimodal.conversion.config import LlavaCheckpointFormat
from fast_llm.models.multimodal.model import MultiModalModel
from fast_llm.tensor import SafeTensorSlice
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
                recurses=True,
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


class PatchEmbeddingWeightConverter(WeightConverter):
    _config: PatchEmbeddingsConfig

    def export_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        return tuple(
            weight_[:].view(
                *weight_[:].shape[:-1],
                self._config.input_channels,
                self._config.patch_height,
                self._config.patch_width,
            )
            for weight_ in weight
        )

    def import_weight(
        self, weight: tuple[torch.Tensor | SafeTensorSlice, ...]
    ) -> tuple[torch.Tensor | SafeTensorSlice, ...]:
        return tuple(
            weight_[:].view(
                *weight_[:].shape[:-3],
                self._config.input_channels * self._config.patch_height * self._config.patch_width,
            )
            for weight_ in weight
        )


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
    def get_converters(
        cls, config: PatchEmbeddingsConfig, fast_llm_prefix: str, hf_prefix: str
    ) -> list[WeightConverter]:
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.patch_embeddings",
                f"{hf_prefix}.patch_conv",
                False,
                PatchEmbeddingWeightConverter,
                config,
            ),
            *cls.normalization_converter_class.get_converters(
                config, f"{fast_llm_prefix}.normalization", f"{hf_prefix}.ln_pre"
            ),
        ]


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
        }

    @classmethod
    def _validate_export(cls, config: MLPConfig) -> None:
        Assert.incl(config.layer_1.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.layer_2.bias.enabled, (None, config.add_linear_biases))

    # --- weight side (imperative) ---

    @classmethod
    def get_converters(cls, config: MLPConfig, fast_llm_prefix: str, hf_prefix: str) -> list[WeightConverter]:
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_1",
                f"{hf_prefix}.linear_1",
                config.add_linear_biases,
                WeightConverter,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.layer_2",
                f"{hf_prefix}.linear_2",
                config.add_linear_biases,
                MLPLayer2Converter,
            ),
        ]


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
        encoder_cls = cls.encoder_converter_class

        def _encoder_export(config: VisionEncoderConfig) -> dict:
            return {(k,): v for k, v in encoder_cls.export_config(config.encoder).items()}

        def _encoder_import(hf_dict: dict) -> dict:
            return {("encoder",): encoder_cls.import_config(hf_dict)}

        return {
            # Flat-merged into vision_config: embeddings (PatchEmbeddingsConverter writes patch_size/etc),
            # encoder (LlamaDecoderConverter dispatch — Custom-wrapped since it stays imperative).
            "embeddings": NestedConfigConverter(("embeddings",), cls.embeddings_converter_class),
            "encoder": CustomConfigConverter(
                fast_llm_paths=(("encoder",),),
                hf_paths=(
                    ("num_hidden_layers",),
                    *encoder_cls.block_converter_class._consumed_hf_paths(),
                ),
                export_fn=_encoder_export,
                import_fn=_encoder_import,
                recurses=True,
            ),
            "hidden_size": RenameConfigConverter(("hidden_size",), ("hidden_size",)),
            # Llava's vision_config carries a literal ``model_type: "pixtral"``;
            # ``ConstantExportConfigConverter`` emits on export and asserts equality on import.
            "model_type": ConstantExportConfigConverter(("model_type",), cls.model_type),
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

    # --- weight side (imperative) ---

    @classmethod
    def get_converters(cls, config: VisionEncoderConfig) -> list[WeightConverter]:
        return [
            *cls.embeddings_converter_class.get_converters(
                config.embeddings, "vision_encoder.embeddings", "vision_tower"
            ),
            *cls.encoder_converter_class.get_converters(
                config.encoder, "vision_encoder.encoder", "vision_tower.transformer.layers"
            ),
            *LlavaVisionAdapterConverter.get_converters(
                config.adapter, "vision_encoder.adapter", "multi_modal_projector"
            ),
        ]


class LlavaHeadConverter(MistralHeadConverter):
    @classmethod
    def get_converters(
        cls,
        config: LanguageModelConfig,
        exported_config: dict,
    ) -> list[WeightConverter]:
        return [
            *cls.normalization_converter_class.get_converters(
                config.head.normalization,
                f"head.final_norm",
                f"language_model.model.norm",
            ),
            get_parameter_converter(
                f"head.output_weights",
                "language_model.lm_head.weight",
                drop_on_import=exported_config["tie_word_embeddings"],
            ),
        ]


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
                recurses=True,
            ),
            "vision_encoder": NestedConfigConverter(("vision_encoder",), vision_cls, hf_path=("vision_config",)),
            # Adapter flat-merged at top level: its import sees text_config.hidden_size as a sibling key.
            "adapter": NestedConfigConverter(("vision_encoder", "adapter"), adapter_cls, hf_path=None),
            "image_token_index": RenameConfigConverter(("image_token_index",), ("image_token_index",)),
            "vision_feature_select_strategy": ConstantExportConfigConverter(
                ("vision_feature_select_strategy",), "full"
            ),
            "vision_feature_layer": ConstantExportConfigConverter(("vision_feature_layer",), -1),
        }

    @classmethod
    def _validate_export(cls, config: MultiModalBaseModelConfig) -> None:
        # Llava requires both a vision encoder and an image_token_index to be set.
        Assert.custom(lambda v: v is not None, config.vision_encoder)
        Assert.custom(lambda v: v is not None, config.image_token_index)

    # --- weight side (imperative) ---

    @classmethod
    def get_converters(cls, config: MultiModalBaseModelConfig, exported_config: dict) -> list[WeightConverter]:
        text_base_cls = cls.language_model_converter_class
        decoder_config = config.decoder
        block_config = (
            decoder_config.block
            if isinstance(decoder_config, FixedBlockSequenceConfig)
            else next(iter(decoder_config.blocks.values()))
        )
        block_converters: list[WeightConverter] = []
        for block_index in range(decoder_config.num_blocks):
            block_converters += text_base_cls.block_converter_class.get_converters(
                block_config, f"decoder.{block_index}", f"language_model.model.layers.{block_index}"
            )
        return [
            *cls.vision_model_converter_class.get_converters(config.vision_encoder),
            *text_base_cls.embeddings_converter_class.get_converters(
                config.embeddings, "embeddings", "language_model.model"
            ),
            *block_converters,
            *text_base_cls.head_converter_class.get_converters(config, {"tie_word_embeddings": False}),
        ]


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
