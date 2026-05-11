import math
import typing

from transformers import PretrainedConfig

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import (
    ConfigSectionConverter,
    CustomConfigConverter,
    DefaultConfigConverter,
    IgnoredConfigConverter,
    RenameConfigConverter,
    WeightConverter,
)
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.block.config import BlockSequenceConfig, FixedBlockSequenceConfig, PatternBlockSequenceConfig
from fast_llm.layers.decoder.config import DecoderBlockConfig
from fast_llm.layers.ssm.config import GatedDeltaNetConfig, KimiDeltaAttentionConfig, MambaConfig
from fast_llm.models.gpt.config import GPTModelConfig
from fast_llm.models.gpt.conversion.config import AprielHybridSSMCheckpointFormat
from fast_llm.models.gpt.conversion.llama import (
    effective_bias,
    get_parameter_converter,
    get_weight_and_bias_converters,
)
from fast_llm.models.gpt.conversion.mistral import (
    MistralBaseModelConverter,
    MistralBlockConverter,
    MistralDecoderConverter,
    MistralHeadConverter,
    MistralHuggingfaceCheckpointHandler,
)
from fast_llm.utils import Assert, safe_merge_dicts


class AprielMambaConverter(ConfigSectionConverter):
    """Converts ``MambaConfig`` ↔ Apriel hybrid SSM HF dict (``ssm_cfg`` subdict + root-level fallbacks).

    A few of MambaConfig's defaults are derived from the HF root's ``hidden_size`` (``d_inner`` defaults
    to ``hidden_size * expand``, ``d_xb`` defaults to ``hidden_size``, ``dt_rank="auto"`` resolves to
    ``ceil(hidden_size / 16)``). Those declarations read the root HF dict directly, so each leaf
    converter sees the full HF root passed by the parent block dispatcher.
    """

    fast_llm_config_class = MambaConfig

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "state_size": RenameConfigConverter(("state_size",), ("ssm_cfg", "d_state")),
            "d_inner": DefaultConfigConverter(
                ("d_inner",),
                ("ssm_cfg", "d_inner"),
                hf_default_fn=lambda hf: hf["hidden_size"] * hf.get("ssm_cfg", {}).get("expand", 1),
            ),
            "d_xb": DefaultConfigConverter(
                ("d_xb",),
                ("ssm_cfg", "d_xb"),
                hf_default_fn=lambda hf: hf["hidden_size"],
            ),
            "dt_rank": CustomConfigConverter(
                fast_llm_paths=(("dt_rank",),),
                hf_paths=(("ssm_cfg", "dt_rank"),),
                export_fn=lambda c: {("ssm_cfg", "dt_rank"): c.dt_rank},
                import_fn=lambda hf: {
                    ("dt_rank",): (
                        math.ceil(hf["hidden_size"] / 16)
                        if hf.get("ssm_cfg", {}).get("dt_rank", "auto") == "auto"
                        else hf["ssm_cfg"]["dt_rank"]
                    )
                },
            ),
            "add_linear_biases": RenameConfigConverter(("add_linear_biases",), ("ssm_cfg", "bias")),
            "repeat_kv_before_conv": DefaultConfigConverter(
                ("repeat_kv_before_conv",),
                ("ssm_cfg", "repeat_kv_before_conv"),
                hf_default_fn=lambda hf: True,
            ),
            "convolution_layer_bias": CustomConfigConverter(
                fast_llm_paths=(("convolution_layer",), ("convolution_layer", "bias")),
                hf_paths=(("ssm_cfg", "conv_bias"),),
                export_fn=lambda c: {
                    ("ssm_cfg", "conv_bias"): effective_bias(c.convolution_layer, c.add_linear_biases)
                },
                import_fn=lambda hf: {
                    ("convolution_layer", "bias", "enabled"): hf.get("ssm_cfg", {}).get("conv_bias", True)
                },
            ),
            # CausalConv1dConfig fields not represented in Apriel HF: weight rides the tensor side via
            # ``conv1d.weight``; kernel_size/activation round-trip implicitly at the Fast-LLM defaults.
            "convolution_layer_unmapped": IgnoredConfigConverter(
                ("convolution_layer", "weight"),
                ("convolution_layer", "kernel_size"),
                ("convolution_layer", "activation"),
            ),
            "dt_layer_bias": CustomConfigConverter(
                fast_llm_paths=(("dt_layer",), ("dt_layer", "bias")),
                hf_paths=(("ssm_cfg", "dt_proj_bias"),),
                export_fn=lambda c: {("ssm_cfg", "dt_proj_bias"): effective_bias(c.dt_layer, c.add_linear_biases)},
                import_fn=lambda hf: {
                    ("dt_layer", "bias", "enabled"): hf.get("ssm_cfg", {}).get("dt_proj_bias", True)
                },
            ),
            # AffineLinearConfig.weight rides the tensor side via ``dt_proj.weight``.
            "dt_layer_unmapped": IgnoredConfigConverter(("dt_layer", "weight")),
            # Per-layer biases that must round-trip implicitly via add_linear_biases (validated below).
            "linear_layers": IgnoredConfigConverter(
                ("z_layer",),
                ("x_layer",),
                ("b_layer",),
                ("c_layer",),
                ("output_layer",),
                ("dt_input_layer",),
            ),
            # Parameter sub-configs Mamba doesn't expose to HF; coverage-only.
            "parameters": IgnoredConfigConverter(("d_weight",), ("a_log_weight",)),
        }

    @classmethod
    def _validate_export(cls, config: MambaConfig) -> None:
        Assert.incl(config.z_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.x_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.b_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.c_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.dt_input_layer.bias.enabled, (None, config.add_linear_biases))
        Assert.incl(config.output_layer.bias.enabled, (None, config.add_linear_biases))

    @classmethod
    def import_config(cls, hf_dict: dict) -> dict:
        # Inject the Fast-LLM dynamic-type discriminator: the parent (AprielBlockConverter) selects this
        # leaf via `hybrid_block_layout`, not via a nested HF discriminator, so DispatchConfigConverter's
        # auto-injection isn't in play and we must add `type` manually.
        return {"type": "mamba", **super().import_config(hf_dict)}

    @classmethod
    def get_converters(
        cls,
        config: MambaConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        return [
            # TODO: Conv
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.in_proj",
                f"{hf_prefix}.in_proj",
                config.add_linear_biases,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.dt_in_proj",
                f"{hf_prefix}.dt_in_proj",
                config.add_linear_biases,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.dt_proj",
                f"{hf_prefix}.dt_proj",
                effective_bias(config.dt_layer, config.add_linear_biases),
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.convolution",
                f"{hf_prefix}.conv1d",
                effective_bias(config.convolution_layer, config.add_linear_biases),
                drop_on_export=drop_on_export,
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.A_log",
                f"{hf_prefix}.A_log",
                drop_on_export=drop_on_export,
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.D",
                f"{hf_prefix}.D",
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.out_proj",
                f"{hf_prefix}.out_proj",
                config.add_linear_biases,
                drop_on_export=drop_on_export,
            ),
        ]


class GatedDeltaNetConverter(ConfigSectionConverter):
    """Converts ``GatedDeltaNetConfig`` ↔ Apriel HF ``linear_attn_config`` subdict."""

    fast_llm_config_class = GatedDeltaNetConfig

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "value_heads": RenameConfigConverter(("value_heads",), ("linear_attn_config", "gdn_num_value_heads")),
            "key_heads": RenameConfigConverter(("key_heads",), ("linear_attn_config", "gdn_num_key_heads")),
            "key_head_dim": RenameConfigConverter(("key_head_dim",), ("linear_attn_config", "gdn_key_head_dim")),
            "value_head_dim": RenameConfigConverter(("value_head_dim",), ("linear_attn_config", "gdn_value_head_dim")),
            "convolution_layer_kernel": CustomConfigConverter(
                fast_llm_paths=(("convolution_layer",), ("convolution_layer", "kernel_size")),
                hf_paths=(("linear_attn_config", "gdn_linear_conv_kernel_size"),),
                export_fn=lambda c: {
                    ("linear_attn_config", "gdn_linear_conv_kernel_size"): c.convolution_layer.kernel_size
                },
                import_fn=lambda hf: {
                    ("convolution_layer", "kernel_size"): hf["linear_attn_config"]["gdn_linear_conv_kernel_size"]
                },
            ),
            "convolution_layer_unmapped": IgnoredConfigConverter(
                ("convolution_layer", "weight"),
                ("convolution_layer", "bias"),
                ("convolution_layer", "activation"),
            ),
            # Sub-configs without HF representation; coverage-only.
            "sub_configs": IgnoredConfigConverter(
                ("normalization",),
                ("qkv_projection_layer",),
                ("ba_projection_layer",),
                ("output_layer",),
                ("dt_bias_weight",),
                ("a_log_weight",),
            ),
        }

    @classmethod
    def import_config(cls, hf_dict: dict) -> dict:
        return {"type": "gdn", **super().import_config(hf_dict)}

    @classmethod
    def get_converters(
        cls,
        config: GatedDeltaNetConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.in_proj_qkvz",
                f"{hf_prefix}.in_proj_qkvz",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.in_proj_ba",
                f"{hf_prefix}.in_proj_ba",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.convolution",
                f"{hf_prefix}.convolution",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.out_proj",
                f"{hf_prefix}.out_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.A_log",
                f"{hf_prefix}.A_log",
                drop_on_export=drop_on_export,
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.dt_bias",
                f"{hf_prefix}.dt_bias",
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.norm",
                f"{hf_prefix}.norm",
                False,
                drop_on_export=drop_on_export,
            ),
        ]


class KimiDeltaAttentionConverter(ConfigSectionConverter):
    """Converts ``KimiDeltaAttentionConfig`` ↔ Apriel HF ``linear_attn_config`` subdict."""

    fast_llm_config_class = KimiDeltaAttentionConfig

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            "head_dim": RenameConfigConverter(("head_dim",), ("linear_attn_config", "head_dim")),
            "heads": RenameConfigConverter(("heads",), ("linear_attn_config", "num_heads")),
            "convolution_layer_kernel": CustomConfigConverter(
                fast_llm_paths=(("convolution_layer",), ("convolution_layer", "kernel_size")),
                hf_paths=(("linear_attn_config", "short_conv_kernel_size"),),
                export_fn=lambda c: {
                    ("linear_attn_config", "short_conv_kernel_size"): c.convolution_layer.kernel_size
                },
                import_fn=lambda hf: {
                    ("convolution_layer", "kernel_size"): hf["linear_attn_config"]["short_conv_kernel_size"]
                },
            ),
            "convolution_layer_unmapped": IgnoredConfigConverter(
                ("convolution_layer", "weight"),
                ("convolution_layer", "bias"),
                ("convolution_layer", "activation"),
            ),
            # Sub-configs without HF representation; coverage-only.
            "sub_configs": IgnoredConfigConverter(
                ("normalization",),
                ("q_projection_layer",),
                ("k_projection_layer",),
                ("v_projection_layer",),
                ("f_a_projection_layer",),
                ("f_b_projection_layer",),
                ("g_a_projection_layer",),
                ("g_b_projection_layer",),
                ("beta_projection_layer",),
                ("output_projection_layer",),
                ("dt_bias_weight",),
                ("a_log_weight",),
            ),
        }

    @classmethod
    def import_config(cls, hf_dict: dict) -> dict:
        return {"type": "kda", **super().import_config(hf_dict)}

    @classmethod
    def get_converters(
        cls,
        config: KimiDeltaAttentionConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        return [
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.q_proj",
                f"{hf_prefix}.q_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.k_proj",
                f"{hf_prefix}.k_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.v_proj",
                f"{hf_prefix}.v_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.q_conv",
                f"{hf_prefix}.q_conv",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.k_conv",
                f"{hf_prefix}.k_conv",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.v_conv",
                f"{hf_prefix}.v_conv",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.f_a_proj",
                f"{hf_prefix}.f_a_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.f_b_proj",
                f"{hf_prefix}.f_b_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.g_a_proj",
                f"{hf_prefix}.g_a_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.g_b_proj",
                f"{hf_prefix}.g_b_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.beta_proj",
                f"{hf_prefix}.beta_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.o_proj",
                f"{hf_prefix}.o_proj",
                False,
                drop_on_export=drop_on_export,
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.A_log",
                f"{hf_prefix}.A_log",
                drop_on_export=drop_on_export,
            ),
            get_parameter_converter(
                f"{fast_llm_prefix}.dt_bias",
                f"{hf_prefix}.dt_bias",
                drop_on_export=drop_on_export,
            ),
            *get_weight_and_bias_converters(
                f"{fast_llm_prefix}.norm",
                f"{hf_prefix}.norm",
                False,
                drop_on_export=drop_on_export,
            ),
        ]


class AprielKimiDeltaAttentionBlockConverter(MistralBlockConverter):
    mixer_converter_class: typing.ClassVar[type[KimiDeltaAttentionConverter]] = KimiDeltaAttentionConverter
    hf_mixer_name: typing.ClassVar[str] = "mixer"


class AprielMambaBlockConverter(MistralBlockConverter):
    mixer_converter_class: typing.ClassVar[type[AprielMambaConverter]] = AprielMambaConverter
    hf_mixer_name: typing.ClassVar[str] = "mixer"


class AprielGatedDeltaNetBlockConverter(MistralBlockConverter):
    mixer_converter_class: typing.ClassVar[type[GatedDeltaNetConverter]] = GatedDeltaNetConverter
    hf_mixer_name: typing.ClassVar[str] = "mixer"


class AprielBlockConverter:
    """Per-block dispatcher: the mixer type is encoded in the parent's ``hybrid_block_layout`` list,
    not in a nested HF discriminator, so this dispatcher stays imperative rather than using
    :class:`DispatchConfigConverter`. Each branch delegates to a regular declarative block converter.
    """

    layout_names = {
        AttentionConfig: "t",
        MambaConfig: "m2",
        GatedDeltaNetConfig: "gdn",
    }
    _converter_classes = {
        AttentionConfig: MistralBlockConverter,
        MambaConfig: AprielMambaBlockConverter,
        KimiDeltaAttentionConfig: AprielKimiDeltaAttentionBlockConverter,
        GatedDeltaNetConfig: AprielGatedDeltaNetBlockConverter,
    }
    _config_classes = {value: key for key, value in layout_names.items()}

    @classmethod
    def import_config(cls, config: dict, layout_name: str = "t") -> dict:
        return cls._converter_classes[cls._config_classes[layout_name]].import_config(config)

    @classmethod
    def export_config(cls, config) -> dict:
        return cls._converter_classes[type(config.mixer)].export_config(config)

    @classmethod
    def _consumed_hf_paths(cls) -> frozenset[tuple[str, ...]]:
        """Union of consumed HF paths across every per-mixer-type block converter — used by the parent's
        decoder Custom to pre-claim Apriel's flat top-level keys for the HF coverage check."""
        paths: set[tuple[str, ...]] = set()
        for sub in cls._converter_classes.values():
            paths |= sub._consumed_hf_paths()
        return frozenset(paths)

    @classmethod
    def get_converters(
        cls,
        config: DecoderBlockConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        return cls._converter_classes[type(config.mixer)].get_converters(
            config, fast_llm_prefix, hf_prefix, drop_on_export=drop_on_export
        )


class AprielDecoderConverter(MistralDecoderConverter):
    """Pattern-style decoder dispatched via Apriel's ``hybrid_block_layout`` list (one entry per block).
    Stays imperative because the layout-list shape doesn't match the declarative ``decoder.type``
    discriminator that Apriel2 uses.
    """

    block_converter_class: typing.ClassVar[type[AprielBlockConverter]] = AprielBlockConverter

    @classmethod
    def import_config(cls, config: dict) -> dict:
        layout = config["hybrid_block_layout"]
        if len(layout) == 1:
            return {
                "block": cls.block_converter_class.import_config(config, layout[0]),
                "num_blocks": config["num_hidden_layers"],
            }
        else:
            return {
                "type": "pattern",
                "blocks": {
                    layout_name: cls.block_converter_class.import_config(config, layout_name)
                    for layout_name in set(layout)
                },
                "pattern": layout,
                "num_blocks": config["num_hidden_layers"],
            }

    @classmethod
    def export_config(cls, config: BlockSequenceConfig) -> dict:
        if type(config) is FixedBlockSequenceConfig:
            block_configs = [config.block]
            pattern_block_configs = [config.block]
        elif type(config) is PatternBlockSequenceConfig:
            block_configs = config.blocks.values()
            pattern_block_configs = [config.blocks[block_name] for block_name in config.pattern]
        else:
            raise NotImplementedError()
        # Each block emits non-overlapping HF keys (attention -> flat, mamba -> ssm_cfg.*,
        # gdn/kda -> linear_attn_config.*) so safe_merge_dicts is sufficient to combine them.
        return safe_merge_dicts(
            *[cls.block_converter_class.export_config(block_config) for block_config in block_configs],
            {
                "num_hidden_layers": config.num_blocks,
                "hybrid_block_layout": [
                    cls.block_converter_class.layout_names[type(block_config.mixer)]
                    for block_config in pattern_block_configs
                ],
            },
        )

    @classmethod
    def get_converters(
        cls,
        config: PatternBlockSequenceConfig,
        fast_llm_prefix: str,
        hf_prefix: str,
        drop_on_export: bool = False,
    ) -> list[WeightConverter]:
        converters = []
        for block_index in range(config.num_blocks):
            block_config = config.blocks[config.pattern[block_index % len(config.pattern)]]
            converters += cls.block_converter_class.get_converters(
                block_config,
                f"{fast_llm_prefix}.{block_index}",
                f"{hf_prefix}.{block_index}",
                drop_on_export,
            )
        return converters


class AprielHeadConverter(MistralHeadConverter):
    block_converter_class: typing.ClassVar[type[AprielBlockConverter]] = AprielBlockConverter


class AprielBaseModelConverter(MistralBaseModelConverter):
    decoder_converter_class: typing.ClassVar[type[AprielDecoderConverter]] = AprielDecoderConverter
    head_converter_class: typing.ClassVar[type[AprielHeadConverter]] = AprielHeadConverter


class AprielHuggingfaceCheckpointHandler(MistralHuggingfaceCheckpointHandler):
    format: typing.ClassVar[type[CheckpointFormat]] = AprielHybridSSMCheckpointFormat
    architecture: typing.ClassVar[str] = "AprielHybridSSMForCausalLM"
    base_model_converter_class: typing.ClassVar[type[AprielBaseModelConverter]] = AprielBaseModelConverter

    @classmethod
    def get_transformers_configuration_class(cls) -> type[PretrainedConfig]:
        from fast_llm_external_models.apriel_hybrid_ssm.configuration_apriel_hybrid_ssm import AprielHybridSSMConfig

        return AprielHybridSSMConfig

    @classmethod
    def get_model_files(cls) -> tuple[str, str, str | None]:
        from fast_llm_external_models.apriel_hybrid_ssm import (
            configuration_apriel_hybrid_ssm,
            modeling_apriel_hybrid_ssm,
        )

        return configuration_apriel_hybrid_ssm.__file__, modeling_apriel_hybrid_ssm.__file__, None

    @classmethod
    def _export_config(cls, config: GPTModelConfig) -> dict[str, typing.Any]:
        return safe_merge_dicts(
            super()._export_config(config),
            {
                "auto_map": {
                    "AutoConfig": "configuration_apriel_hybrid_ssm.AprielHybridSSMConfig",
                    "AutoModel": "modeling_apriel_hybrid_ssm.AprielHybridSSMModel",
                    "AutoModelForCausalLM": "modeling_apriel_hybrid_ssm.AprielHybridSSMForCausalLM",
                },
            },
        )
