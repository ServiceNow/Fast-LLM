import math
import typing

from transformers import PretrainedConfig

from fast_llm.config import Config, get_nested_dict_value
from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import (
    ConfigConverter,
    ConfigSectionConverter,
    CustomConfigConverter,
    DefaultConfigConverter,
    IgnoredConfigConverter,
    RenameConfigConverter,
    WeightConverter,
    _get_attr_path,
    _safe_set_nested_dict_value,
)
from fast_llm.layers.attention.config import AttentionConfig
from fast_llm.layers.block.config import FixedBlockSequenceConfig, PatternBlockSequenceConfig
from fast_llm.layers.decoder.config import DecoderBlockConfig
from fast_llm.layers.ssm.config import GatedDeltaNetConfig, KimiDeltaAttentionConfig, MambaConfig
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTModelConfig
from fast_llm.models.gpt.conversion.config import AprielHybridSSMCheckpointFormat
from fast_llm.models.gpt.conversion.llama import (
    effective_bias,
    get_parameter_converter,
    get_weight_and_bias_converters,
)
from fast_llm.models.gpt.conversion.mistral import (
    MistralBaseModelConverter,
    MistralBlockConverter,
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
            # ``ssm_cfg.expand`` is consumed only as a fallback input to ``d_inner``'s default; declared here
            # so the HF coverage walker accepts real Apriel Mamba configs that carry the key.
            "ssm_expand": IgnoredConfigConverter(hf_paths=(("ssm_cfg", "expand"),)),
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


class ListDispatchConfigConverter(ConfigConverter):
    """Block-sequence dispatch driven by a positional HF list.

    The HF config carries one layout-discriminator string per pipeline position; each registered
    block converter handles one mixer type, and their outputs flat-merge into the parent HF root
    (each claims a disjoint subset of top-level keys; shared scalars cross-validate via the safe-set
    guard). The Fast-LLM target is a :class:`FixedBlockSequenceConfig` when the layout has a single
    entry, a :class:`PatternBlockSequenceConfig` otherwise.

    Two registries:

    * ``registry``: mixer-config class → block ``ConfigSectionConverter``. Domain of the HF coverage
      walker — every block type whose HF keys must be claimed appears here.
    * ``layout_names``: mixer-config class → layout discriminator string. Must be a subset of
      ``registry`` keys (checked in ``__init__``). Domain of the config-side round-trip — classes
      absent here can't be config-exported (no layout name to emit) or config-imported (no reverse
      lookup).
    """

    fast_llm_recurses: typing.ClassVar[bool] = True

    def __init__(
        self,
        fast_llm_path: tuple[str, ...],
        hf_layout_path: tuple[str, ...],
        hf_num_blocks_path: tuple[str, ...],
        registry: "dict[type[Config], type[ConfigSectionConverter]]",
        layout_names: "dict[type[Config], str]",
    ):
        Assert.custom(lambda lns: set(lns) <= set(registry), layout_names)
        self.fast_llm_paths = (fast_llm_path,)
        # Layout/num-blocks paths and per-block claims register through ``_consumed_hf_paths``.
        self.hf_paths = ()
        self._hf_layout_path = hf_layout_path
        self._hf_num_blocks_path = hf_num_blocks_path
        self._registry = registry
        self._layout_names = layout_names
        self._config_classes = {name: cls for cls, name in layout_names.items()}

    def export_to(self, fast_llm_config: Config, hf_out: dict) -> None:
        block_sequence = _get_attr_path(fast_llm_config, self.fast_llm_paths[0])
        if type(block_sequence) is FixedBlockSequenceConfig:
            distinct_blocks = [block_sequence.block]
            pattern_blocks = [block_sequence.block]
        elif type(block_sequence) is PatternBlockSequenceConfig:
            distinct_blocks = block_sequence.blocks.values()
            pattern_blocks = [block_sequence.blocks[name] for name in block_sequence.pattern]
        else:
            raise NotImplementedError(type(block_sequence).__name__)
        for block_config in distinct_blocks:
            converter_class = self._registry[type(block_config.mixer)]
            sub_hf = converter_class.export_config(block_config)
            for key, value in sub_hf.items():
                _safe_set_nested_dict_value(hf_out, (key,), value)
        layout = [self._layout_names[type(block_config.mixer)] for block_config in pattern_blocks]
        _safe_set_nested_dict_value(hf_out, self._hf_layout_path, layout)
        _safe_set_nested_dict_value(hf_out, self._hf_num_blocks_path, block_sequence.num_blocks)

    def import_to(self, hf_dict: dict, fast_llm_out: dict) -> None:
        layout = get_nested_dict_value(hf_dict, self._hf_layout_path)
        num_blocks = get_nested_dict_value(hf_dict, self._hf_num_blocks_path)
        if len(layout) == 1:
            converter_class = self._registry[self._config_classes[layout[0]]]
            sub_fast_llm = {"block": converter_class.import_config(hf_dict), "num_blocks": num_blocks}
        else:
            sub_fast_llm = {
                "type": "pattern",
                "blocks": {
                    layout_name: self._registry[self._config_classes[layout_name]].import_config(hf_dict)
                    for layout_name in set(layout)
                },
                "pattern": layout,
                "num_blocks": num_blocks,
            }
        _safe_set_nested_dict_value(fast_llm_out, self.fast_llm_paths[0], sub_fast_llm)

    def _consumed_hf_paths(self) -> frozenset[tuple[str, ...]]:
        paths: set[tuple[str, ...]] = {self._hf_layout_path, self._hf_num_blocks_path}
        for sub_class in self._registry.values():
            paths |= sub_class._consumed_hf_paths()
        return frozenset(paths)


class AprielBlockConverter:
    """Per-mixer-type Apriel block-converter registries plus a weight-side dispatch helper.

    ``layout_names`` maps the mixer-config classes that participate in Apriel's
    ``hybrid_block_layout`` discriminator to their string layout names. ``_converter_classes``
    maps every mixer-config class whose weights can appear in an Apriel checkpoint to its block
    converter — a superset of ``layout_names`` keys that adds ``KimiDeltaAttentionConfig`` for
    weight-only coverage. :meth:`get_converters` picks the right block converter from
    ``_converter_classes`` by the mixer's runtime type.
    """

    layout_names: typing.ClassVar[dict[type[Config], str]] = {
        AttentionConfig: "t",
        MambaConfig: "m2",
        GatedDeltaNetConfig: "gdn",
    }
    _converter_classes: typing.ClassVar[dict[type[Config], type[ConfigSectionConverter]]] = {
        AttentionConfig: MistralBlockConverter,
        MambaConfig: AprielMambaBlockConverter,
        KimiDeltaAttentionConfig: AprielKimiDeltaAttentionBlockConverter,
        GatedDeltaNetConfig: AprielGatedDeltaNetBlockConverter,
    }

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


class AprielHeadConverter(MistralHeadConverter):
    block_converter_class: typing.ClassVar[type[AprielBlockConverter]] = AprielBlockConverter


class AprielBaseModelConverter(MistralBaseModelConverter):
    """Section converter for the Apriel hybrid-SSM base model.

    The decoder uses :class:`ListDispatchConfigConverter` driven by the format's
    ``hybrid_block_layout`` list (per-position mixer-type discriminator); each block converter's
    HF keys flat-merge into the parent HF root.
    """

    head_converter_class: typing.ClassVar[type[AprielHeadConverter]] = AprielHeadConverter
    # Distinct from the parent's ``block_converter_class`` (a single ``ConfigSectionConverter``); this
    # one holds the per-mixer-type dispatch registries that :class:`ListDispatchConfigConverter` and
    # the weight-side loop below consume.
    block_dispatcher_class: typing.ClassVar[type[AprielBlockConverter]] = AprielBlockConverter

    @classmethod
    def _create_config_converters(cls) -> dict:
        return {
            **super()._create_config_converters(),
            "decoder": ListDispatchConfigConverter(
                fast_llm_path=("decoder",),
                hf_layout_path=("hybrid_block_layout",),
                hf_num_blocks_path=("num_hidden_layers",),
                registry=cls.block_dispatcher_class._converter_classes,
                layout_names=cls.block_dispatcher_class.layout_names,
            ),
        }

    @classmethod
    def get_converters(cls, config: GPTBaseModelConfig, exported_config: dict) -> list[WeightConverter]:
        decoder = config.decoder
        converters = [*cls.embeddings_converter_class.get_converters(config.embeddings, "embeddings", "model")]
        for block_index in range(decoder.num_blocks):
            block_config = decoder.blocks[decoder.pattern[block_index % len(decoder.pattern)]]
            converters += cls.block_dispatcher_class.get_converters(
                block_config,
                f"decoder.{block_index}",
                f"model.layers.{block_index}",
            )
        converters += cls.head_converter_class.get_converters(config, exported_config)
        return converters


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
