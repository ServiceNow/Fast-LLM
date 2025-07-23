import json
import os
import pathlib
import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import (
    ConstantExportParamConverter,
    ConstantImportParamConverter,
    IgnoreImportParamConverter,
    IgnoreImportWeightConverter,
    MappedConfigParamConverter,
    ParamConverter,
    RenameParamConverter,
    SplitWeightConverter,
    WeightConverter,
)
from fast_llm.engine.checkpoint.huggingface import HuggingfaceStateDictCheckpointHandler
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.functional.config import ActivationType
from fast_llm.layers.common.config import RMSNormalizationConfig
from fast_llm.layers.ssm.config import SSMBlockType
from fast_llm.models.gpt.conversion import CommonLlamaHuggingfaceCheckpointHandler, MLPLayer2Converter
from fast_llm.models.ssm.config import (
    AprielSSMHHybridHuggingfaceCheckpointFormat,
    AprielSSMHuggingfaceCheckpointFormat,
    AprielThinkerSSMHHybridHuggingfaceCheckpointFormat,
    HybridSSMModelConfig,
    LLambaHuggingfaceCheckpointFormat,
)
from fast_llm.models.ssm.model import HybridSSMModel
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    pass


class HybridModelCheckpointHandler(HuggingfaceStateDictCheckpointHandler):
    _model: HybridSSMModel
    _model_class: typing.ClassVar[FastLLMModelConfig] = HybridSSMModelConfig
    _default_block_type: str = SSMBlockType.mamba2_discrete.value

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        block_converter = RenameParamConverter(
            fast_llm_names=(("hybrid_block_layout",),),
            export_names=(("hybrid_block_layout",),),
            ignore_missing=True,
            default_value=[cls._default_block_type],
        )
        return super()._create_config_converters() + [block_converter]


class CommonSSMHuggingfaceCheckpointHandler(HuggingfaceStateDictCheckpointHandler):
    _model: HybridSSMModel
    _model_class: typing.ClassVar[FastLLMModelConfig] = HybridSSMModelConfig

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            RenameParamConverter(
                fast_llm_names=(("ssm", "state_size"),),
                export_names=(
                    (
                        "ssm_cfg",
                        "d_state",
                    ),
                ),
            ),
            RenameParamConverter(
                fast_llm_names=(("ssm", "n_v_heads"),),
                export_names=(
                    (
                        "ssm_cfg",
                        "n_v_heads",
                    ),
                ),
            ),
            RenameParamConverter(
                fast_llm_names=(("ssm", "n_qk_heads"),),
                export_names=(
                    (
                        "ssm_cfg",
                        "n_qk_heads",
                    ),
                ),
            ),
            RenameParamConverter(
                fast_llm_names=(("ssm", "expansion_factor"),),
                export_names=(
                    (
                        "ssm_cfg",
                        "expand",
                    ),
                ),
            ),
            RenameParamConverter(
                fast_llm_names=(("ssm", "chunk_size"),),
                export_names=(
                    (
                        "ssm_cfg",
                        "chunk_size",
                    ),
                ),
            ),
            RenameParamConverter(
                fast_llm_names=(("ssm", "add_bias_linear"),),
                export_names=(
                    (
                        "ssm_cfg",
                        "bias",
                    ),
                ),
            ),
            MappedConfigParamConverter(
                fast_llm_names=(("ssm", "activation_type"),),
                export_names=(
                    (
                        "ssm_cfg",
                        "activation",
                    ),
                ),
                fast_llm_value=ActivationType.from_hf_name,
                export_value=lambda activation_type: activation_type.hf_name,
            ),
            # ================================================
            # Mamba2 specific parameters: they dont exist in old checkpoints exported for discrete Mamba2, hence need backward compatibility
            RenameParamConverter(
                fast_llm_names=(("ssm", "dt_rank"),),
                export_names=(
                    (
                        "ssm_cfg",
                        "dt_rank",
                    ),
                ),
                ignore_missing=True,
                default_value=None,
            ),
            RenameParamConverter(
                fast_llm_names=(("ssm", "dt_min"),),
                export_names=(
                    (
                        "ssm_cfg",
                        "dt_min",
                    ),
                ),
                ignore_missing=True,
                default_value=0.001,
            ),
            RenameParamConverter(
                fast_llm_names=(("ssm", "dt_max"),),
                export_names=(
                    (
                        "ssm_cfg",
                        "dt_max",
                    ),
                ),
                ignore_missing=True,
                default_value=0.1,
            ),
            RenameParamConverter(
                fast_llm_names=(("ssm", "dt_init_floor"),),
                export_names=(
                    (
                        "ssm_cfg",
                        "dt_init_floor",
                    ),
                ),
                ignore_missing=True,
                default_value=1e-4,
            ),
            RenameParamConverter(
                fast_llm_names=(("ssm", "dt_scale"),),
                export_names=(
                    (
                        "ssm_cfg",
                        "dt_scale",
                    ),
                ),
                ignore_missing=True,
                default_value=1.0,
            ),
            RenameParamConverter(
                fast_llm_names=(("ssm", "d_xb"),),
                export_names=(
                    (
                        "ssm_cfg",
                        "d_xb",
                    ),
                ),
                ignore_missing=True,
                default_value=None,
            ),
            RenameParamConverter(
                fast_llm_names=(("ssm", "conv_kernel_dimension"),),
                export_names=(
                    (
                        "ssm_cfg",
                        "d_conv",
                    ),
                ),
                ignore_missing=True,
                default_value=4,
            ),
            RenameParamConverter(
                fast_llm_names=(("ssm", "dt_init"),),
                export_names=(
                    (
                        "ssm_cfg",
                        "dt_init",
                    ),
                ),
                ignore_missing=True,
                default_value="random",
            ),
        ]

    def _create_weight_converters(self) -> list[WeightConverter]:
        converters = super()._create_weight_converters() or []

        num_layers = self._model.config.base_model.transformer.num_layers
        ssm_bias: bool = self._model.config.base_model.ssm.add_bias_linear

        for i in range(num_layers):
            # SSM
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+1}.mixer.in_proj", f"model.layers.{i}.mixer.in_proj", ssm_bias
            )
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+1}.mixer.out_proj", f"model.layers.{i}.mixer.out_proj", ssm_bias
            )
            converters.append(
                WeightConverter(f"layers.{i+1}.mixer.D", f"model.layers.{i}.mixer.D", self._model.config.base_model)
            )
            converters.append(
                WeightConverter(
                    f"layers.{i+1}.mixer.z_bias", f"model.layers.{i}.mixer.z_bias", self._model.config.base_model
                )
            )
            converters.append(
                WeightConverter(
                    f"layers.{i+1}.mixer.z_bias", f"model.layers.{i}.mixer.z_bias", self._model.config.base_model
                )
            )
            converters.append(
                WeightConverter(
                    f"layers.{i+1}.mixer.conv1d_weight",
                    f"model.layers.{i}.mixer.conv1d.weight",
                    self._model.config.base_model,
                )
            )
            converters.append(
                WeightConverter(
                    f"layers.{i+1}.mixer.conv1d_bias",
                    f"model.layers.{i}.mixer.conv1d.bias",
                    self._model.config.base_model,
                )
            )
            # ================================================
            # Mamba2 specific parameters
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+1}.mixer.dt_proj", f"model.layers.{i}.mixer.dt_proj", False
            )
            # bias is treated separately in Mamba2 and must always exist (https://github.com/jxiw/M1/blob/537a1ca5407a786a99dc6c721873493cf8750d5e/mamba/hybrid_mamba_layer.py)
            converters.append(
                WeightConverter(
                    f"layers.{i+1}.mixer.dt_proj_bias",
                    f"model.layers.{i}.mixer.dt_proj.bias",
                    self._model.config.base_model,
                )
            )

            converters.append(
                WeightConverter(
                    f"layers.{i+1}.mixer.A_log", f"model.layers.{i}.mixer.A_log", self._model.config.base_model
                )
            )

        return converters

    def _get_weight_and_bias_converters(
        self,
        fast_llm_prefix: str | tuple[str, ...],
        hf_prefix: str | tuple[str, ...],
        use_bias: bool,
        cls=WeightConverter,
    ) -> list[WeightConverter]:
        if isinstance(fast_llm_prefix, str):
            fast_llm_prefix = (fast_llm_prefix,)
        if isinstance(hf_prefix, str):
            hf_prefix = (hf_prefix,)
        converters = [
            cls(
                tuple(f"{prefix}.weight" for prefix in fast_llm_prefix),
                tuple(f"{prefix}.weight" for prefix in hf_prefix),
                self._model.config.base_model,
            )
        ]
        if use_bias:
            converters.append(
                cls(
                    tuple(f"{prefix}.bias" for prefix in fast_llm_prefix),
                    tuple(f"{prefix}.bias" for prefix in hf_prefix),
                    self._model.config.base_model,
                )
            )
        return converters


class LLambaHuggingfaceCheckpointHandler(CommonSSMHuggingfaceCheckpointHandler):
    _model: HybridSSMModel
    _model_class: typing.ClassVar[FastLLMModelConfig] = HybridSSMModelConfig
    format: typing.ClassVar[type[CheckpointFormat]] = LLambaHuggingfaceCheckpointFormat
    _hf_prefix: str = "backbone"
    architecture: typing.ClassVar[str] = "LlambaForCausalLM"

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        """
        Create config converters for the model, see args under https://huggingface.co/cartesia-ai/Llamba-8B/blob/main/config.json
        """
        return super()._create_config_converters() + [
            RenameParamConverter(
                fast_llm_names=(("vocab_size",),),
                export_names=(("vocab_size",),),
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "normalization", "epsilon"),), export_names=(("norm_epsilon",),)
            ),
            ConstantImportParamConverter(fast_llm_names=(("use_position_embeddings",),), fast_llm_value=False),
            RenameParamConverter(
                fast_llm_names=(("transformer", "num_layers"),),
                export_names=(("n_layer",),),
            ),
            ConstantImportParamConverter(fast_llm_names=(("transformer", "gated"),), fast_llm_value=True),
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "normalization", "type"),),
                fast_llm_value=RMSNormalizationConfig.dynamic_type_name,
            ),
            MappedConfigParamConverter(
                fast_llm_names=(("transformer", "activation_type"),),
                export_names=(
                    (
                        "mlp_cfg",
                        "act_fn",
                    ),
                ),
                fast_llm_value=ActivationType.from_hf_name,
                export_value=lambda activation_type: activation_type.hf_name,
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "add_linear_biases"),),
                export_names=(
                    (
                        "mlp_cfg",
                        "bias",
                    ),
                ),
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "ffn_hidden_size"),),
                export_names=(
                    (
                        "mlp_cfg",
                        "intermediate_size",
                    ),
                ),
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "hidden_size"),),
                export_names=(("d_model",),),
            ),
            RenameParamConverter(
                fast_llm_names=(("tie_word_embeddings",),),
                export_names=(("tie_embeddings",),),
            ),
        ]

    def _create_weight_converters(self) -> list[WeightConverter]:
        # not using super() because LLamba model is called backbone in the checkpoints
        converters = []
        num_layers = self._model.config.base_model.transformer.num_layers
        norm_bias: bool = False
        ssm_bias: bool = self._model.config.base_model.ssm.add_bias_linear

        # Embedding and output
        if self._model.config.base_model.tie_word_embeddings:
            converters.append(
                WeightConverter("layers.0.word_embeddings_weight", f"{self._hf_prefix}.embedding.weight")
            )
            converters.append(IgnoreImportWeightConverter((), f"{self._hf_prefix}.lm_head.weight"))
        else:
            converters.append(
                WeightConverter("layers.0.word_embeddings_weight", f"{self._hf_prefix}.embedding.weight")
            )
            converters.append(
                WeightConverter(f"layers.{num_layers + 1}.output_weights", f"{self._hf_prefix}.lm_head.weight")
            )

        # Final norm
        converters += self._get_weight_and_bias_converters(
            f"layers.{num_layers + 1}.final_norm", f"{self._hf_prefix}.final_layernorm", norm_bias
        )

        for i in range(num_layers):
            # SSM
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+1}.mixer.in_proj", f"{self._hf_prefix}.layers.{i}.mixer.in_proj", ssm_bias
            )
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+1}.mixer.out_proj", f"{self._hf_prefix}.layers.{i}.mixer.out_proj", ssm_bias
            )
            converters.append(
                WeightConverter(
                    f"layers.{i+1}.mixer.D", f"{self._hf_prefix}.layers.{i}.mixer.D", self._model.config.base_model
                )
            )
            converters.append(
                WeightConverter(
                    f"layers.{i+1}.mixer.z_bias",
                    f"{self._hf_prefix}.layers.{i}.mixer.z_bias",
                    self._model.config.base_model,
                )
            )
            converters.append(
                WeightConverter(
                    f"layers.{i+1}.mixer.conv1d_weight",
                    f"{self._hf_prefix}.layers.{i}.mixer.conv1d.weight",
                    self._model.config.base_model,
                )
            )
            converters.append(
                WeightConverter(
                    f"layers.{i+1}.mixer.conv1d_bias",
                    f"{self._hf_prefix}.layers.{i}.mixer.conv1d.bias",
                    self._model.config.base_model,
                )
            )

            # Norm
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+1}.norm_1", f"{self._hf_prefix}.layers.{i}.input_layernorm", norm_bias
            )
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+1}.norm_2", f"{self._hf_prefix}.layers.{i}.post_attention_layernorm", norm_bias
            )

            # MLP
            converters += self._get_mlp_converters(f"layers.{i+1}", f"{self._hf_prefix}.layers.{i}")

        return converters

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str) -> list[WeightConverter]:
        linear_bias: bool = self._model.config.base_model.transformer.add_linear_biases
        return [
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_1",
                (f"{hf_prefix}.mlp.gate_proj", f"{hf_prefix}.mlp.up_proj"),
                linear_bias,
                SplitWeightConverter,
            ),
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_2",
                f"{hf_prefix}.mlp.down_proj",
                linear_bias,
                MLPLayer2Converter,
            ),
        ]

    def _get_weight_and_bias_converters(
        self,
        fast_llm_prefix: str | tuple[str, ...],
        hf_prefix: str | tuple[str, ...],
        use_bias: bool,
        cls=WeightConverter,
    ) -> list[WeightConverter]:
        if isinstance(fast_llm_prefix, str):
            fast_llm_prefix = (fast_llm_prefix,)
        if isinstance(hf_prefix, str):
            hf_prefix = (hf_prefix,)
        converters = [
            cls(
                tuple(f"{prefix}.weight" for prefix in fast_llm_prefix),
                tuple(f"{prefix}.weight" for prefix in hf_prefix),
                self._model.config.base_model,
            )
        ]
        if use_bias:
            converters.append(
                cls(
                    tuple(f"{prefix}.bias" for prefix in fast_llm_prefix),
                    tuple(f"{prefix}.bias" for prefix in hf_prefix),
                    self._model.config.base_model,
                )
            )
        return converters

    @classmethod
    def _load_config(cls, directory: pathlib.Path | str) -> dict:
        if not os.path.exists(directory / "config.json"):
            raise FileNotFoundError(f"config.json not found in {directory}")
        with open(directory / "config.json") as f:
            config = json.load(f)
        Assert.eq(config["model_type"], cls.get_huggingface_model_type())
        return config

    @classmethod
    def _save_config(cls, directory: pathlib.Path | str, config: dict[str, typing.Any]) -> None:
        with open(directory / "config.json", "w") as f:
            json.dump(config, f)


class AprielSSMHuggingfaceCheckpointHandler(CommonSSMHuggingfaceCheckpointHandler):
    """
    Lamba-like configs, pure SSM models.
    """

    _model: HybridSSMModel
    _model_class: typing.ClassVar[FastLLMModelConfig] = HybridSSMModelConfig
    format: typing.ClassVar[type[CheckpointFormat]] = AprielSSMHuggingfaceCheckpointFormat
    architecture: typing.ClassVar[str] = "AprielSSMForCausalLM"

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            RenameParamConverter(
                fast_llm_names=(("vocab_size",),),
                export_names=(("vocab_size",),),
            ),
            RenameParamConverter(
                fast_llm_names=(("ssm", "d_inner"),),
                export_names=(("ssm_cfg", "d_inner"),),
            ),
            ConstantExportParamConverter(export_names=(("mlp_bias",),), export_value=False),
            ConstantImportParamConverter(fast_llm_names=(("use_position_embeddings",),), fast_llm_value=False),
            MappedConfigParamConverter(
                fast_llm_names=(("transformer", "activation_type"),),
                export_names=(("hidden_act",),),
                fast_llm_value=ActivationType.from_hf_name,
                export_value=lambda activation_type: activation_type.hf_name,
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "num_layers"),),
                export_names=(("num_hidden_layers",),),
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "hidden_size"),),
                export_names=(("hidden_size",),),
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "ffn_hidden_size"),),
                export_names=(("intermediate_size",),),
            ),
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "normalization", "type"),),
                fast_llm_value=RMSNormalizationConfig.dynamic_type_name,
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "normalization", "epsilon"),), export_names=(("rms_norm_eps",),)
            ),
            ConstantImportParamConverter(fast_llm_names=(("transformer", "gated"),), fast_llm_value=True),
            ConstantImportParamConverter(fast_llm_names=(("transformer", "add_linear_biases"),), fast_llm_value=False),
            RenameParamConverter(
                fast_llm_names=(("tie_word_embeddings",),),
                export_names=(("tie_word_embeddings",),),
            ),
        ]

    def _create_weight_converters(self) -> list[WeightConverter]:
        converters = super()._create_weight_converters()
        num_layers = self._model.config.base_model.transformer.num_layers
        norm_bias: bool = False

        # Embedding and output
        if self._model.config.base_model.tie_word_embeddings:
            converters.append(WeightConverter("layers.0.word_embeddings_weight", "model.embed_tokens.weight"))
            converters.append(IgnoreImportWeightConverter((), "lm_head.weight"))
        else:
            converters.append(WeightConverter("layers.0.word_embeddings_weight", "model.embed_tokens.weight"))
            converters.append(WeightConverter(f"layers.{num_layers + 1}.output_weights", "lm_head.weight"))

        # Final norm
        converters += self._get_weight_and_bias_converters(
            f"layers.{num_layers + 1}.final_norm", "model.norm", norm_bias
        )

        for i in range(num_layers):
            # Norm
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+1}.norm_1", f"model.layers.{i}.input_layernorm", norm_bias
            )
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+1}.norm_2", f"model.layers.{i}.post_attention_layernorm", norm_bias
            )

            # MLP
            converters += self._get_mlp_converters(f"layers.{i+1}", f"model.layers.{i}")

        return converters

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str) -> list[WeightConverter]:
        linear_bias: bool = self._model.config.base_model.transformer.add_linear_biases
        return [
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_1",
                (f"{hf_prefix}.mlp.gate_proj", f"{hf_prefix}.mlp.up_proj"),
                linear_bias,
                SplitWeightConverter,
            ),
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_2",
                f"{hf_prefix}.mlp.down_proj",
                linear_bias,
                MLPLayer2Converter,
            ),
        ]

    @classmethod
    def _load_config(cls, directory: pathlib.Path | str) -> dict:
        if not os.path.exists(directory / "config.json"):
            raise FileNotFoundError(f"config.json not found in {directory}")
        with open(directory / "config.json") as f:
            config = json.load(f)
        Assert.eq(config["model_type"], cls.get_huggingface_model_type())
        return config

    @classmethod
    def _save_config(cls, directory: pathlib.Path | str, config: dict[str, typing.Any]) -> None:
        with open(directory / "config.json", "w") as f:
            json.dump(config, f)


class AprielSSMHHybridHuggingfaceCheckpointHandler(
    HybridModelCheckpointHandler,  # handles the block structure parameter
    CommonSSMHuggingfaceCheckpointHandler,  # handles the SSM layers
    CommonLlamaHuggingfaceCheckpointHandler,  # handles the LLama layers
):
    """
    Lamba-like configs, models that interleave LLama like layers with LLamba-like SSM layers.
    """

    _model: HybridSSMModel
    _model_class: typing.ClassVar[FastLLMModelConfig] = HybridSSMModelConfig
    format: typing.ClassVar[type[CheckpointFormat]] = AprielSSMHHybridHuggingfaceCheckpointFormat
    _default_block_type: str = SSMBlockType.mamba2_discrete.value
    architecture: typing.ClassVar[str] = "AprielSSMHybridForCausalLM"

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            RenameParamConverter(
                fast_llm_names=(("ssm", "d_inner"),),
                export_names=(("ssm_cfg", "d_inner"),),
            ),
            ConstantExportParamConverter(export_names=(("attention_bias",),), export_value=False),
            ConstantExportParamConverter(export_names=(("mlp_bias",),), export_value=False),
        ]

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str) -> list[WeightConverter]:
        linear_bias: bool = self._model.config.base_model.transformer.add_linear_biases
        return [
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_1",
                (f"{hf_prefix}.mlp.gate_proj", f"{hf_prefix}.mlp.up_proj"),
                linear_bias,
                SplitWeightConverter,
            ),
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_2",
                f"{hf_prefix}.mlp.down_proj",
                linear_bias,
                MLPLayer2Converter,
            ),
        ]

    @classmethod
    def _load_config(cls, directory: pathlib.Path | str) -> dict:
        if not os.path.exists(directory / "config.json"):
            raise FileNotFoundError(f"config.json not found in {directory}")
        with open(directory / "config.json") as f:
            config = json.load(f)
        Assert.eq(config["model_type"], cls.get_huggingface_model_type())
        return config

    @classmethod
    def _save_config(cls, directory: pathlib.Path | str, config: dict[str, typing.Any]) -> None:
        with open(directory / "config.json", "w") as f:
            json.dump(config, f)


class AprielThinkerSSMHHybridHuggingfaceCheckpointHandler(
    HybridModelCheckpointHandler,  # handles the block structure parameter
    CommonSSMHuggingfaceCheckpointHandler,  # handles the SSM layers
    CommonLlamaHuggingfaceCheckpointHandler,  # handles the LLama layers
):
    """
    Lamba-like configs, models that interleave LLama like layers with LLamba-like SSM layers.
    """

    _model: HybridSSMModel
    _model_class: typing.ClassVar[FastLLMModelConfig] = HybridSSMModelConfig
    format: typing.ClassVar[type[CheckpointFormat]] = AprielThinkerSSMHHybridHuggingfaceCheckpointFormat
    _default_block_type: str = SSMBlockType.mamba2_discrete.value
    _hf_prefix: str = "model"
    architecture: typing.ClassVar[str] = "AprielThinkerSSMHybridForCausalLM"

    def _create_weight_converters(self) -> list[WeightConverter]:
        converters = super()._create_weight_converters()
        # num_layers = self._model.config.base_model.transformer.num_layers
        # # Embedding and output
        # if self._model.config.base_model.tie_word_embeddings:
        #     converters.append(
        #         WeightConverter("layers.0.word_embeddings_weight", f"{self._hf_prefix}.embedding.weight")
        #     )
        #     converters.append(IgnoreImportWeightConverter((), f"{self._hf_prefix}.lm_head.weight"))
        # else:
        #     converters.append(
        #         WeightConverter("layers.0.word_embeddings_weight", f"{self._hf_prefix}.embedding.weight")
        #     )
        #     converters.append(
        #         WeightConverter(f"layers.{num_layers + 1}.output_weights", f"{self._hf_prefix}.lm_head.weight")
        #     )
        return converters

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        return super()._create_config_converters() + [
            RenameParamConverter(
                fast_llm_names=(("ssm", "d_inner"),),
                export_names=(("ssm_cfg", "d_inner"),),
            ),
            IgnoreImportParamConverter(export_names=(("sliding_window",),), ignore_export_value=None),
        ]

    def _get_mlp_converters(self, fast_llm_prefix: str, hf_prefix: str) -> list[WeightConverter]:
        linear_bias: bool = self._model.config.base_model.transformer.add_linear_biases
        return [
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_1",
                (f"{hf_prefix}.mlp.gate_proj", f"{hf_prefix}.mlp.up_proj"),
                linear_bias,
                SplitWeightConverter,
            ),
            *self._get_weight_and_bias_converters(
                f"{fast_llm_prefix}.mlp.layer_2",
                f"{hf_prefix}.mlp.down_proj",
                linear_bias,
                MLPLayer2Converter,
            ),
        ]

    @classmethod
    def _load_config(cls, directory: pathlib.Path | str) -> dict:
        if not os.path.exists(directory / "config.json"):
            raise FileNotFoundError(f"config.json not found in {directory}")
        with open(directory / "config.json") as f:
            config = json.load(f)
        Assert.eq(config["model_type"], cls.get_huggingface_model_type())
        return config

    @classmethod
    def _save_config(cls, directory: pathlib.Path | str, config: dict[str, typing.Any]) -> None:
        with open(directory / "config.json", "w") as f:
            json.dump(config, f)
