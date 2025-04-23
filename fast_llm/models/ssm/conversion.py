import json
import os
import pathlib
import typing

from fast_llm.engine.checkpoint.config import CheckpointFormat
from fast_llm.engine.checkpoint.external import (
    ConstantImportParamConverter,
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
from fast_llm.layers.common.config import NormalizationType
from fast_llm.models.gpt.conversion import MLPLayer2Converter
from fast_llm.models.ssm.config import HybridSSMModelConfig, LLambaHuggingfaceCheckpointFormat
from fast_llm.models.ssm.model import HybridSSMModel
from fast_llm.utils import Assert

if typing.TYPE_CHECKING:
    pass


class LLambaHuggingfaceCheckpointHandler(HuggingfaceStateDictCheckpointHandler):
    _model: HybridSSMModel
    _model_class: typing.ClassVar[FastLLMModelConfig] = HybridSSMModelConfig
    format: typing.ClassVar[type[CheckpointFormat]] = LLambaHuggingfaceCheckpointFormat

    @classmethod
    def _create_config_converters(cls) -> list[ParamConverter]:
        """
        Create config converters for the model, see args under https://huggingface.co/cartesia-ai/Llamba-8B/blob/main/config.json
        """
        return super()._create_config_converters() + [
            ConstantImportParamConverter(fast_llm_names=(("use_position_embeddings",),), fast_llm_value=False),
            RenameParamConverter(
                fast_llm_names=(("transformer", "num_layers"),),
                export_names=(("n_layer",),),
            ),
            ConstantImportParamConverter(fast_llm_names=(("transformer", "gated"),), fast_llm_value=True),
            # TODO: is there an equivalen of pad_vocab_size_multiple in FastLLM, does it matter?
            RenameParamConverter(
                fast_llm_names=(("transformer", "normalization", "epsilon"),), export_names=(("norm_epsilon",),)
            ),
            RenameParamConverter(
                fast_llm_names=(("ssm", "normalization", "epsilon"),), export_names=(("norm_epsilon",),)
            ),
            ConstantImportParamConverter(
                fast_llm_names=(("transformer", "normalization", "type"),), fast_llm_value=NormalizationType.rms_norm
            ),
            RenameParamConverter(
                fast_llm_names=(("vocab_size",),),
                export_names=(("vocab_size",),),
            ),
            RenameParamConverter(
                fast_llm_names=(("tie_word_embeddings",),),
                export_names=(("tie_embeddings",),),
            ),
            RenameParamConverter(
                fast_llm_names=(("transformer", "hidden_size"),),
                export_names=(("d_model",),),
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
                fast_llm_names=(("transformer", "add_linear_biases"),),
                export_names=(
                    (
                        "mlp_cfg",
                        "bias",
                    ),
                ),
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
        ]

    def _create_weight_converters(self) -> list[WeightConverter]:
        converters = []
        num_layers = self._model.config.base_model.transformer.num_layers
        norm_bias: bool = False
        ssm_bias: bool = self._model.config.base_model.ssm.add_bias_linear

        # Embedding and output
        if self._model.config.base_model.tie_word_embeddings:
            converters.append(WeightConverter("layers.0.word_embeddings_weight", "backbone.embedding.weight"))
            converters.append(IgnoreImportWeightConverter((), "lm_head.weight"))
        else:
            converters.append(WeightConverter("layers.0.word_embeddings_weight", "backbone.embedding.weight"))
            converters.append(WeightConverter(f"layers.{num_layers + 1}.output_weights", "lm_head.weight"))

        # Final norm
        converters += self._get_weight_and_bias_converters(
            f"layers.{num_layers + 1}.final_norm", "backbone.final_layernorm", norm_bias
        )

        for i in range(num_layers):
            # SSM
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+1}.mixer.in_proj", f"backbone.layers.{i}.mixer.in_proj", ssm_bias
            )
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+1}.mixer.out_proj", f"backbone.layers.{i}.mixer.out_proj", ssm_bias
            )
            converters.append(
                WeightConverter(f"layers.{i+1}.mixer.D", f"backbone.layers.{i}.mixer.D", self._model.config.base_model)
            )
            converters.append(
                WeightConverter(
                    f"layers.{i+1}.mixer.z_bias", f"backbone.layers.{i}.mixer.z_bias", self._model.config.base_model
                )
            )
            converters.append(
                WeightConverter(
                    f"layers.{i+1}.mixer.conv1d_weight",
                    f"backbone.layers.{i}.mixer.conv1d.weight",
                    self._model.config.base_model,
                )
            )
            converters.append(
                WeightConverter(
                    f"layers.{i+1}.mixer.conv1d_bias",
                    f"backbone.layers.{i}.mixer.conv1d.bias",
                    self._model.config.base_model,
                )
            )

            # Norm
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+1}.norm_1", f"backbone.layers.{i}.input_layernorm", norm_bias
            )
            converters += self._get_weight_and_bias_converters(
                f"layers.{i+1}.norm_2", f"backbone.layers.{i}.post_attention_layernorm", norm_bias
            )

            # MLP
            converters += self._get_mlp_converters(f"layers.{i+1}", f"backbone.layers.{i}")

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
