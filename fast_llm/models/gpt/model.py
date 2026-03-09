import functools
import logging
import re
import typing

import torch

from fast_llm.data.document.config import LanguageModelBatchPreprocessingConfig
from fast_llm.data.document.language_model import LanguageModelInput
from fast_llm.engine.base_model.base_model import BaseModel
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.inference.runner import InferenceRunner
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.layers.block.config import BlockKwargs
from fast_llm.layers.language_model.config import LanguageModelKwargs
from fast_llm.layers.language_model.language_model import LanguageModel
from fast_llm.models.gpt.config import GPTBaseModelConfig, GPTModelConfig
from fast_llm.models.gpt.megatron import get_init_megatron
from fast_llm.tensor import ParameterMeta
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class GPTBaseModel[ConfigType: GPTBaseModelConfig](LanguageModel[ConfigType], BaseModel[ConfigType]):
    """
    A transformer-based language model generalizing the GPT model architecture.
    """

    _config: ConfigType

    def __init__(
        self,
        config: ConfigType,
        distributed_config: DistributedConfig,
    ):
        super().__init__(config, distributed_config, lr_scale=config.lr_scale, peft=config.peft)
        if self._config.use_megatron_initialization:
            for param in self.parameters():
                Assert.custom(isinstance, param, ParameterMeta)
                param.init_parameter = get_init_megatron(param, self._config.decoder.block, config.hidden_size)  # Noqa

    def preprocess_batch(
        self,
        model_inputs: list[LanguageModelInput],
        *,
        phase: PhaseType,
        iteration: int,
        metrics: dict | None = None,
        extra_kwargs: dict[str, typing.Any] | None = None,
        device: torch.device | None,
    ) -> list[tuple[torch.Tensor, dict]]:
        reference_preprocessed_batches = {}
        for name, reference_model in self._reference_models.items():
            reference_preprocessed_batches[name] = reference_model.fast_llm_model.base_model.preprocess_batch(
                model_inputs,
                phase=PhaseType.inference,
                iteration=iteration,
                device=device,
            )

        preprocessed = []
        for input_index, model_input in enumerate(model_inputs):
            if device is not None:
                model_input.to_device_(device)
            kwargs = model_input.to_kwargs()
            kwargs[LanguageModelKwargs.iteration] = iteration
            if extra_kwargs is not None:
                Assert.empty(kwargs.keys() & extra_kwargs.keys())
                kwargs.update(extra_kwargs)
            if phase == PhaseType.inference:
                kwargs[BlockKwargs.output_hidden_states].add(re.compile(r"head\..*logits.*$"))

            if not model_input.is_meta:
                for name, reference_model in self._reference_models.items():
                    reference_tokens, reference_kwargs = reference_preprocessed_batches[name][input_index]
                    if name in self._decoder_reference_models:
                        # TODO: Get the actual names
                        reference_kwargs[BlockKwargs.output_hidden_states].add(
                            re.compile(r"decoder\.\d+\.mixer_output$")
                        )

                    reference_model.forward(reference_tokens, reference_kwargs, iteration=iteration)

                    kwargs[f"reference_{name}_hidden_states"] = {
                        layer_name: tensor
                        for layer_name, (meta, tensor) in reference_kwargs[BlockKwargs.hidden_states].items()
                    }
                self.preprocess(kwargs)
            preprocessed.append((model_input.tokens, kwargs))

        return preprocessed

    def get_tied_parameters(self) -> dict[str, tuple[ParameterMeta, tuple[int, ...]]]:
        # TODO: Integrate to the `LayerBase` interface, move to `LanguageModel`, `MultiTokenPrediction`?
        output_weights = self.head.get_output_weights() + self.multi_token_prediction.get_output_weights()
        if self._config.tied_embedding_weight:
            output_weights.insert(0, self.embeddings.word_embeddings_weight)
        # print("WWWWWWWWW", [x.tensor_name for x in output_weights], self.multi_token_prediction.get_output_weights())
        return {output_weights[0].tensor_name: output_weights} if len(output_weights) > 1 else {}

    @functools.cached_property
    def _decoder_reference_models(self) -> set[str]:
        out = self._config.decoder.get_reference_models()
        Assert.leq(out, self._reference_models.keys())
        Assert.leq(len(out), 1)
        return out

    @functools.cached_property
    def _head_reference_models(self) -> set[str]:
        out = self._config.head.get_reference_models()
        Assert.leq(out, self._reference_models.keys())
        return out


class GPTModel[ConfigType: GPTModelConfig](FastLLMModel[ConfigType]):
    def get_preprocessing_config(
        self, phase: PhaseType, micro_batch_splits: int = 1
    ) -> LanguageModelBatchPreprocessingConfig:
        return LanguageModelBatchPreprocessingConfig(
            phase=phase,
            micro_batch_splits=micro_batch_splits,
            **self._base_model.get_preprocessing_config(),
        )


class GPTInferenceRunner(InferenceRunner):
    model_class: typing.ClassVar[type[GPTModel]] = GPTModel
