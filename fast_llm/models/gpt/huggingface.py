import functools
import logging
import random
import re
import typing

import torch
import transformers.modeling_outputs

from fast_llm.data.document.language_model import LanguageModelBatch, LanguageModelInput
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.inference.config import HuggingfaceModelConfig
from fast_llm.engine.inference.huggingface import HuggingfacePreTrainedModel
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.models.gpt.config import GPTModelConfig
from fast_llm.models.gpt.model import GPTBaseModel, GPTInferenceRunner

logger = logging.getLogger(__name__)


class HuggingfaceGPTModelConfig(HuggingfaceModelConfig):
    model_type = "fast_llm_gpt"
    model_config_class = GPTModelConfig
    fast_llm_config: GPTModelConfig


class HuggingfaceGPTModelForCausalLM(HuggingfacePreTrainedModel):
    config_class = HuggingfaceGPTModelConfig
    config: HuggingfaceGPTModelConfig
    runner_class: typing.ClassVar[type[GPTInferenceRunner]] = GPTInferenceRunner
    fast_llm_base_model: GPTBaseModel

    def inner_forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | transformers.modeling_outputs.CausalLMOutputWithPast:
        return self._inner_forward(
            self._get_batch(input_ids, attention_mask),
            input_ids.shape,
            past_key_values,
            inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

    def _get_batch(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> LanguageModelBatch:
        # NOTE: We are ignoring position_ids as we reconstruct them from attention_mask via sequence_lengths.
        if attention_mask is None:
            sequence_lengths = [input_ids.size(1)] * input_ids.size(0)
        else:
            # First non-zero indexes or zero index if the row is all zeros (invalid row)
            first_non_zero_indexes = attention_mask.argmax(dim=1)

            # Check if the sequence is left-padded and if the remaining ones are continuous 1-ns
            assert (attention_mask.sum(axis=1) == (attention_mask.shape[1] - first_non_zero_indexes)).all()

            sequence_lengths = [
                el_
                for el in first_non_zero_indexes.tolist()
                for el_ in torch.tensor(
                    [attention_mask.shape[1]] if el == 0 else [el, attention_mask.shape[1] - el], dtype=torch.int64
                )
            ]
        return LanguageModelBatch(tokens=input_ids.flatten(), lengths=sequence_lengths)

    def _inner_forward(
        self,
        batch: LanguageModelInput,
        input_shape: tuple[int],
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> transformers.modeling_outputs.CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if output_attentions:
            raise NotImplementedError()
        if inputs_embeds is not None:
            raise NotImplementedError()
        if labels is not None:
            raise NotImplementedError()

        # Iteration serves as a random seed, using random module because it's not seeded by Fast LLM
        iteration = random.randint(0, 2**32)

        model_input = self._get_input(
            batch,
            past_key_values,
            use_cache,
            output_hidden_states,
        )
        ((input_, kwargs),) = self.fast_llm_base_model.preprocess_batch(
            [model_input],
            phase=PhaseType.inference,
            iteration=iteration,
            device=self._fast_llm_model.distributed.device,
        )

        self._inference_runner.forward(input_, kwargs, iteration=iteration)

        # TODO: Make a proper way of returning the model output.
        hidden_states = {
            name: meta.local_to_global(tensor)[0].unflatten(0, input_shape)
            for name, (meta, tensor) in kwargs[AttentionKwargs.hidden_states].items()
        }

        # TODO: Handle MTP.
        logits = hidden_states.pop("head.logits")

        output = transformers.modeling_outputs.CausalLMOutputWithPast(
            logits=logits,
            hidden_states=hidden_states or None,
            past_key_values=kwargs[AttentionKwargs.presents],
        )
        return (
            output
            if return_dict
            else tuple(x for x in (output.logits, output.hidden_states, output.past_key_values) if x is not None)
        )

    def _get_input(
        self,
        batch: LanguageModelBatch,
        past_key_values=None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> LanguageModelInput:
        (model_input,) = batch.get_model_inputs(self.preprocessing_config)

        if output_hidden_states:
            if isinstance(output_hidden_states, bool):
                # Hugging Face expect the last layer to include the final norm.
                # Note: We can't index `decoder` with slice because it tries to create a new block sequence instance.
                output_hidden_states = (
                    [self.fast_llm_base_model.embeddings.module_name + "$"]
                    + [layer.module_name + "$" for layer in self.fast_llm_base_model.decoder][:-1]
                    + [self.fast_llm_base_model.head.final_norm.module_name + "$"]
                )

            # This needs to be set before preprocessing so it propagates to layers with namespace.
            # kwargs is shallow-copied so changes will propagate back to the main namespace.
            model_input.output_hidden_states.update(re.compile(pattern) for pattern in output_hidden_states)

        if past_key_values is not None:
            # The transformers will use the past keys and values to this list.
            model_input.pasts = past_key_values
            # TODO: preprocess needs to know about the past.
            raise NotImplementedError()
        if use_cache:
            # The transformers will save the present keys and values to this list.
            model_input.presents = []

        # Propagate to sub-configs if needed.
        model_input.set_children_attributes()
        return model_input

    @functools.cached_property
    def preprocessing_config(self):
        return self._fast_llm_model.get_preprocessing_config(PhaseType.inference)
