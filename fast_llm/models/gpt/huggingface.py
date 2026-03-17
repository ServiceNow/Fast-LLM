import logging
import random
import re
import typing

import torch
import transformers.modeling_outputs

from fast_llm.data.sample.language_model import LanguageModelBatch
from fast_llm.data.sample.token import TokenBatch
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.inference.config import HuggingfaceModelConfig
from fast_llm.engine.inference.huggingface import HuggingfacePreTrainedModel
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.layers.block.config import BlockKwargs
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
            self._get_batch(input_ids, attention_mask, position_ids),
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
        position_ids: torch.Tensor | None = None,
    ):
        # NOTE: We are ignoring position_ids as we reconstruct them from attention_mask via sequence_lengths.
        if attention_mask is not None:
            # First non zero indexes or zero index if the row is all zeros (invalid row)
            first_non_zero_indexes = attention_mask.argmax(dim=1)

            # Check if the sequence is left-padded and if the remaining ones are continuous 1-ns
            assert (attention_mask.sum(axis=1) == (attention_mask.shape[1] - first_non_zero_indexes)).all()

            sequence_lenghts = [
                torch.tensor(
                    [attention_mask.shape[1]] if el == 0 else [el, attention_mask.shape[1] - el], dtype=torch.int64
                )
                for el in first_non_zero_indexes.tolist()
            ]
        else:
            sequence_lenghts = None
        return LanguageModelBatch(TokenBatch(input_ids, lengths=sequence_lenghts))

    def _inner_forward(
        self,
        batch: LanguageModelBatch,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: list[str | re.Pattern] | bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | transformers.modeling_outputs.CausalLMOutputWithPast:
        # TODO: Most of this is generalizable.
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

        ((input_meta, kwargs_meta),) = self.fast_llm_base_model.preprocess_meta(batch, phase=PhaseType.inference)

        if output_hidden_states:
            if isinstance(output_hidden_states, bool):
                # Hugging Face expect the last layer to include the final norm.
                # Note: We can't index `decoder` with slice because it tries to create a new block sequence instance.
                output_hidden_states = [layer.module_name + "$" for layer in self.fast_llm_base_model.decoder][:-1] + [
                    self.fast_llm_base_model.head.heads[0].final_norm.module_name + "$"
                ]

            # This needs to be set before preprocessing so it propagates to layers with namespace.
            # kwargs is shallow-copied so changes will propagate back to the main namespace.
            kwargs_meta[BlockKwargs.output_hidden_states] = [re.compile(pattern) for pattern in output_hidden_states]

        ((input_, kwargs),) = self.fast_llm_base_model.preprocess_batch(
            batch, [(input_meta, kwargs_meta)], phase=PhaseType.inference, iteration=iteration
        )

        if past_key_values is not None:
            # The transformers will use the past keys and values to this list.
            kwargs[AttentionKwargs.past_key_values] = past_key_values
            # TODO: preprocess needs to know about the past.
            raise NotImplementedError()
        if use_cache:
            # The transformers will save the present keys and values to this list.
            kwargs[AttentionKwargs.presents] = []

        self._inference_runner.forward(input_, kwargs, iteration=iteration)

        # TODO: Make a proper way of returning the model output.
        # TODO: Handle MTP.
        logits_meta, logits = kwargs[AttentionKwargs.hidden_states]["head.logits"]
        logits, _ = logits_meta.local_to_global(logits)
        logits = logits.unflatten(
            0, (kwargs[AttentionKwargs.batch_dim].global_size, kwargs[AttentionKwargs.sequence_q_dim].global_size)
        )

        if output_hidden_states:
            hidden_states = {
                key: tensor if meta is None else meta.local_to_global(tensor)[0]
                for key, (meta, tensor) in kwargs[AttentionKwargs.hidden_states].items()
            }
        else:
            hidden_states = None

        if not return_dict:
            # TODO: Then implementing cache, check hidden state goes before past in the tuple
            if output_hidden_states:
                outputs = (logits, hidden_states)
            else:
                outputs = (logits,)

            if use_cache:
                outputs += (kwargs[AttentionKwargs.presents],)
            return outputs

        return transformers.modeling_outputs.CausalLMOutputWithPast(
            logits=logits,
            hidden_states=hidden_states,
            past_key_values=kwargs[AttentionKwargs.presents],
        )
