import logging
import random
import typing

import torch
import transformers.modeling_outputs

from fast_llm.data.data.gpt.data import GPTBatch
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.engine.inference.config import HuggingfaceModelConfig
from fast_llm.engine.inference.huggingface import HuggingfaceBaseModelForCausalLM
from fast_llm.layers.attention.config import AttentionKwargs
from fast_llm.models.gpt.config import GPTModelConfig
from fast_llm.models.gpt.model import GPTBaseModel, GPTInferenceRunner

logger = logging.getLogger(__name__)


class HuggingfaceGPTModelConfig(HuggingfaceModelConfig):
    model_type = "fast_llm_gpt"
    model_config_class = GPTModelConfig
    fast_llm_config: GPTModelConfig


class HuggingfaceGPTModelForCausalLM(HuggingfaceBaseModelForCausalLM):
    config_class = HuggingfaceGPTModelConfig
    config: HuggingfaceGPTModelConfig
    runner_class: typing.ClassVar[type[GPTInferenceRunner]] = GPTInferenceRunner
    fast_llm_base_model: GPTBaseModel
    # base_model_prefix = ""
    # _no_split_modules = None
    # _supports_cache_class = False
    # _tied_weights_keys = []

    def forward(
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

        # Iteration serves as a random seed, using random module because it's not seeded by Fast LLM
        iteration = random.randint(0, 2**32)
        batch = self.fast_llm_base_model.preprocess(
            GPTBatch(input_ids, sequence_lengths=sequence_lenghts), phase=PhaseType.inference, iteration=iteration
        )
        ((input_, kwargs),) = batch

        if past_key_values is not None:
            # The transformers will use the past keys and values to this list.
            kwargs[AttentionKwargs.past_key_values] = past_key_values
            # TODO: preprocess needs to know about the past.
            raise NotImplementedError()
        if use_cache:
            # The transformers will save the present keys and values to this list.
            kwargs[AttentionKwargs.presents] = []

        if output_hidden_states:
            kwargs["output_hidden_states"] = True
            kwargs["hidden_states"] = {}
        else:
            kwargs["output_hidden_states"] = False

        self._inference_runner.forward(input_, kwargs, iteration=iteration)

        # TODO: Make a proper way of returning the model output.
        logits = kwargs["logits"]

        # TODO: convert hidden state form dict to list to be the same as with HFs
        hidden_states = None
        if output_hidden_states:
            hidden_states = kwargs["hidden_states"]

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
