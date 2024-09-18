import logging
import random

import torch
import transformers.modeling_outputs

from fast_llm.distributed import PhaseType
from fast_llm.engine.huggingface.config import HuggingfaceModelConfig
from fast_llm.engine.huggingface.model import HuggingfacePreTrainedModel
from fast_llm.layers.transformer.config import TransformerKwargs
from fast_llm.models.gpt.model import GPTModel, GPTModelConfig

logger = logging.getLogger(__name__)


class HuggingfaceGPTModelConfig(HuggingfaceModelConfig):
    model_type = "fast_llm_gpt"
    model_class = GPTModel
    fast_llm_config: GPTModelConfig


class HuggingfaceGPTModelForCausalLM(HuggingfacePreTrainedModel):
    config_class = HuggingfaceGPTModelConfig
    config: HuggingfaceGPTModelConfig
    _fast_llm_model: GPTModel
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
        if output_hidden_states:
            raise NotImplementedError()
        if attention_mask is not None:
            raise NotImplementedError()
        if position_ids is not None:
            raise NotImplementedError()
        if inputs_embeds is not None:
            raise NotImplementedError()
        if labels is not None:
            raise NotImplementedError()

        # Iteration serves as a random seed, using random module because it's not seeded by Fast LLM
        iteration = random.randint(0, 2**32)
        batch = self._fast_llm_model.base_model.preprocess(input_ids, phase=PhaseType.inference, iteration=iteration)
        ((_, kwargs),) = batch

        if past_key_values is not None:
            # The transformers will use the past keys and values to this list.
            kwargs[TransformerKwargs.past_key_values] = past_key_values
            # TODO: preprocess needs to know about the past.
            raise NotImplementedError()
        if use_cache:
            # The transformers will save the present keys and values to this list.
            kwargs[TransformerKwargs.presents] = []

        _, _, _ = self._runner.run_step(iter((batch,)), self._schedule, iteration=iteration, preprocessed=True)

        # TODO: Make a proper way of returning the model output.
        logits = kwargs["logits"]

        if not return_dict:
            outputs = (logits,)
            if use_cache:
                outputs += (kwargs[TransformerKwargs.presents],)
            return outputs

        return transformers.modeling_outputs.CausalLMOutputWithPast(
            logits=logits,
            past_key_values=kwargs[TransformerKwargs.presents],
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        raise NotImplementedError()
