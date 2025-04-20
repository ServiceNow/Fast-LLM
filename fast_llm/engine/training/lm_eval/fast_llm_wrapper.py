import logging
import pathlib

from typing import Optional, Union, Any

import transformers
import huggingface_hub
import torch


# make lazy
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import configure_pad_token, stop_sequences_criteria
from lm_eval.api.model import CacheHook

from fast_llm.engine.checkpoint.config import CheckpointLoadConfig
from fast_llm.engine.multi_stage.config import FastLLMModelConfig
from fast_llm.models.auto import model_registry
from fast_llm.engine.huggingface.model import HuggingfaceBaseModelForCausalLM


eval_logger = logging.getLogger(__name__)


# move to fast_llm
class FastLLMWrapper(HFLM):
    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        model: HuggingfaceBaseModelForCausalLM,
        tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
        truncation: Optional[bool] = False,
        logits_cache: bool = True,
        add_bos_token: Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
    ):
        # intitialize manualy fields in base class as we do not want to call init on HFLM
        # super().__init__()
        # TODO: properly inicialize distributed, for now works only on one gpu
        self._rank = 0
        self._world_size = 1
        self.cache_hook = CacheHook(None)

        # set some inputs which are expected in HFLM but are not used by our model curretnly
        backend = "causal"
        revision = "main"
        gguf_file = None
        delta = None
        peft = None

        # set some inputs which are expected in HFLM but are set by our model config
        # TODO: do _batch_config public read only property
        max_length = model._batch_config.sequence_length
        #batch_size = model._batch_config.micro_batch_size
        batch_size = model._batch_config.batch_size
        max_batch_size = batch_size

        self.backend = backend

        # set tokenizer object
        assert isinstance(tokenizer, transformers.PreTrainedTokenizer) or isinstance(
            tokenizer, transformers.PreTrainedTokenizerFast
        )
        self.tokenizer = tokenizer

        # initialize model fields
        self._model = model
        self._device = self._model.device
        self._config = self._model.config

        # access self._model through self.model property outside this method
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.tie_weights()

        self.truncation = truncation
        self.logits_cache = logits_cache
        self.vocab_size = self.tokenizer.vocab_size
        # select (or create) a pad token to use
        self.tokenizer = configure_pad_token(self.tokenizer, model_config=self.config)

        self.add_bos_token = add_bos_token
        # TODO: do we support gemma models?
        if "gemma" in getattr(self.config, "model_type", ""):
            self.add_bos_token = True
            eval_logger.info(
                f"Model type is '{self.config.model_type}', part of the Gemma family--a BOS"
                " token will be used as Gemma underperforms without it."
            )

        self._max_length = max_length
        self.pretrained = model
        self.delta = delta
        self.peft = peft
        self.revision = revision
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size

        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self.batch_size_per_gpu = batch_size[0]
            self.batch_schedule = float(batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self.batch_size_per_gpu = int(batch_size)

        self.custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            eval_logger.info(f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}")

    def _model_call(self, inps, attn_mask=None, labels=None):
        """
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :param attn_mask: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :param labels: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        # TODO: do we need no_grad for our model?
        with torch.no_grad():
            if attn_mask is not None or labels is not None:
                assert attn_mask is not None and labels is not None
                return self.model(
                    input_ids=inps,
                    attention_mask=attn_mask,
                    labels=labels,
                    position_ids=None,
                    past_key_values=None,
                    inputs_embeds=None,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                ).logits
            else:
                return self.model(
                    input_ids=inps,
                    attention_mask=None,
                    position_ids=None,
                    past_key_values=None,
                    inputs_embeds=None,
                    labels=None,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                ).logits

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(self.tokenizer, stop, context.shape[1], context.shape[0])
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=False,
            **generation_kwargs,
        )
