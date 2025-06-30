# Copyright 2024 ServiceNow. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import warnings
from dataclasses import dataclass
from math import ceil
from typing import Any, Optional, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.utils import GenerationMixin
from transformers.utils import ModelOutput, is_torchdynamo_compiling, logging

logger = logging.get_logger(__name__)


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits


def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


# batch_sample_tokens
def batch_sample_tokens(
    logits, mask_indexes, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False
):
    # print(f"batch_sample_tokens: {logits.shape} ")
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        # logit will have different lengths for each sequence so cannot stack - it is not a proper batch???
        logits = torch.stack([top_p_logits(logit[mask], top_p) for logit, mask in zip(logits, mask_indexes)], dim=0)
    if top_k is not None:
        logits = torch.stack([top_k_logits(logit[mask], top_k) for logit, mask in zip(logits, mask_indexes)], dim=0)

    # if logits are not of the same sequence so therefore we can pad them with -inf but need remove them back ...
    probs = torch.softmax(logits, dim=-1)
    # print(f"probs: {probs.shape}")

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)

    # print(f"confidence: {confidence.shape} x0: {x0.shape}")
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0


@dataclass
class SLAMModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[tuple[torch.FloatTensor]] = None


class SLAMGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", "origin")
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass


class SLAMGenerationMixin(GenerationMixin):
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.LongTensor, dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == SLAMGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[SLAMGenerationConfig], **kwargs: dict
    ) -> SLAMGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = SLAMGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: SLAMGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[SLAMGenerationConfig] = None,
        **kwargs,
    ) -> Union[SLAMModelOutput, torch.LongTensor]:
        # fix seed for reproducability torch.random.manual_seed - lm-eval is setting it
        torch.random.manual_seed(0)

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop(
            "generation_tokens_hook_func", lambda step, x, logits, end_of_prompt: x
        )
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=  # The code `has_default_max_length` is not a valid Python code
            # snippet. It seems to be a placeholder or a comment in the code.
            # It does not perform any specific action or functionality in
            # Python.
            has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id")
            and torch.any(input_ids == generation_config.pad_token_id)
            and attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences, input_ids=input_ids, attention_mask=attention_mask
        )

        block_size = kwargs.pop("block_size", None)
        use_cache = kwargs.pop("use_cache", False)
        causal_cache = kwargs.pop("causal_cache", False)

        if block_size is None:
            # Default diffusion generation
            result = self._sample(
                input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                generation_tokens_hook_func=generation_tokens_hook_func,
                generation_logits_hook_func=generation_logits_hook_func,
            )
            return result
        else:
            if causal_cache:
                # Block generation with casual KV Caching only works for Flash attention
                result = self._sample_with_block_with_causal_kv(
                    input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    generation_tokens_hook_func=generation_tokens_hook_func,
                    generation_logits_hook_func=generation_logits_hook_func,
                    block_size=block_size,
                )
                return result
            else:
                # Block generation with (diffusion) KV Caching
                result = self._sample_with_block(
                    input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    generation_tokens_hook_func=generation_tokens_hook_func,
                    generation_logits_hook_func=generation_logits_hook_func,
                    block_size=block_size,
                    use_cache=use_cache,
                )
                return result

    # loop confidence implementation - working same results for bs 1
    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: SLAMGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
    ) -> Union[SLAMModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if (return_dict_in_generate and output_history) else None

        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        input_ids_length = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        # this allows user-defined token control of the intermediate steps
        # x = generation_tokens_hook_func(None, x, None, input_ids_length)

        for i in range(steps):

            logits = self(x, attention_mask, tok_idx).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(i, x, logits)

            t = timesteps[i]
            s = timesteps[i + 1]

            # loop around the batch
            for b in range(batch_size):
                x_row = x[b, :]
                mask_index = x_row == mask_token_id
                # if the sequence is already completed, skip it
                if mask_index.sum() == 0:
                    continue
                mask_logits = logits[b, mask_index]

                if alg == "origin":
                    # p_transfer = 1 - s / t if i < steps - 1 else 1
                    # x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                    # transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                    # _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                    # x[mask_index] = x0.clone()
                    raise RuntimeError("batch origin alg is not supported")
                else:
                    if alg == "maskgit_plus":
                        confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                    elif alg == "topk_margin":
                        confidence, x0 = sample_tokens(
                            mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True
                        )
                    elif alg == "entropy":
                        confidence, x0 = sample_tokens(
                            mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True
                        )
                    else:
                        raise RuntimeError(f"Unknown alg: {alg}")
                    num_mask_token = mask_index.sum()
                    number_transfer_tokens = ceil(num_mask_token * (1 - s / t)) if i < steps - 1 else num_mask_token

                    if number_transfer_tokens > 0:
                        if alg_temp is None or alg_temp == 0:
                            _, transfer_index = torch.topk(confidence, number_transfer_tokens)

                        else:
                            confidence = confidence / alg_temp
                            confidence = F.softmax(confidence, dim=-1)

                            transfer_index = torch.multinomial(confidence, num_samples=number_transfer_tokens)
                        x0_ = torch.zeros_like(x0, device=self.device, dtype=torch.long) + mask_token_id
                        x0_[transfer_index] = x0[transfer_index].clone()
                        x[b, mask_index] = x0_

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits, input_ids_length)

            if not torch.any(x == mask_token_id):
                break

            # Update attention mask based on pad_token_id and eos_token_id
            attention_mask_tmp = torch.where(
                (x == pad_token_id) | (x == eos_token_id),
                torch.tensor(0, device=x.device, dtype=torch.bool),
                torch.tensor(1, device=x.device, dtype=torch.bool),
            )
            attention_mask_tmp = torch.logical_and(
                attention_mask_tmp.unsqueeze(1).unsqueeze(-2),
                attention_mask_tmp.unsqueeze(1).unsqueeze(-1),
            )
            # print(f"attention_mask: {attention_mask_tmp.shape} {attention_mask_tmp}")
            attention_mask = attention_mask_tmp

            if histories is not None:
                histories.append(x.clone())

        if return_dict_in_generate:
            return SLAMModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x

    # block generation with kv cache
    def _sample_with_block(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: SLAMGenerationConfig,
        block_size: int,
        use_cache: bool,
        generation_tokens_hook_func,
        generation_logits_hook_func,
    ) -> Union[SLAMModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        use_cache = use_cache

        histories = [] if (return_dict_in_generate and output_history) else None

        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id
        block_size = block_size
        gen_length = generation_config.max_new_tokens
        num_of_blocks = gen_length // block_size
        steps = steps // num_of_blocks

        assert gen_length % block_size == 0, "gen_length should be divisible by block_size"

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        # TODO: Avoid this check and all future checks by always creating a mask
        # If any padding tokens i.e 0 in attention mask
        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx_base = attention_mask.long().cumsum(-1) - 1
            tok_idx_base.masked_fill_(attention_mask == 0, 1)
            # Leave padding out "<|endoftext|>1+1=2 2+2=" -> [ 1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17]

            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            # Set False for padding tokens and rest True
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx_base = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        input_ids_length = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        past_key_values = None
        past_length = 0
        settled_length = input_ids_length
        x_input = x.clone()
        tok_idx = tok_idx_base.clone() if tok_idx_base is not None else None

        for blk_indx in range(num_of_blocks):
            current_block = (num_of_blocks - (blk_indx + 1)) * block_size

            for i in range(steps):

                model_outputs = self(
                    x_input, attention_mask, tok_idx, use_cache=use_cache, past_key_values=past_key_values
                )

                logits = model_outputs.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                t = timesteps[i]
                s = timesteps[i + 1]

                # loop around the batch
                for b in range(batch_size):
                    x_row = x_input[b, :]
                    mask_index = x_row == mask_token_id

                    # if the sequence is already completed, skip it
                    if mask_index.sum() == 0:
                        continue

                    if current_block > 0:
                        mask_index[-current_block:] = False
                    mask_logits = logits[b, mask_index]

                    if alg == "origin":
                        # p_transfer = 1 - s / t if i < steps - 1 else 1
                        # x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                        # transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                        # _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                        # x[mask_index] = x0.clone()
                        raise RuntimeError("batch origin alg is not supported")
                    else:
                        if alg == "maskgit_plus":
                            confidence, x0 = sample_tokens(
                                mask_logits, temperature=temperature, top_p=top_p, top_k=top_k
                            )
                        elif alg == "topk_margin":
                            confidence, x0 = sample_tokens(
                                mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True
                            )
                        elif alg == "entropy":
                            confidence, x0 = sample_tokens(
                                mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True
                            )
                        else:
                            raise RuntimeError(f"Unknown alg: {alg}")
                        num_mask_token = mask_index.sum()
                        number_transfer_tokens = (
                            ceil(num_mask_token * (1 - s / t)) if i < steps - 1 else num_mask_token
                        )

                        # print(f"block: {blk_indx} step: {i} batch: {b} confidence: {confidence} x0: {x0}")
                        # print(f"number_transfer_tokens: {number_transfer_tokens} num_mask_token: {num_mask_token} ")
                        if number_transfer_tokens > 0:
                            if alg_temp is None or alg_temp == 0:
                                _, transfer_index = torch.topk(confidence, number_transfer_tokens)

                            else:
                                confidence = confidence / alg_temp
                                confidence = F.softmax(confidence, dim=-1)

                                transfer_index = torch.multinomial(confidence, num_samples=number_transfer_tokens)
                            x0_ = torch.zeros_like(x0, device=self.device, dtype=torch.long) + mask_token_id
                            x0_[transfer_index] = x0[transfer_index].clone()
                            x_input[b, mask_index] = x0_

                # this allows user-defined token control of the intermediate steps
                x_input = generation_tokens_hook_func(i, x_input, logits, input_ids_length)

                if use_cache:
                    # 1. Update settled tokens
                    x[:, past_length:] = x_input

                    # TODO: We can avoid these updates by setting a flag in the Attention call to not set KVs for these forward passes and only set when we reach end of the block
                    # Prepare for next forward pass
                    # 2. Update past_key_values to include only settled tokens from previous blocks
                    past_key_values = model_outputs.past_key_values
                    # need to reset this since the Attention call will add new KVs
                    past_key_values.crop(settled_length)
                    # past_key_values are already set from the last forward pass
                    past_length = past_key_values.get_seq_length()

                    # 3. Generic cache-dependent input and position index
                    # https://github.com/huggingface/transformers/blob/5f4ecf2d9f867a1255131d2461d75793c0cf1db2/src/transformers/generation/utils.py#L410C13-L410C53
                    x_input = x[:, past_length:].clone()
                    tok_idx = tok_idx_base[:, past_length:] if tok_idx is not None else None

                    # TODO: optimize this we don't need to compute this every forward pass maybe only change location where tokens are settled; adhering to early stopping
                    # 4. Set attention mask
                    # Update attention mask based from the full x to capture past eos and pad tokens masks
                    attention_mask_tmp = torch.where(
                        (x == pad_token_id) | (x == eos_token_id),
                        torch.tensor(0, device=x.device, dtype=torch.bool),
                        torch.tensor(1, device=x.device, dtype=torch.bool),
                    )
                    attention_mask_tmp = torch.logical_and(
                        attention_mask_tmp.unsqueeze(1).unsqueeze(-2),
                        attention_mask_tmp.unsqueeze(1).unsqueeze(-1),
                    )

                    # Drop values from the 3rd dimension to the size of new x_input so that it current Qs (aka inputs)
                    # [B, 1, Q_dim, K_dim]
                    attention_mask_tmp = attention_mask_tmp[:, :, past_length:, :]
                    attention_mask = attention_mask_tmp
                    # print(f"attention_mask: {attention_mask_tmp.shape}")

                else:
                    x = x_input

                    # Set attention mask
                    # Update attention mask based on pad_token_id and eos_token_id
                    attention_mask_tmp = torch.where(
                        (x_input == pad_token_id) | (x_input == eos_token_id),
                        torch.tensor(0, device=x.device, dtype=torch.bool),
                        torch.tensor(1, device=x.device, dtype=torch.bool),
                    )
                    attention_mask_tmp = torch.logical_and(
                        attention_mask_tmp.unsqueeze(1).unsqueeze(-2),
                        attention_mask_tmp.unsqueeze(1).unsqueeze(-1),
                    )
                    attention_mask = attention_mask_tmp
                    # No need to update tok_idx since we are computing all KVs with original positions again

                if histories is not None:
                    histories.append(x.clone())

                if not torch.any(x == mask_token_id):
                    # print("unmasked all tokens in current x exiting")
                    break

                # print(f"x_input: {x_input.shape} tok_idx: {tok_idx.shape if tok_idx is not None else None} {tok_idx}")

            # A block is completed update settled tokens length
            if not torch.any(x == mask_token_id):
                # print("unmasked all tokens in current x exiting")
                break
            settled_length += block_size

        if return_dict_in_generate:
            return SLAMModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x

    # block generation with casual kv cache for flash attention ONLY
    def _sample_with_block_with_causal_kv(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: SLAMGenerationConfig,
        block_size: int,
        generation_tokens_hook_func,
        generation_logits_hook_func,
    ) -> Union[SLAMModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        generation_config.pad_token_id
        generation_config.eos_token_id

        histories = [] if (return_dict_in_generate and output_history) else None

        block_size = block_size
        gen_length = generation_config.max_new_tokens
        num_of_blocks = gen_length // block_size
        steps = steps // num_of_blocks

        assert gen_length % block_size == 0, "gen_length should be divisible by block_size"

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        # If any padding tokens i.e 0 in attention mask
        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx_base = attention_mask.long().cumsum(-1) - 1
            tok_idx_base.masked_fill_(attention_mask == 0, 1)
            # Leave padding out "<|endoftext|>1+1=2 2+2=" -> [ 1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17]

            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            # Set False for padding tokens and rest True
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx_base = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        input_ids_length = input_ids.shape[1]
        batch_size = input_ids.shape[0]

        past_key_values = None
        past_length = 0
        tok_idx = tok_idx_base.clone() if tok_idx_base is not None else None
        x_input = x.clone()
        # initial settled length is the context/prompt length
        settled_length = input_ids_length
        # 1. Do first forward pass to get past_key_values for context in casual attention
        model_outputs = self(
            x_input,
            attention_mask=attention_mask,
            position_ids=tok_idx,
            use_cache=True,
            past_key_values=past_key_values,
            is_causal=True,
        )
        past_key_values = model_outputs.past_key_values
        # 2. Crop past_key_values to include only context tokens
        past_key_values.crop(settled_length)
        past_length = past_key_values.get_seq_length()
        # 3. Create new input for prediction
        x_input = x[:, past_length:].clone()
        tok_idx = tok_idx_base[:, past_length:] if tok_idx_base is not None else None

        # print(f"settled_length: {settled_length} past_length: {past_length} x_input: {x_input.shape} past_key_values: {past_key_values.get_seq_length()}")

        for blk_indx in range(num_of_blocks):
            current_block = (num_of_blocks - (blk_indx + 1)) * block_size

            for i in range(steps):
                model_outputs = self(
                    x_input,
                    attention_mask=  # The above code is defining a variable
                    # named `attention_mask` in Python.
                    attention_mask,
                    position_ids=tok_idx,
                    use_cache=True,
                    past_key_values=past_key_values,
                    is_causal=False,
                )

                logits = model_outputs.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                t = timesteps[i]
                s = timesteps[i + 1]

                # loop around the batch
                for b in range(batch_size):
                    x_row = x_input[b, :]
                    mask_index = x_row == mask_token_id

                    # if the sequence is already completed, skip it
                    if mask_index.sum() == 0:
                        continue

                    if current_block > 0:
                        mask_index[-current_block:] = False

                    mask_logits = logits[b, mask_index]

                    if alg == "origin":
                        # p_transfer = 1 - s / t if i < steps - 1 else 1
                        # x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                        # transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                        # _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                        # x[mask_index] = x0.clone()
                        raise RuntimeError("batch origin alg is not supported")
                    else:
                        if alg == "maskgit_plus":
                            confidence, x0 = sample_tokens(
                                mask_logits, temperature=temperature, top_p=top_p, top_k=top_k
                            )
                        elif alg == "topk_margin":
                            confidence, x0 = sample_tokens(
                                mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True
                            )
                        elif alg == "entropy":
                            confidence, x0 = sample_tokens(
                                mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True
                            )
                        else:
                            raise RuntimeError(f"Unknown alg: {alg}")
                        num_mask_token = mask_index.sum()
                        number_transfer_tokens = (
                            ceil(num_mask_token * (1 - s / t)) if i < steps - 1 else num_mask_token
                        )

                        # print(f"block: {blk_indx} step: {i} batch: {b} confidence: {confidence} x0: {x0}")
                        # print(f"number_transfer_tokens: {number_transfer_tokens} num_mask_token: {num_mask_token}")
                        if number_transfer_tokens > 0:
                            if alg_temp is None or alg_temp == 0:
                                _, transfer_index = torch.topk(confidence, number_transfer_tokens)

                            else:
                                confidence = confidence / alg_temp
                                confidence = F.softmax(confidence, dim=-1)

                                transfer_index = torch.multinomial(confidence, num_samples=number_transfer_tokens)
                            x0_ = torch.zeros_like(x0, device=self.device, dtype=torch.long) + mask_token_id
                            x0_[transfer_index] = x0[transfer_index].clone()
                            x_input[b, mask_index] = x0_

                # this allows user-defined token control of the intermediate steps
                x_input = generation_tokens_hook_func(i, x_input, logits, input_ids_length)

                # Update settled tokens
                x[:, past_length:] = x_input

                # Prepare for next forward pass

                # 1. Update past_key_values to include only settled tokens from previous blocks
                # past_key_values = model_outputs.past_key_values

                # needed bcuz Attention module adds them to cache so we need to remove them for next forward pass
                # we can stop the Attention module from adding them with a param for speedup
                past_key_values.crop(settled_length)
                # past_length = past_key_values.get_seq_length()
                # print(f"past_length: {past_length} x_input: {x_input.shape} past_length: {past_length} past_key_values: {past_key_values.get_seq_length()}")

                # # only works for sdpa
                # attention_mask_tmp = torch.where(
                #     (x == pad_token_id) | (x == eos_token_id),
                #     torch.tensor(0, device=x.device, dtype=torch.bool),
                #     torch.tensor(1, device=x.device, dtype=torch.bool)
                # )
                # attention_mask_tmp = torch.logical_and(
                #     attention_mask_tmp.unsqueeze(1).unsqueeze(-2),
                #     attention_mask_tmp.unsqueeze(1).unsqueeze(-1),
                # )

                # # Drop values from the 3rd dimension to the size of new x_input so that it current Qs (aka inputs)
                # # [B, 1, Q_dim, K_dim]
                # attention_mask_tmp = attention_mask_tmp[:, :, past_length:, :]
                # attention_mask = attention_mask_tmp

                if histories is not None:
                    histories.append(x.clone())

            # A block is completed update settled tokens length
            if not torch.any(x == mask_token_id):
                break
            settled_length += block_size
            model_outputs = self(
                x_input,
                attention_mask=attention_mask,
                position_ids=tok_idx,
                use_cache=True,
                past_key_values=past_key_values,
                is_causal=True,
            )
            past_key_values = model_outputs.past_key_values
            past_key_values.crop(settled_length)
            past_length = past_key_values.get_seq_length()
            x_input = x[:, past_length:].clone()
            tok_idx = tok_idx_base[:, past_length:] if tok_idx is not None else None

            # # Only works for sdpa
            # attention_mask_tmp = torch.where(
            #     (x == pad_token_id) | (x == eos_token_id),
            #     torch.tensor(0, device=x.device, dtype=torch.bool),
            #     torch.tensor(1, device=x.device, dtype=torch.bool)
            # )
            # attention_mask_tmp = torch.logical_and(
            #     attention_mask_tmp.unsqueeze(1).unsqueeze(-2),
            #     attention_mask_tmp.unsqueeze(1).unsqueeze(-1),
            # )

            # # Drop values from the 3rd dimension to the size of new x_input so that it current Qs (aka inputs)
            # # [B, 1, Q_dim, K_dim]
            # attention_mask_tmp = attention_mask_tmp[:, :, past_length:, :]
            # attention_mask = attention_mask_tmp
            # print(f"settled_length: {settled_length} past_length: {past_length} x_input: {x_input.shape} past_key_values: {past_key_values.get_seq_length()}")

        if return_dict_in_generate:
            return SLAMModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x
