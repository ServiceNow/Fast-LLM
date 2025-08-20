import copy
import logging

import jinja2
import lm_eval.api.instance
import lm_eval.api.model
import lm_eval.evaluator
import lm_eval.models.utils
import lm_eval.utils
import torch
import torch.nn.functional as F
import tqdm.auto
import transformers

from fast_llm.core.distributed import gather_object, safe_barrier, scatter_object
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.evaluation.lm_eval.utils import prepare_lm_eval_simple_eval_params, process_lm_eval_results
from fast_llm.engine.inference.huggingface import HuggingfaceBaseModelForCausalLM
from fast_llm.layers.attention.rotary.config import NoRotaryConfig

logger = logging.getLogger(__name__)


class FastLLMLmEvalWrapper(lm_eval.api.model.TemplateLM):
    _DEFAULT_MAX_LENGTH = 2048
    _DEFAULT_MAX_GEN_TOKENS = 256

    def __init__(
        self,
        model: HuggingfaceBaseModelForCausalLM,
        tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast,
        truncation: bool | None = False,
        logits_cache: bool = True,
        add_bos_token: bool | None = False,
        prefix_token_id: int | None = None,
        max_length: int | None = None,
    ):
        super().__init__()

        # === Distributed setup ===
        self._rank = 0  # For lm_eval: always run on main rank
        self._world_size = 1
        self._distributed: Distributed = model._inference_runner._fast_llm_model.distributed

        if (
            self._distributed.config.sequence_data_rank == 0
            and self._distributed.config.pipeline_rank == 0
            and self._distributed.config.tensor_rank == 0
        ):
            self._group = self._distributed.batch_data_group
        else:
            self._group = torch.distributed.GroupMember.NON_GROUP_MEMBER

        # === Model & tokenizer setup ===
        self._model = model
        self._device = model.device
        self._config = model.config

        assert isinstance(tokenizer, (transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast))
        self._tokenizer = tokenizer
        self._tokenizer = lm_eval.models.utils.configure_pad_token(self._tokenizer, model_config=self._config)

        # === Generation/configuration parameters ===
        self._truncation = truncation
        self._logits_cache = logits_cache
        self._add_bos_token = add_bos_token
        self._max_length = max_length
        self._custom_prefix_token_id = prefix_token_id
        if prefix_token_id is not None:
            logger.info(f"Loglikelihood prefix token id used in evaluation: {self.prefix_token_id}")

        # === Internal constants ===
        self._backend = "causal"
        self._vocab_size = self._tokenizer.vocab_size

        # === Batch configuration ===
        self._batch_schedule = 1
        self._batch_sizes = {}  # Not used dynamically by lm_eval
        self._batch_size_per_gpu = model._inference_runner._batch_config.micro_batch_size
        self._batch_size = self._batch_size_per_gpu * self._distributed.config.batch_data_parallel
        self._max_batch_size = self._batch_size

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self._tokenizer.eos_token_id

    # overrides from TemplateLM, but not used externally
    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        if self._custom_prefix_token_id is not None:
            return self._custom_prefix_token_id
        if self._tokenizer.bos_token_id is not None:
            return self._tokenizer.bos_token_id
        return self._tokenizer.eos_token_id

    @property
    def max_length(self):
        # if max length manually set, return it
        if self._max_length:
            return self._max_length

        # check if it is absolute positional encoding and return max_position_embeddings
        if hasattr(self._config.fast_llm_config.base_model, "transformer"):
            # NOTE: will need to extend if more relative encoding types will be added
            if isinstance(self._config.fast_llm_config.base_model.transformer.rotary, NoRotaryConfig):
                return self._config.fast_llm_config.base_model.max_position_embeddings

        # check if tokenizer holds model sequence leigh info
        if hasattr(self._tokenizer, "model_max_length"):
            if self._tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self._tokenizer.model_max_length

        # finally try to get sequence length from batch config
        if hasattr(self._model._inference_runner._batch_config, "sequence_length"):
            return self._model._inference_runner._batch_config.sequence_length

        return self._DEFAULT_MAX_LENGTH

    # @property
    # def device(self):
    #     # only used for world_size when lm_eval world size > 1 and
    #     # should not be called with current lm_eval support implementation
    #     return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def tokenizer_name(self) -> str:
        return self._tokenizer.name_or_path.replace("/", "__")

    def run(self, cli_args: list[str], completed_steps: int, run_index: int):
        if self._distributed.config.rank == 0:
            args, simple_eval_kwargs = prepare_lm_eval_simple_eval_params(cli_args, completed_steps, run_index)
            simple_eval_kwargs["model"] = self

            # Needed for reporting as batch_size is set from args not lm for reporting in evaluate
            simple_eval_kwargs["batch_size"] = self._batch_size
            simple_eval_kwargs["max_batch_size"] = self._max_batch_size

            # As of lm_eval commit 758c5ed891b1ca48acd8d3a0d309a827215796b7
            # Expected to be a string even if empty and not None in simple_evaluate
            simple_eval_kwargs["model_args"] = ""

            results = lm_eval.evaluator.simple_evaluate(**simple_eval_kwargs)
            self.stop_workers()

            # Evaluation_tracker save expects model to be either string, but if model is passed
            # LM wrapper needs to be deep copyable and json serializable
            simple_eval_kwargs["evaluation_tracker"].general_config_tracker.model_source = (
                self._model.config.name_or_path
            )

            if results is not None:
                process_lm_eval_results(
                    args,
                    results,
                    simple_eval_kwargs["evaluation_tracker"],
                    completed_steps,
                )
        else:
            self.worker_model_invoke()

        # TODO: do we need it here as self.stop_workers() and self.worker_model_invoke()
        #       already have barrier
        safe_barrier(self._distributed.world_group, f"lm_eval Run end")

    def _model_invoke(
        self,
        input_ids,
        attention_mask,
        labels,
        max_length,
        stop,
        generate: bool,
        continue_generate: bool,
        **generation_kwargs,
    ):
        # TODO: Consider passing true messages and payloads around instead of combining all data into a large tuple.
        # Messages could include types like logits, generate, finished.

        # Group is always None if world size is 1
        if self._group is None:
            # Must not be called with continue_generate false on one process
            assert continue_generate
            return self._model_invoke_inner(
                input_ids, attention_mask, labels, max_length, stop, generate, **generation_kwargs
            )

        world_size = self._group.size()

        assert self._group.rank() == 0

        if continue_generate:
            assert input_ids is not None
            if generate:
                assert max_length is not None and stop is not None

            # always divide by world_size, if not full batch, some ranks will get less work or not at all
            assert self._batch_size % world_size == 0
            step = self._batch_size // world_size

            input_ids = [input_ids[i * step : (i + 1) * step] for i in range(world_size)]
            attention_mask = [
                attention_mask[i * step : (i + 1) * step] if attention_mask is not None else None
                for i in range(world_size)
            ]
            labels = [labels[i * step : (i + 1) * step] if labels is not None else None for i in range(world_size)]

            scatter_list = [
                [
                    input_ids[i],
                    attention_mask[i],
                    labels[i],
                    max_length,
                    stop,
                    generate,
                    continue_generate,
                    generation_kwargs,
                ]
                for i in range(world_size)
            ]
        else:
            scatter_list = [[None, None, None, None, None, None, False, None] for _ in range(world_size)]

        input_ids, attention_mask, labels, max_length, stop, generate, continue_generate, generation_kwargs = (
            scatter_object(
                scatter_list,
                group=self._group,
            )
        )

        if not continue_generate:
            return None

        assert len(input_ids) > 0

        result = self._model_invoke_inner(
            input_ids, attention_mask, labels, max_length, stop, generate, **generation_kwargs
        )

        gather_list = gather_object(result, group=self._group)
        # Clean gather list from empty shards
        gather_list = [el for el in gather_list if len(el) > 0]

        # If it was model generate tensors could be of different length
        # so we aggregate results to list instead of a tensor
        if generate:
            result = sum((el.tolist() for el in gather_list), [])
        else:
            assert all(el.device.type == "cpu" for el in gather_list)
            result = torch.cat(gather_list, dim=0)

        return result

    def worker_model_invoke(self):
        assert self._group is not None
        # if isinstance(self.group, dist.ProcessGroup):
        if not isinstance(self._group, int):
            # groups is None for world_size 1
            assert self._group.rank() != 0
            # on worker ranks the function need to wait to be called multiple times
            while True:
                input_ids, attention_mask, labels, max_length, stop, generate, continue_generate, generation_kwargs = (
                    scatter_object(
                        None,
                        group=self._group,
                    )
                )

                # Stop signal was send, end waiting/processing loop
                if not continue_generate:
                    break

                # if some data was received, work, otherwise return empty tensor
                if len(input_ids) > 0:
                    result = self._model_invoke_inner(
                        input_ids, attention_mask, labels, max_length, stop, generate, **generation_kwargs
                    )
                else:
                    result = input_ids

                gather_object(result, group=self._group)
        else:
            # TODO: implement distributed model support
            assert self._group == torch.distributed.GroupMember.NON_GROUP_MEMBER
        safe_barrier(self._distributed.world_group, "lm_eval_end")

    def stop_workers(self):
        # Group is always None if world size is 1
        if self._group is None:
            return
        self._model_invoke(None, None, None, None, None, None, continue_generate=False)
        safe_barrier(self._distributed.world_group, "lm_eval_end")

    def _model_invoke_inner(
        self, input_ids, attention_mask, labels, max_length, stop, generate: bool, **generation_kwargs
    ):
        if generate:
            return self._model_generate_inner(input_ids, attention_mask, max_length, stop, **generation_kwargs)
        else:
            return self._model_call_inner(input_ids, attention_mask, labels)

    def _model_call(self, input_ids, attention_mask=None, labels=None):
        return self._model_invoke(
            input_ids, attention_mask, labels, None, None, generate=False, continue_generate=True
        )

    def _model_generate(self, input_ids, attention_mask, max_length, stop, **generation_kwargs):
        return self._model_invoke(
            input_ids,
            attention_mask,
            None,
            max_length,
            stop,
            generate=True,
            continue_generate=True,
            **generation_kwargs,
        )

    def _model_call_inner(self, input_ids, attention_mask=None, labels=None):
        """
        :param input_ids: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :param attention_mask: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :param labels: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        if attention_mask is not None or labels is not None:
            assert attention_mask is not None and labels is not None

        # TODO: do we need no_grad for fast_llm model?
        with torch.no_grad():
            # We move logits to the CPU because they will be copied across processes and nodes
            # in a multi-GPU, multi-node setup and eventually collected on the main rank.
            # We cannot afford to accumulate them on rank 0 GPU, as GPU memory may already be tight.
            # CPU tensors are slower, but we typically have much more CPU RAM available.

            # TODO: Check if it's possible to move some of the _loglikelihood_tokens work here
            # and pass only the results around instead of the full logits.
            # Computing errors here is also preferable, as copying logits across nodes and GPUs
            # is inefficient and can involve gigabytes of data.
            return self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            ).logits.cpu()

    def _model_generate_inner(self, input_ids, attention_mask, max_length, stop, **generation_kwargs):
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
        stopping_criteria = lm_eval.models.utils.stop_sequences_criteria(
            self._tokenizer, stop, input_ids.shape[1], input_ids.shape[0]
        )

        return self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self._tokenizer.pad_token_id,
            use_cache=False,
            **generation_kwargs,
        )

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> list[int]:
        """ """
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_bos_token is set
        if add_special_tokens is None:
            if self._backend == "causal":
                special_tokens_kwargs = {"add_special_tokens": False or self._add_bos_token}
        # otherwise the method explicitly defines the value
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self._tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def tok_batch_encode(
        self,
        strings: list[str],
        padding_side: str = "left",
        left_truncate_len: int = None,
        truncation: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # encode a batch of strings. converts to tensors and pads automatically, unlike tok_encode.
        old_padding_side = self._tokenizer.padding_side
        self._tokenizer.padding_side = padding_side

        add_special_tokens = {}
        if self._backend == "causal":
            add_special_tokens = {"add_special_tokens": False or self._add_bos_token}

        encoding = self._tokenizer(
            strings,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            **add_special_tokens,
        )
        if left_truncate_len:
            original_lengths = encoding["input_ids"].size(1)
            if original_lengths > left_truncate_len:
                logger.warn(
                    f"Left truncation applied. Original sequence length was {original_lengths}, "
                    f"truncating to last {left_truncate_len} tokens. Some content will be lost.",
                )
            encoding["input_ids"] = encoding["input_ids"][:, -left_truncate_len:]
            encoding["attention_mask"] = encoding["attention_mask"][:, -left_truncate_len:]
        self._tokenizer.padding_side = old_padding_side

        return encoding["input_ids"], encoding["attention_mask"]

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self._tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _select_cont_toks(self, logits: torch.Tensor, contlen: int = None, inplen: int = None) -> torch.Tensor:
        if self._backend == "causal":
            assert contlen and inplen, "Must pass input len and cont. len to select scored logits for causal LM"
            # discard right-padding.
            # also discard the input/context tokens. we'll only score continuations.
            logits = logits[inplen - contlen : inplen]
        elif self._backend == "seq2seq":
            assert contlen and not inplen, "Selecting scored logits for Seq2SeqLM requires only cont. len"
            # only discard right-padding.
            # the logits input to this fn only contain decoder-side tokens.
            logits = logits[:contlen]

        return logits

    def loglikelihood_rolling(
        self, requests: list[lm_eval.api.instance.Instance], disable_tqdm: bool = False
    ) -> list[float]:
        adaptive_batch_size = None
        if self._batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        # First, collect all windows from all requests
        all_windows = []  # List of (request_idx, window) tuples
        request_window_counts = []  # Track number of windows per request

        for req_idx, (string,) in enumerate(
            tqdm.auto.tqdm(
                [req.args for req in requests],
                disable=(disable_tqdm or (self.rank != 0)),
            )
        ):
            # The tokenizer may raise: "Token indices sequence length is longer than the specified maximum sequence length for this model"
            # This is expected and fine, as the sequence will be split into chunks of max_length later.
            rolling_token_windows: list[tuple[list[int], list[int]]] = list(
                map(
                    lm_eval.utils.make_disjoint_window,
                    lm_eval.utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # TODO: Right now, we pass single EOT token to the Encoder and the full context to the decoder, in seq2seq case
            windows = [(None,) + x for x in rolling_token_windows]

            # Store windows with their request index
            all_windows.extend((req_idx, window) for window in windows)
            request_window_counts.append(len(windows))

        # Handle distributed case padding
        pad_amnt = 0
        if self.world_size > 1:
            mytensor = torch.tensor(len(all_windows), device=self._device)
            gathered = self.accelerator.gather(mytensor).cpu().detach().numpy().tolist()
            pad_amnt = max(gathered) - gathered[self.rank]
            if pad_amnt > 0:
                all_windows += pad_amnt * [all_windows[0]]

        all_nlls = []
        batch_size = adaptive_batch_size or self._batch_size
        for i in range(0, len(all_windows), batch_size):
            batch = all_windows[i : i + batch_size]
            # Extract just the windows for processing, keeping track of request indices
            batch_indices, batch_windows = zip(*batch)

            batch_nlls = self._loglikelihood_tokens(
                requests=batch_windows,
                disable_tqdm=False,
                override_bs=len(batch_windows),
            )
            # Store results with their request indices
            all_nlls.extend(zip(batch_indices, batch_nlls))

        # Remove padding if necessary
        if (self.world_size > 1) and (pad_amnt > 0):
            all_nlls = all_nlls[:-pad_amnt]

        # Reconstruct per-request loglikelihoods
        loglikelihoods = []
        current_idx = 0
        for window_count in request_window_counts:
            # Get all nlls for this request
            request_nlls = all_nlls[current_idx : current_idx + window_count]
            # Sum up the nlls for this request (discarding is_greedy)
            request_total = sum(nll[0] for _, nll in request_nlls)
            loglikelihoods.append(request_total)
            current_idx += window_count

            string = requests[len(loglikelihoods) - 1].args[0]
            self.cache_hook.add_partial("loglikelihood_rolling", (string,), request_total)

        return loglikelihoods

    def _batch_scheduler(self, pos, n_reordered_requests):
        sched = pos // int(len(n_reordered_requests) / self._batch_schedule)
        if sched in self._batch_sizes:
            return self._batch_sizes[sched]
        if (len(self._batch_sizes) > 1) and (self._batch_sizes[sched - 1] == self._max_batch_size):
            # if previous batch size is already maximal, skip recomputation
            self._batch_sizes[sched] = self._max_batch_size
            return self._batch_sizes[sched]
        print(f"Passed argument batch_size = auto:{self._batch_schedule}. Detecting largest batch size")
        self._batch_sizes[sched] = self._detect_batch_size(n_reordered_requests, pos)
        print(f"Determined largest batch size: {self._batch_sizes[sched]}")
        return self._batch_sizes[sched]

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int = None,
    ) -> list[tuple[float, bool]]:
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        # NOTE: for the sort_fn, the negative sign on len(toks) sorts descending - this has a few advantages:
        # - time estimates will always be over not underestimates, which is more useful for planning
        # - to know the size of a batch when going through the list, you know the first one is always the batch
        #   padded context length. this is useful to simplify the batching logic and more importantly to make
        #   automatic adaptive batches much much easier to implement
        # - any OOMs will happen right away rather than near the end
        # NOTE: the group_fn  Defines the key to group and lookup one-token continuations
        # Use with group_by="contexts" (optional)"
        # allows for the creation of a lookup, so we can reuse logits in case of one-token continuations.
        # speeds up some multiple-choice tasks proportionally to the number of choices.
        # groups requests by context+continuation[:-1] and infer on one request/group.
        re_ord = lm_eval.models.utils.Collator(
            requests,
            sort_fn=lambda req: (-(len(req[1]) + len(req[2])), tuple(req[1]) + tuple(req[2])),
            group_by="contexts" if self._backend == "causal" and self._logits_cache else None,
            group_fn=lambda req: req[-2] + req[-1][:-1],
        )

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        n_reordered_requests = len(re_ord)
        batch_size = self._batch_size if self._batch_size != "auto" else override_bs if override_bs is not None else 0
        batch_fn = (
            self._batch_scheduler
            if self._batch_size == "auto" and n_reordered_requests > 0 and not override_bs
            else None
        )

        chunks = re_ord.get_batched(n=batch_size, batch_fn=batch_fn)
        pbar = tqdm.auto.tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            encoder_attns = []

            padding_len_inp = None
            padding_len_cont = None
            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                if self._backend == "causal":
                    total_length = len(context_enc) + len(continuation_enc)
                    if total_length > self.max_length + 1:
                        logger.warning(
                            f"Combined length of context ({len(context_enc)}) and continuation ({len(continuation_enc)}) "
                            f"exceeds model's maximum length ({self.max_length}). "
                            f"Truncating {total_length - self.max_length + 1} tokens from the left."
                        )
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                        dtype=torch.long,
                        device=self._device,
                    )
                    (inplen,) = inp.shape
                elif self._backend == "seq2seq":
                    inp = torch.tensor(
                        (context_enc)[-self.max_length :],
                        dtype=torch.long,
                        device=self._device,
                    )
                    (inplen,) = inp.shape

                    # build encoder attn masks
                    encoder_attns.append(torch.ones_like(inp))

                    cont = torch.tensor(
                        (continuation_enc)[-self.max_length :],
                        # TODO: left-shift these?
                        # TODO: our code assumes we never end up truncating conts for either model type
                        dtype=torch.long,
                        device=self._device,
                    )
                    (contlen,) = cont.shape

                    conts.append(cont)

                    padding_len_cont = max(padding_len_cont, contlen) if padding_len_cont is not None else contlen

                padding_len_inp = max(padding_len_inp, inplen) if padding_len_inp is not None else inplen

                inps.append(inp)  # [1, inp_length]
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)

            # create encoder attn mask and batched conts, if seq2seq
            call_kwargs = {}
            if self._backend == "causal":
                batched_inps = lm_eval.models.utils.pad_and_concat(
                    padding_len_inp, inps, padding_side="right"
                )  # [batch, padding_len_inp]
            elif self._backend == "seq2seq":
                # TODO: left-pad encoder inps and mask?
                batched_inps = lm_eval.models.utils.pad_and_concat(padding_len_inp, inps)  # [batch, padding_len_inp]
                batched_conts = lm_eval.models.utils.pad_and_concat(
                    padding_len_cont, conts
                )  # [batch, padding_len_cont]
                batched_encoder_mask = lm_eval.models.utils.pad_and_concat(
                    padding_len_inp, encoder_attns
                )  # [batch, padding_len_inp]
                call_kwargs = {
                    "attention_mask": batched_encoder_mask,
                    "labels": batched_conts,
                }

            multi_logits = F.log_softmax(
                self._model_call(batched_inps, **call_kwargs), dim=-1
            )  # [batch, padding_length (inp or cont), vocab]

            # TODO: Consider moving this part to per-shard execution in a multi-GPU and multi-node setup
            # to avoid copying logits between GPUs and nodes, and to enable performing logits computations on the GPU.
            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = inplen + (logits.shape[0] - padding_len_inp) if self._backend == "causal" else None
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    # NOTE: Currently, computations are performed on the CPU due to limited GPU memory.
                    cont_toks = torch.tensor(cont_toks, dtype=torch.long, device="cpu").unsqueeze(0)  # [1, seq]

                    max_equal = (greedy_tokens == cont_toks).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)

                    if request_str is not None:
                        # special case: loglikelihood_rolling produces a number of loglikelihood requests
                        # all with cache key None. instead do add_partial on the per-example level
                        # in the loglikelihood_rolling() function for those.
                        self.cache_hook.add_partial("loglikelihood", request_str, answer)
                    pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)

    def generate_until(self, requests: list[lm_eval.api.instance.Instance], disable_tqdm: bool = False) -> list[str]:
        res = []

        pbar = tqdm.auto.tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )
        adaptive_batch_size = None
        if self._batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size
        # for each different set of kwargs, we execute all requests, by batch.
        batch_size = (
            self._batch_size
            if self._batch_size != "auto"
            else adaptive_batch_size if adaptive_batch_size is not None else 0
        )
        batch_fn = self._batch_scheduler if self._batch_size == "auto" and not adaptive_batch_size else None

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        # group_fn=lambda x: x[1] -> x=(context, gen_kwargs)
        # NOTE: for sort_fn, the negative sign on len(toks) sorts descending - this has a few advantages:
        # - time estimates will always be over not underestimates, which is more useful for planning
        # - to know the size of a batch when going through the list, you know the first one is always the batch
        #   padded context length. this is useful to simplify the batching logic and more importantly to make
        #   automatic adaptive batches much much easier to implement
        # - any OOMs will happen right away rather than near the end
        re_ords = lm_eval.models.utils.Collator(
            [reg.args for reg in requests],
            sort_fn=lambda req: (-len(self.tok_encode(req[0])), req[0]),
            group_by="gen_kwargs",
            group_fn=lambda x: x[1],
        )
        chunks = re_ords.get_batched(n=batch_size, batch_fn=batch_fn)
        eos = self.tok_decode(self.eot_token_id, skip_special_tokens=False)

        for chunk in chunks:
            contexts, all_gen_kwargs = zip(*chunk)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            # unpack our keyword arguments.
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)  # edge case for repeats > 1
                # add EOS token to stop sequences
                until = lm_eval.models.utils.handle_stop_sequences(kwargs.pop("until", None), eos=eos)
            else:
                raise ValueError(f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}")
            if "max_gen_toks" in kwargs.keys():
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self._DEFAULT_MAX_GEN_TOKENS

            # set the max length in tokens of inputs ("context_enc")
            if self._backend == "causal":
                # max len for inputs = max length, minus room to generate the max new tokens
                max_ctx_len = self.max_length - max_gen_toks
                assert (
                    max_ctx_len > 0
                ), f"Invalid configuration: requested max tokens to generate ({max_gen_toks}) must be less than model's maximum sequence length ({self.max_length})."
            elif self._backend == "seq2seq":
                # max len for inputs = encoder's whole max_length
                max_ctx_len = self.max_length

            # encode, pad, and truncate contexts for this batch
            input_ids, attention_mask = self.tok_batch_encode(
                contexts,
                left_truncate_len=max_ctx_len,
                truncation=self._truncation,
            )
            input_ids = input_ids.to(self._device)
            attention_mask = attention_mask.to(self._device)

            if "max_length" not in kwargs:
                kwargs["max_length"] = input_ids.shape[1] + max_gen_toks

            # perform batched generation
            cont = self._model_generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                stop=until,
                **kwargs,
            )

            # cont_toks_list = cont.tolist()
            cont_toks_list = cont

            for cont_toks, context in zip(cont_toks_list, contexts):
                # discard context + left-padding toks if using causal decoder-only LM
                if self._backend == "causal":
                    cont_toks = cont_toks[input_ids.shape[1] :]

                s = self.tok_decode(cont_toks)

                # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                for term in until:
                    if len(term) > 0:
                        # ignore '' separator,
                        # for seq2seq case where self.tok_decode(self.eot_token_id) = ''
                        s = s.split(term)[0]

                res.append(s)

                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), s)
                pbar.update(1)
        # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()

        return res

    def apply_chat_template(self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        try:
            chat_templated = self._tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )
        except jinja2.exceptions.TemplateError:
            logger.warning("Failed to apply chat template. removing the system role in chat history.")
            chat_history = [msg for msg in chat_history if msg["role"] != "system"]
            chat_templated = self._tokenizer.apply_chat_template(
                chat_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                continue_final_message=not add_generation_prompt,
            )

        return chat_templated
