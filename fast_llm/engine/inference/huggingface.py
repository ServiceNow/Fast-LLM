import logging
import os
import pathlib
import typing

import torch
import transformers.generation.utils
import transformers.modeling_outputs

from fast_llm.core.distributed import broadcast_object, broadcast_optional, safe_barrier
from fast_llm.engine.checkpoint.config import CheckpointLoadConfig, FastLLMCheckpointFormat
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.inference.config import HuggingfaceModelConfig
from fast_llm.engine.inference.runner import InferenceRunner
from fast_llm.engine.multi_stage.config import StageMode
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class HuggingfacePreTrainedModel(transformers.PreTrainedModel):
    config_class: typing.ClassVar[type[HuggingfaceModelConfig]] = HuggingfaceModelConfig
    runner_class: typing.ClassVar[type[InferenceRunner]] = InferenceRunner
    config: HuggingfaceModelConfig
    # base_model_prefix = ""
    # _no_split_modules = None
    # _supports_cache_class = False
    # _tied_weights_keys = []

    def __init__(
        self,
        fast_llm_model: FastLLMModel,
        config: HuggingfaceModelConfig | None = None,
        runner: ScheduleRunner | None = None,
        **kwargs,
    ):
        if config is None:
            config = self.config_class(fast_llm_model.config)

        assert self.runner_class.model_class.config_class is config.model_config_class
        assert config.fast_llm_config is fast_llm_model.config
        assert isinstance(config, self.config_class)

        # The HF constructor performs a deep copy of the config,
        # but config.fast_llm_config may contain non-picklable items like process groups.
        # Temporarily remove it before the call and restore it afterward.
        # TODO: Find a clean solution — overriding __deepcopy__ doesn't work here
        # because internally they use copy.deepcopy(self.__dict__).
        fast_llm_config = config.fast_llm_config
        config.fast_llm_config = None
        super().__init__(config, **kwargs)
        config.fast_llm_config = fast_llm_config

        self._inference_runner = self.runner_class(fast_llm_model, runner)

        # A model can be created from pretrained which set it up in the current HF wrapper api
        # or set existing model which  also must be setup, so, do not accept not setup model
        assert fast_llm_model.is_setup

        self._inference_runner.setup()

        # We only support data parallel and tensor parallel for now
        Assert.eq(fast_llm_model.distributed.config.pipeline_parallel, 1)
        Assert.eq(fast_llm_model.distributed.config.sequence_data_parallel, 1)

        # Transformers needs to be able to inspect the base model.
        self.fast_llm_base_model = fast_llm_model.base_model

        with transformers.modeling_utils.no_init_weights():
            self.post_init()

        if fast_llm_model.config.multi_stage.zero_stage == 3:
            logger.warning(
                "zero_stage=3 is used for the model; forward and generate will be extremely slow during inference."
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | CheckpointLoadConfig,
        *updates: dict[str | tuple[str, ...], typing.Any],
        optimizer_state_names: tuple[str, ...] | None = None,
        # setup: bool = True,
        mode: StageMode = StageMode.training,
        use_cpu: bool = False,
        stage_filter: set | None = None,
        **kwargs,
    ) -> typing.Self:
        # Pretrained config.
        if not isinstance(pretrained_model_name_or_path, CheckpointLoadConfig):
            pretrained_model_name_or_path = CheckpointLoadConfig(
                path=pathlib.Path(pretrained_model_name_or_path),
                format=FastLLMCheckpointFormat,
            )

        # Create the model
        # always set up model and create distributed instance internally for now
        fast_llm_model = cls.runner_class.model_class.from_pretrained(
            pretrained_model_name_or_path,
            *updates,
            optimizer_state_names=optimizer_state_names,
            setup=True,
            mode=mode,
            use_cpu=use_cpu,
            stage_filter=stage_filter,
        )

        return cls(fast_llm_model, **kwargs)

    def _init_weights(self, module) -> None:
        raise NotImplementedError(module)


class HuggingfaceBaseModelForCausalLM(HuggingfacePreTrainedModel, transformers.generation.utils.GenerationMixin):
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
        # Meant to be overridden in derived classes
        raise NotImplementedError()

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
        coordinator_forward: bool = False,
        communication_timeout_sec: float = 600.0,
        continue_work: bool = True,
    ) -> tuple | transformers.modeling_outputs.CausalLMOutputWithPast | None:
        """
        Forward pass compatible with HuggingFace forward.

        Additional arguments:
            coordinator_forward (bool):
                If True, only the TP group coordinator (rank 0) should call forward;
                other ranks must call worker_forward.
                If False, all TP group ranks call forward independently and return logits.
            communication_timeout_sec (float): Maximum time (in seconds) to wait for the start of
                forward or for a stop signal to worker ranks before timing out in worker_forward.
                Must match the value passed to worker_forward.
            continue_work (bool): Whether to continue processing in a TP group.
                Only applies for coordinator_forward=True.

        Notes:
            - In coordinator_forward=True mode, forward on rank 0 distributes data to other ranks.
            - After processing, the coordinator (rank 0) must call `stop_workers()` before continuing,
            to unblock worker_forward on other ranks.
            - This mode augments HuggingFace generate with tensor-parallel capability.
        """
        distributed: Distributed = self._inference_runner._fast_llm_model.distributed

        if coordinator_forward and distributed.world_group and distributed.tensor_group:
            assert distributed.tensor_group.rank() == 0
            assert past_key_values is None and not use_cache

            # Some tasks may post-process too slowly, so waiting for the next batch or
            # the end of work can exceed the standard 60s timeout.
            safe_barrier(distributed.tensor_group, "forward_wait", timeout=communication_timeout_sec)

            broadcast_optional(input_ids, distributed.tensor_group, 0)
            broadcast_optional(attention_mask, distributed.tensor_group, 0)
            broadcast_optional(position_ids, distributed.tensor_group, 0)
            broadcast_optional(inputs_embeds, distributed.tensor_group, 0)
            broadcast_optional(labels, distributed.tensor_group, 0)

            broadcast_object(
                (past_key_values, use_cache, output_attentions, output_hidden_states, return_dict, continue_work),
                distributed.tensor_group,
                0,
            )

        if not coordinator_forward or continue_work:
            return self.inner_forward(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
            )

        return None

    def worker_forward(
        self,
        communication_timeout_sec: float = 600.0,
    ):
        """
        Run the forward loop on worker ranks in coordinated mode.

        This function must be called on all worker ranks (i.e., all ranks except the
        coordinator/leading data-parallel rank). In coordinated mode, the coordinator
        rank calls `forward`, which distributes inputs to workers. Each worker then
        receives its inputs and runs a forward pass.

        Workers stay in this loop until a stop signal is broadcast, which happens when
        the coordinator rank calls `stop_workers`.

        Args:
            communication_timeout_sec (float): Maximum time (in seconds) to wait for the
                start of a forward call or for a stop signal from the coordinator before
                timing out. Must match the value passed to `forward`.

        Notes:
            - Coordinator rank: calls `forward` in coordinated mode and later
              `stop_workers` to unblock workers.
            - Worker ranks: call `worker_forward` once and remain inside the loop,
              executing forward passes with broadcasted inputs until a stop signal
              is received.
        """
        distributed: Distributed = self._inference_runner._fast_llm_model.distributed
        assert distributed.world_group and distributed.tensor_group and distributed.tensor_group.rank() != 0

        while True:
            # Some tasks may post-process too slowly, so waiting for the next batch or
            # the end of work can exceed the standard 60s timeout.
            safe_barrier(distributed.tensor_group, "forward_wait", timeout=communication_timeout_sec)

            input_ids = broadcast_optional(None, distributed.tensor_group, 0)
            attention_mask = broadcast_optional(None, distributed.tensor_group, 0)
            position_ids = broadcast_optional(None, distributed.tensor_group, 0)
            inputs_embeds = broadcast_optional(None, distributed.tensor_group, 0)
            labels = broadcast_optional(None, distributed.tensor_group, 0)

            past_key_values, use_cache, output_attentions, output_hidden_states, return_dict, continue_work = (
                broadcast_object(None, distributed.tensor_group, 0)
            )

            if not continue_work:
                break

            self.inner_forward(
                input_ids,
                attention_mask,
                position_ids,
                past_key_values,
                inputs_embeds,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,
            )

        safe_barrier(distributed.world_group, "forward_work_end")

    def stop_workers(self):
        distributed: Distributed = self._inference_runner._fast_llm_model.distributed
        # On single gpu or no tp, no worker_forward to stop
        if distributed.world_group is None or distributed.tensor_group is None:
            return
        self.forward(coordinator_forward=True, continue_work=False)
        safe_barrier(distributed.world_group, "forward_work_end")
