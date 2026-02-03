import logging
import os
import pathlib
import typing

import torch
import transformers.generation.utils
import transformers.modeling_outputs
import transformers.utils.generic

from fast_llm.core.distributed import broadcast, broadcast_object, safe_barrier
from fast_llm.engine.checkpoint.config import CheckpointLoadConfig, FastLLMCheckpointFormat
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.inference.config import HuggingfaceModelConfig
from fast_llm.engine.inference.runner import InferenceRunner
from fast_llm.engine.multi_stage.config import StageMode
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.schedule.runner import ScheduleRunner
from fast_llm.utils import Assert

logger = logging.getLogger(__name__)


class HuggingfacePreTrainedModel(transformers.PreTrainedModel, transformers.generation.utils.GenerationMixin):
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
            stage_filter=stage_filter,
        )

        return cls(fast_llm_model, **kwargs)

    def _init_weights(self, module) -> None:
        raise NotImplementedError(module)

    def forward(
        self,
        *args,
        coordinator_forward: bool = False,
        communication_timeout_sec: float = 600.0,
        continue_work: bool = True,
        **kwargs,
    ) -> tuple | transformers.utils.generic.ModelOutput | None:
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

            # Some tasks may post-process too slowly, so waiting for the next batch or
            # the end of work can exceed the standard 60s timeout.
            safe_barrier(distributed.tensor_group, "forward_wait", timeout=communication_timeout_sec)

            # Broadcast all input arguments, handling tensor and non-tensor arguments separately
            # TODO: Support nested tensor in arguments (ex. past_key_values)
            # TODO: Bypassed if passed as positional argument.
            assert kwargs.get("past_key_values") is None and not kwargs.get("use_cache")
            broadcast_kwargs = {**kwargs, **{i: arg for i, arg in enumerate(args)}, "continue_work": continue_work}
            tensor_kwargs = {key: value for key, value in broadcast_kwargs if torch.is_tensor(value)}
            broadcast_object(
                [(key, tensor.shape, tensor.dtype) for key, tensor in tensor_kwargs.items()],
                distributed.tensor_group,
                0,
            )
            for tensor in tensor_kwargs.values():
                broadcast(tensor.to(distributed.device), 0, distributed.tensor_group)
            non_tensor_kwargs = {key: value for key, value in broadcast_kwargs if key not in tensor_kwargs}
            broadcast_object(
                non_tensor_kwargs,
                distributed.tensor_group,
                0,
            )

        if not coordinator_forward or continue_work:
            return self.inner_forward(*args, **kwargs)

        return None

    def worker_forward(self, communication_timeout_sec: float = 600.0):
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

            broadcast_kwargs = {}
            for key, shape, dtype in broadcast_object(None, distributed.tensor_group, 0):
                tensor = torch.empty(shape, dtype=dtype, device=distributed.device)
                broadcast(tensor, 0, distributed.tensor_group)
                broadcast_kwargs[key] = tensor

            broadcast_kwargs.update(
                broadcast_object(
                    None,
                    distributed.tensor_group,
                    0,
                )
            )

            if not broadcast_kwargs.pop("continue_work"):
                break

            arg_kwargs = {key: value for key, value in broadcast_kwargs.items() if isinstance(key, int)}

            self.inner_forward(
                *(arg_kwargs[i] for i in range(len(arg_kwargs))),
                **{key: value for key, value in broadcast_kwargs.items() if key not in arg_kwargs},
            )

        safe_barrier(distributed.world_group, "forward_work_end")

    def stop_workers(self):
        distributed: Distributed = self._inference_runner._fast_llm_model.distributed
        # On single gpu or no tp, no worker_forward to stop
        if distributed.world_group is None or distributed.tensor_group is None:
            return
        self.forward(coordinator_forward=True, continue_work=False)
        safe_barrier(distributed.world_group, "forward_work_end")

    def inner_forward(*args, **kwargs) -> tuple | transformers.utils.generic.ModelOutput:
        # Meant to be overridden in derived classes
        raise NotImplementedError()

    @classmethod
    def can_generate(cls) -> bool:
        return True
