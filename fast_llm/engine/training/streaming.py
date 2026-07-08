import json
import logging
import time
import typing

import torch

from fast_llm.core.distributed import broadcast as _broadcast
from fast_llm.core.distributed import broadcast_object as _broadcast_object
from fast_llm.engine.distributed.config import DistributedBackend
from fast_llm.engine.distributed.distributed import ProcessGroupPool
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.training.config import StreamingTrainerCallbackConfig, TrainerCallback

logger = logging.getLogger(__name__)


REDIS_TRAINING_STREAM = "fast_llm_events"
REDIS_TRAINING_FIELD = "event"


class StreamingTrainerCallback[ConfigType: StreamingTrainerCallbackConfig](TrainerCallback[ConfigType]):
    def __init__(self, config: ConfigType, model: "FastLLMModel"):
        super().__init__(config)
        self._model = model
        self._do_broadcast = self._model.config.distributed.rank == 0
        # Weight-sync timing, measured on the broadcasting rank only. On CUDA we bracket the sync with
        # events and read the result one step later (`_weight_sync_pending`), so the always-on timer
        # never forces a CPU stall on the broadcast, which otherwise overlaps the next step's compute.
        self._use_cuda_timing = (
            self._do_broadcast
            and self._config.broadcast.backend == DistributedBackend.nccl
            and torch.cuda.is_available()
        )
        self._weight_sync_time_ms: float | None = None
        self._weight_sync_pending = False
        if self._use_cuda_timing:
            self._weight_sync_start = torch.cuda.Event(enable_timing=True)
            self._weight_sync_end = torch.cuda.Event(enable_timing=True)
        if self._do_broadcast:
            self._client = self._config.get_client()
            init_method = f"tcp://{config.broadcast.host}:{config.broadcast.port}"
            logger.info(f"Waiting for weights broadcast rendezvous at {init_method} ...")
            world_size = config.broadcast.external_world_size + 1
            self._pool = ProcessGroupPool(
                rank=0,
                world_size=world_size,
                local_world_size=1,
                timeout=self._config.timeout,
                device=None if self._config.broadcast.backend == DistributedBackend.nccl else torch.device("cpu"),
                init_method=init_method,
                backend=self._config.broadcast.backend,
            )
            self._process_group = self._pool.get_process_group(range(world_size), 0)
            logger.info(f"Weights broadcast rendezvous at {init_method} connected")

    def run_begin(self, step: int):
        # TODO: ====== Send a train / run begin signal? ======
        self._broadcast_weights(step)

    def step_end(
        self,
        step: int,
        reduced_losses: dict[str, float | int],
        update_successful: bool,
        train_metrics: dict[str, typing.Any] | None,
    ) -> dict[str, typing.Any] | None:
        # Harvest the previous broadcast's timing before re-recording the events on this step's sync.
        if self._weight_sync_pending and self._weight_sync_end.query():
            self._weight_sync_time_ms = self._weight_sync_start.elapsed_time(self._weight_sync_end)
            self._weight_sync_pending = False
        if update_successful:
            self._broadcast_weights(step)
        if self._weight_sync_time_ms is None:
            return None
        return {"weight_sync_time_ms": self._weight_sync_time_ms}

    def train_end(self, step: int):
        # TODO: ====== Send something on unsuccessful ends? ======
        if self._do_broadcast:
            self._client.xadd(REDIS_TRAINING_STREAM, {REDIS_TRAINING_FIELD: json.dumps({"type": "training_finished"})})
        self._clear()

    def __del__(self):
        self._clear()

    def _clear(self):
        if hasattr(self, "_pool"):
            self._pool.shutdown()
            del self._pool
            del self._process_group

    def _broadcast_weights(self, step: int):
        if self._do_broadcast:
            self._client.xadd(
                REDIS_TRAINING_STREAM, {REDIS_TRAINING_FIELD: json.dumps({"type": "weights_ready", "step": step})}
            )
        if self._use_cuda_timing:
            self._weight_sync_start.record()
        elif self._do_broadcast:
            sync_start_time = time.perf_counter()
        for shard_name, layer_name, tensor in self._model.iter_checkpoint(self._config.export, {}):
            if self._do_broadcast:
                # TODO: ====== Broadcast metadata in advance =======
                _broadcast_object((shard_name, layer_name, tensor.shape, tensor.dtype), self._process_group, src=0)
                _broadcast(tensor, 0, self._process_group)
        # Broadcast end of weights broadcast
        if self._do_broadcast:
            _broadcast_object(None, self._process_group, src=0)
        if self._use_cuda_timing:
            self._weight_sync_end.record()
            self._weight_sync_pending = True
        elif self._do_broadcast:
            self._weight_sync_time_ms = (time.perf_counter() - sync_start_time) * 1000
