import json
import logging
import typing

import torch.distributed

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
        if self._do_broadcast:
            self._client = self._config.get_client()
            init_method = f"tcp://{config.broadcast.host}:{config.broadcast.port}"
            logger.info(f"Waiting for weights broadcast rendezvous at {init_method} ...")
            # TODO: Create a custom process group instead.
            self._process_group = torch.distributed.init_process_group(
                backend="nccl",
                init_method=init_method,
                world_size=config.broadcast.external_world_size + 1,
                rank=0,
            )
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
    ):
        if update_successful:
            self._broadcast_weights(step)

    def train_end(self, step: int):
        # TODO: ====== Send something on unsuccessful ends? ======
        if self._do_broadcast:
            self._client.xadd(REDIS_TRAINING_STREAM, {REDIS_TRAINING_FIELD: json.dumps({"type": "training_finished"})})
        self._clear()

    def __del__(self):
        self._clear()

    def _clear(self):
        if hasattr(self, "_process_group"):
            torch.distributed.destroy_process_group(self._process_group)
            del self._process_group

    def _broadcast_weights(self, step: int):
        if self._do_broadcast:
            self._client.xadd(
                REDIS_TRAINING_STREAM, {REDIS_TRAINING_FIELD: json.dumps({"type": "weights_ready", "step": step})}
            )
        for shard_name, layer_name, tensor in self._model.iter_checkpoint(self._config.export, {}):
            if self._do_broadcast:
                # TODO: ====== Broadcast metadata in advance =======
                meta = [(shard_name, layer_name, tensor.shape, tensor.dtype)]
                torch.distributed.broadcast_object_list(meta, group=self._process_group, group_src=0)
                torch.distributed.broadcast(tensor, group=self._process_group, group_src=0)
        # Broadcast end of weights broadcast
        if self._do_broadcast:
            meta = [None]
            torch.distributed.broadcast_object_list(meta, group=self._process_group, group_src=0)
