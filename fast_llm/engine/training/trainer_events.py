import logging

import orjson
import redis
import torch.distributed

from fast_llm.engine.config_utils.run import is_main_rank
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.training.config import RedisEventMessageConfig, TrainerEventsConfig, TrainingExportConfig

logger = logging.getLogger(__name__)


class OptionalEventSender:
    """
    Internal helper: wraps a single event config and a redis client.
    If config is None, .send() becomes a no-op.
    """

    def __init__(self, config: RedisEventMessageConfig):
        self.config = config
        self.client = None

        if self.config.enabled and is_main_rank():
            # Each event may use a separate Redis server
            self.client = redis.Redis(
                host=config.host,
                port=config.port,
            )

    @property
    def enabled(self):
        return self.client.enabled

    def send(self, payload: dict | None = None):
        """
        Send an event message as a Redis stream entry.
        If config is None → does nothing.
        """
        if not self.config.enabled or not is_main_rank():
            return  # No event configured or not main rank → skip

        msg_key = self.config.message_key
        msg_val = self.config.message

        if not payload:
            payload = {}

        payload.update({"type": msg_val})

        payload = {msg_key: orjson.dumps(payload)}

        # Final message dict to send

        self.client.xadd(self.config.stream_key, payload)


class TrainerEvents:
    """
    Main helper class holding all event channels.
    Each event may have its own RedisConfig.

    Usage:
        events = TrainerEvents(cfg.events)
        events.weights_broadcast.send({"step": 100})
        events.training_finished.send()
    """

    def __init__(self, config: TrainerEventsConfig):
        self.config = config

        self.weights_broadcast = OptionalEventSender(config.weights_broadcast)
        self.training_finished = OptionalEventSender(config.training_finished)

        if config.weights_broadcast.enabled and is_main_rank():
            init_method = (
                f"tcp://{config.weights_broadcast.rdvz_master_address}:{config.weights_broadcast.rdvz_master_port}"
            )
            logger.info(f"Waiting for weights broadcast rendezvous at {init_method} ...")
            self.weights_pg = torch.distributed.init_process_group(
                backend="nccl",
                init_method=init_method,
                world_size=config.weights_broadcast.world_size,
                rank=config.weights_broadcast.rank,
            )
            logger.info(f"Weights broadcast rendezvous at {init_method} connected")
        else:
            self.weights_pg = None

    def send_weights(self, step: int, model: FastLLMModel, export_config: TrainingExportConfig):
        if self.config.weights_broadcast.enabled:
            if is_main_rank():
                self.weights_broadcast.send({"step": step})
            # broadcast weights
            for shard_name, layer_name, tensor in model.iter_checkpoint(export_config.get_save_config("", 10), {}):
                if is_main_rank():
                    meta = [(shard_name, layer_name, tensor.shape, tensor.dtype)]
                    torch.distributed.broadcast_object_list(
                        meta, group=self.weights_pg, group_src=self.config.weights_broadcast.rank
                    )
                    torch.distributed.broadcast(
                        tensor, group=self.weights_pg, group_src=self.config.weights_broadcast.rank
                    )
            # Broadcast end of weights broadcast
            if is_main_rank():
                meta = [None]
                torch.distributed.broadcast_object_list(
                    meta, group=self.weights_pg, group_src=self.config.weights_broadcast.rank
                )

    def send_training_finished(self):
        if is_main_rank():
            self.training_finished.send()
            if self.config.weights_broadcast.enabled:
                torch.distributed.destroy_process_group()
