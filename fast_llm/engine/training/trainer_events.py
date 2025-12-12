import logging

import orjson
import redis
import torch.distributed

from fast_llm.engine.config_utils.run import is_main_rank
from fast_llm.engine.multi_stage.fast_llm_model import FastLLMModel
from fast_llm.engine.training.config import TrainerEventsConfig, TrainerEventsRedisConfig, TrainingExportConfig

logger = logging.getLogger(__name__)


class RedisEventSender:
    def __init__(self, config: TrainerEventsRedisConfig):
        self.config = config
        self.client = None

        if is_main_rank():
            self.client = redis.Redis(
                host=config.host,
                port=config.port,
            )

    def send(self, msg_type: str, payload: dict | None = None):
        if not is_main_rank():
            return

        if not payload:
            payload = {}
        payload.update({"type": msg_type})

        self.client.xadd(self.config.stream_key, {self.config.payload_key: orjson.dumps(payload)})


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

        if config.weights_broadcast.enabled or config.training_finished.enabled:
            self.sender = RedisEventSender(config.redis)
        else:
            self.sender = None

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
                self.sender.send(
                    msg_type=self.config.weights_broadcast.weights_ready_message_type, payload={"step": step}
                )
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
            self.sender.send(msg_type=self.config.training_finished.training_finished_message_type)
            if self.config.weights_broadcast.enabled:
                torch.distributed.destroy_process_group()
