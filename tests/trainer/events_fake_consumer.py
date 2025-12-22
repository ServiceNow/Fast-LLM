import json
import sys
from pathlib import Path

import redis
import safetensors.torch
import torch.distributed
import yaml


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m tests.trainer.events_fake_consumer <config_path>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"Config file {config_path} does not exist")
        sys.exit(1)

    with config_path.open("rt") as f:
        config = yaml.safe_load(f)

    consumer_cfg = config["consumer"]
    world_size = consumer_cfg["world_size"]
    rank = consumer_cfg["rank"]
    results_path = Path(consumer_cfg["results_path"])
    results_path.mkdir(parents=True, exist_ok=True)

    consumer_id = f"[Consumer {rank}/{world_size}]"

    print(f"{consumer_id} Started with config:")
    print(yaml.safe_dump(config))

    assert config["events"]["weights_broadcast"]["enabled"]
    assert config["events"]["training_finished"]["enabled"]

    redis_client = redis.Redis(host=config["events"]["redis"]["host"], port=config["events"]["redis"]["port"])

    print(f"{consumer_id} waiting for pg rendezvous...")
    weights_pg = torch.distributed.init_process_group(
        backend="nccl",
        init_method=f'tcp://{config["events"]["weights_broadcast"]["rdvz_master_address"]}:'
        f'{config["events"]["weights_broadcast"]["rdvz_master_port"]}',
        world_size=world_size,
        rank=rank,
    )
    broadcast_source_rank = config["events"]["weights_broadcast"]["rank"]

    last_id = "0-0"
    msg_key = config["events"]["redis"]["payload_key"].encode()
    stream_key = config["events"]["redis"]["stream_key"]

    print(f"{consumer_id} waiting for messages...")
    while True:
        result = redis_client.xread(
            streams={stream_key: last_id},
            count=1,
            block=200,
        )

        if not result:
            continue

        _, events = result[0]

        for event_id, msg in events:
            last_id = event_id
            assert msg_key in msg
            msg = json.loads(msg[msg_key].decode())
            print(f"{consumer_id} msg received: {msg}")
            if msg["type"] == config["events"]["weights_broadcast"]["weights_ready_message_type"] or (
                msg["type"] == config["events"]["weights_broadcast"]["initial_weights_step_message_type"]
                and config["events"]["weights_broadcast"]["initial_weights_step_message_includes_weights"]
            ):
                weights = {}
                while True:
                    meta = [None]
                    torch.distributed.broadcast_object_list(meta, group=weights_pg, group_src=broadcast_source_rank)
                    meta = meta[0]
                    if meta is None:
                        print(f"{consumer_id} weight broadcast finished")
                        break
                    shard_name, layer_name, tensor_size, tensor_type = meta
                    tensor = torch.zeros(
                        tuple(tensor_size), dtype=tensor_type, device="cuda"
                    )  # so far consumer is single gpu only
                    torch.distributed.broadcast(tensor, group=weights_pg, group_src=broadcast_source_rank)
                    print(f"{consumer_id} {shard_name} layer {layer_name} {tensor_size} {tensor_type} received")
                    if shard_name == "weights":
                        weights[layer_name] = tensor
                safetensors.torch.save_file(weights, results_path / f"{msg["step"]}.safetensors")

            elif msg["type"] == config["events"]["training_finished"]["training_finished_message_type"]:
                torch.distributed.destroy_process_group()
                (results_path / "training_finished").touch()
                return
            else:
                raise RuntimeError(f"{consumer_id} Received unknown message type {msg}")
            if msg["type"] == config["events"]["weights_broadcast"]["initial_weights_step_message_type"]:
                (results_path / "initial_weights_step").touch()


if __name__ == "__main__":
    main()
