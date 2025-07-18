import argparse
import glob
import os
import random
from datetime import timedelta
from pathlib import Path
from typing import Optional

import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler, InitProcessGroupKwargs, set_seed
from easy_context import (
    apply_seq_parallel_monkey_patch,
    apply_unsloth_offloaded_gradient_checkpoint_monkey_patch,
    prepare_dataloader,
    prepare_seq_parallel_inputs,
)
from flash_attn.losses.cross_entropy import CrossEntropyLoss

# from transformers import AutoModelForCausalLM
from model_llama import LlamaForCausalLM
from packed_dataset import PackedDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed

apply_unsloth_offloaded_gradient_checkpoint_monkey_patch()

# train_data_config = [
#     ("train_slim", 0.693584),
#     ("train_star", 0.306416),
# ]
train_data_config = [
    # ("zyda_2_sample", 1.0),
    # ("gsm8k_sample", 1.0),
    ("fineweb_sample", 1.0),
]
val_data_config = None


def transition(x_0, sigma, maskable_mask, mask_token_id):
    # move_chance = 1 - (-sigma).exp()
    move_chance = sigma
    move_indices = (torch.rand(*x_0.shape, device=x_0.device) < move_chance) & maskable_mask
    x_t = torch.where(move_indices, mask_token_id, x_0)
    return x_t


def create_dataloader(
    batch_size: int,
    block_size: int,
    data_dir: Path,
    accelerator,
    shuffle: bool = True,
    seed: int = 4756,
    split="train",
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        random.seed(seed)
        random.shuffle(filenames)
        print(f"[RANK {accelerator.process_index}] found {len(filenames)} files", flush=True)

        dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size.
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=8,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed + accelerator.process_index,
            num_processes=accelerator.num_processes,
            process_rank=accelerator.process_index,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    # weights = [weight for _, weight in data_config]
    # sum_weights = sum(weights)
    # weights = [el / sum_weights for el in weights]

    combined_dataset = datasets[0]  # CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    accelerator,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        accelerator=accelerator,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train",
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            accelerator=accelerator,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation",
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


def main(args):

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.wandb:
        import wandb

        wandb_api_key_path = os.environ.get("WANDB_API_KEY_PATH")
        if wandb_api_key_path and os.path.exists(wandb_api_key_path):
            with open(wandb_api_key_path, "r") as f:
                wandb_api_key = f.read().strip()
            wandb.login(key=wandb_api_key)
        else:
            wandb.login()

    set_seed(args.seed)

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulate_every,
        mixed_precision="bf16",
        log_with="wandb" if args.wandb else None,
        kwargs_handlers=[timeout],
        # fsdp_plugin=fsdp_plugin,
    )

    wandb_config = vars(args)
    accelerator.init_trackers(
        project_name=args.wandb,
        config=wandb_config,
        init_kwargs={
            "wandb": {"name": args.output_dir.split("/")[-1], "entity": args.wandb_entity, "group": "slam_diffusion"}
        },
    )
    accelerator.print(f"Total GPUS: {accelerator.num_processes}")

    if accelerator.state.deepspeed_plugin is not None:
        ds_config = accelerator.state.deepspeed_plugin.deepspeed_config
        if args.wandb:
            wandb_config["deepspeed_config"] = ds_config
            accelerator.log({"deepspeed_config": ds_config}, step=0)

    train_loader, val_dataloader = create_dataloaders(
        batch_size=args.batch_size,
        block_size=args.seq_length,
        accelerator=accelerator,
        train_data_dir=Path(args.dataset),
        val_data_dir=None,
        seed=3407,
    )

    model = LlamaForCausalLM.from_pretrained(
        args.model,
        # device_map=accelerator.device,
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    )

    model_type = "llama" if isinstance(model, transformers.LlamaForCausalLM) else "mistral"
    apply_seq_parallel_monkey_patch(args.parallel_mode, model_type)

    if args.learning_rate != 2e-5:
        accelerator.print(
            f"Warning: You also need to modify accelerate_configs/zero3_offload.json to change the learning rate"
        )
    optim = DummyOptim(model.parameters(), lr=args.learning_rate)
    scheduler = DummyScheduler(
        optim,
        num_training_steps=args.max_train_steps,
        total_num_steps=args.max_train_steps,
    )

    model, optim, scheduler = accelerator.prepare(model, optim, scheduler)
    train_loader = prepare_dataloader(args.parallel_mode, train_loader, accelerator)
    model.gradient_checkpointing_enable()

    accelerator.register_for_checkpointing(scheduler)

    accelerator.print(f"Max train steps: {args.max_train_steps}")
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    model.train()
    loss_func = CrossEntropyLoss(inplace_backward=True, reduction="none")

    sampling_eps = 1e-3
    mask_token_id = args.mask_token  # mask token id. can be a new token or an existing token.

    for step, batch in enumerate(train_loader):

        input_ids = batch[..., : args.seq_length + 1]
        # print(input_ids.shape)
        target_ids = batch[..., : args.seq_length + 1]
        position_ids = torch.arange(args.seq_length + 1).unsqueeze(0).expand(input_ids.shape[0], -1)
        # shard the input_ids according to the world size and rank according to zig zag attention

        prepared = prepare_seq_parallel_inputs(
            args.parallel_mode,
            input_ids,
            position_ids,
            target_ids,
            accelerator.process_index,
            accelerator.num_processes,
            accelerator.device,
        )
        local_input_ids = prepared["local_input_ids"]
        local_position_ids = prepared["local_position_ids"]
        local_target_ids = prepared["local_target_ids"]
        src_mask = torch.zeros_like(local_input_ids, dtype=torch.bool, device=local_input_ids.device)

        # change range to [sampling_eps, 1 - sampling_eps]
        t = (1 - (2 * sampling_eps)) * torch.rand(
            local_input_ids.shape[0], device=local_input_ids.device
        ) + sampling_eps
        sigma = t
        dsigma = torch.reciprocal(t)  # dsigma = 1 / t

        local_input_ids = transition(
            local_input_ids, sigma[:, None], maskable_mask=~src_mask, mask_token_id=mask_token_id
        )
        loss_log = None
        loss_mask = local_input_ids == mask_token_id
        with accelerator.accumulate(model):
            logits = model(
                # Drop the last token from forward pass
                local_input_ids[:, :-1],
                position_ids=local_position_ids[:, :-1],
            ).logits

            # logits = logits[:,:-1] # do not drop the last token here since already dropped in the forward pass
            loss_mask = loss_mask[:, 1:]
            local_target_ids = local_target_ids[:, 1:]
            loss = loss_func(logits.reshape(-1, logits.shape[-1]), local_target_ids.reshape(-1)).reshape(
                local_target_ids.shape[0], -1
            )
            loss = loss.masked_fill(~loss_mask, 0)
            loss = (dsigma[:, None] * loss).sum() / torch.clamp(
                loss_mask.sum(), min=1
            )  # avg token loss if 0 set it to 1 to avoid NaN
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                # pay attention here. When any seq parallel algo is turned on. This technically only log the very first chunk's loss
                # and what is the first chunk really depends on how do you shard the sequence
                # for zig zag attention, the first chunk contains the left most and rightmost tokens
                # so you cannot compare the (logged) loss of dist attention and zigzag ring attention.
                # loss_log = {"loss": loss.item(), "ppl": math.exp(loss.item())}

                # we now try gathered loss to verify if ring attention and dist flash attention produce the same loss
                # this may slow down the training
                gathered_loss = accelerator.reduce(loss.clone().detach(), "mean")
                loss_log = {
                    "loss": gathered_loss.item(),
                    # "ppl": math.exp(gathered_loss.item()), # same as loss
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                accelerator.log(loss_log, step=completed_steps)

            optim.step()
            scheduler.step()
            optim.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            if loss_log is not None:
                progress_bar.set_postfix(loss_log)
            completed_steps += 1

            if (completed_steps % args.checkpoint_interval) == 0:
                accelerator.print(f"[Saving checkpoint at step {completed_steps}", flush=True)
                accelerator.wait_for_everyone()
                # All processes need to do this to ensure the model is saved correctly
                state_dict = accelerator.get_state_dict(model)

                try:
                    # Only main process saves the model
                    accelerator.unwrap_model(model).save_pretrained(
                        f"{args.output_dir}/checkpoint-{completed_steps}",
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                        state_dict=state_dict,
                    )
                except Exception as e:
                    accelerator.print(f"Error saving checkpoint: {e}")
                    accelerator.print(f"Model class at save time: {model.__class__}")

                # Wait for the main to finish to sync all processes
                accelerator.wait_for_everyone()

        if completed_steps >= args.max_train_steps:
            break

    accelerator.print(f"Training Finished")

    if args.output_dir is not None:
        accelerator.print(f"Saving model to {args.output_dir}")

        accelerator.wait_for_everyone()

        state_dict = accelerator.get_state_dict(model)

        accelerator.unwrap_model(model).save_pretrained(
            f"{args.output_dir}/final",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=state_dict,
        )

        accelerator.print(f"Saving Finished")

    accelerator.end_training()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch-size", type=int, default=1)
    args.add_argument("--gradient-accumulate-every", type=int, default=8)
    args.add_argument("--output-dir", type=str, required=True)
    args.add_argument("--wandb", type=str)
    args.add_argument("--wandb-entity", type=str, default="akshaykalkunte")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--max-train-steps", type=int, default=400)
    args.add_argument("--learning-rate", type=float, default=2e-5)
    args.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    args.add_argument(
        "--dataset",
        type=str,
        default="/work/nvme/bbzy/shivama2/TinyLlama/data/slim_star_combined/",
    )  # Path to processed dataset from TinyLlama pre-processing.
    args.add_argument("--seq-length", type=int, default=16384)
    args.add_argument("--mask_token", type=int, default=811)
    args.add_argument(
        "--parallel_mode",
        type=str,
        choices=["dist_flash_attn", "ulysses_attn", "data_parallel"],
    )
    args.add_argument("--checkpoint-interval", type=int, default=500, help="Number of steps between checkpoints")
    main(args.parse_args())
