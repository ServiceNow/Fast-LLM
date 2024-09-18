# Launch and monitor training

## Requirements

At this point, you should already have:

- Access to a cluster with 16 DGX nodes with 8x A100/H100-80GB GPUs ([Or at least 4 GPUs](prepare_training.md)),
connected through an Infiniband (preferred) and/or Ethernet interconnect,
and sharing a common fast storage.
- A [docker image](getting_started.md) for Fast-LLM, available on all nodes.
- A local copy of the [Mistral weights](prepare_mistral.md) on the common storage
- A [preprocessed dataset](prepare_data.md) in json format on the common storage.
- (Optional) A Wandb account and API key.


## Launching the experiment

To launch the experiment, we perform the following on each node,
or use a cluster-specific tool to automate the process:
1. Launch a docker container running our docker image,
ensuring access to all necessary hardware (GPUs, interconnects, etc.),
and mounting the pretrained weights, dataset and an experiment directory.
    ```bash
   docker run --rm -it --gpus all --net=host --ipc=host [-v ...] my_fast_llm_image bash
    ```
2. Note the mounted paths and host address:
    ```bash
    export PRETRAINED_MISTRAL_PATH=...
    export JSON_DATA_PATH=...
    export EXP_BASE_DIR=...
    export HOST_NODE_ADDR=...
    ```
3. Set up the experiment configuration as described in the previous sections:
    ```bash

    export ARCHITECTURE_ARGS_MISTRAL_PRETRAINED="\
   --pretrained_checkpoint_type=huggingface \
   --pretrained_checkpoint_path=$PRETRAINED_MISTRAL_PATH \
   "

   export MODEL_ARGS_MISTRAL_PRETRAINED="\
   $ARCHITECTURE_ARGS_MISTRAL_PRETRAINED \
   --window_size=4096 \
   "

    export DATA_ARGS="\
    --split=9998,2,0 \
    --dataset_source=file \
    --data_path=$JSON_DATA_PATH \
    "

    export TRAINING_ARGS="\
    --batch_size=128 \
    --sequence_length=8192 \
    --train_iters=500000 \
    --weight_decay=0.1 \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --clip_grad=1.0 \
    --lr=0.0001 \
    --lr_warmup_iters=1000 \
    --lr_decay_style=cosine \
    --lr_decay_iters=500000 \
    --min_lr=0.000003 \
    "

    export PERFORMANCE_ARGS="\
    --training_dtype=bf16 \
    --num_workers=8 \
    "

    export MONITORING_ARGS="\
    --experiment_dir=$EXP_BASE_DIR \
    --validation_iters=25 \
    --validation_interval=1000 \
    --max_checkpoints=5 \
    --export_interval=25000 \
    --log_interval=10 \
    --log_offset=0 \
    --checkpoint_interval=500 \
    "
    ```
4. Launch the experiment:
    ```bash
    torchrun --nproc-per-node=8 --nnodes=16 --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR pretrain_fast_llm.py \
    $MODEL_ARGS_MISTRAL_PRETRAINED $DATA_ARGS $TRAINING_ARGS $PERFORMANCE_ARGS $MONITORING_ARGS
    ```

## Monitoring the experiment

After launching the experiment, you may observe the progress through either stdout,
or the log file at `[EXP_BASE_DIR]/runs/0/logs/logs_rank_000.txt`.
If you set up Wandb logging, progress will also be reported there.
