# Prepare the training configuration

# Training parameters

Our example training scheme is as follows:
1. We train over 500 K iteration, each made of 128 samples of 8192 tokens, for a total of 524 B training tokens.
2. We use the Adam optimizer with weight decay (Adamw), and gradient clipping.
3. We warm up the learning rate for the first 1000 steps, then use cosine decay from 1e-4 to 3e-6.

This translates into the following Fast-LLM configuration:
```bash
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
```

# Performance parameters

Our training setup is simple enough that the default distributed configuration
(data parallel with [ZeRO stage 1](https://www.deepspeed.ai/tutorials/zero/))
is sufficient for a near-optimal training throughput of around 9000 tokens/s/GPU on H100 GPUs (440 tflops/GPU).
We only need to specify the training dtype and the number of data loader workers.
```bash
export PERFORMANCE_ARGS="\
--training_dtype=bf16 \
--num_workers=8 \
"
```

Note that this configuration requires exactly 16 nodes.
It may be adjusted run on fewer than 16 nodes,
by using gradient accumulation to keep the micro-batch size constant and adding some memory optimizations.
We suggest the following configuration for 4 to 64 GPUs (seet details the in next section):
```bash
export PERFORMANCE_ARGS_SMALL_CLUSTER="\
$PERFORMANCE_ARGS \
--micro_batch_size=1 \
--zero_stage=2 \
"
```

# (Optional) More on Mistral performance optimization

The performance optimization of Mistral at the configuration level
is mainly determined through the following guidelines:

- **Use larger micro-batches**: The GPU runs more efficiently with larger kernels,
so we want the micro-batches to be as large as allowed by memory and other constraints.
Our configuration requires 36 GiB of activation memory,
so a micro-batch or 8192 tokens per GPU is a reasonable choice.
A value of 16384 tokens per GPU is technically feasible,
but would require aggressive state memory optimizations and a higher batch size.
2 - **Reduce model parallelism**: Model parallelism (tensor or pipeline) comes with a large overhead,
so we should avoid or limit it whenever possible.
For Mistral, no model parallelism is needed.
3 - **Optimize the memory usage**: Additional memory optimizations are available to enable configurations that would
otherwise not be possible. We already saw the most important one, the ZeRO stage (`--zero_stage` see note below).
An additional one is the recomputation of the MLP activations `--mlp_recompute_level` ,
which significantly lower the activation memory usage, for a small (`activation`) or moderate (`full`) overhead.
Note that Fast-LLM does not implement activation recomputation for the entire transformer layer,
as it comes with a large overhead (~33%) and it can be avoided in (almost) all practical scenario.


??? note "More on ZeRO stages"

    Fast-LLM provides a custom implementation of the training state partitioning
    first described in the [ZeRO (Zero Redundancy Optimizer) paper](https://arxiv.org/abs/1910.02054).
    The method comes in three "stages", which progressively reduce the memory footprint from the training state:

    - **Stage 1**: Partition the optimizer state and its update across the data-parallel GPUs.
      This stage reduces the state memory by around 3x (for mixed precision training with full-precision gradients),
      while simultanuously speeding up training through a faster weight update.

    - **Stage 2**: Extend partitioning to the (reduced) gradients.
      This stage reduces the state memory by a further 3x,
      but may come with a minor overhead (depending on the implementation),
      and may require multiple reductions with gradient accumulation.

    - **Stage 3**: Extend partitioning to the weights.
      This stage drops the vast majority of the remaining state memory,
      but requires extra network communication.

    Fast-LLM implements all three of these stages, selected through the `--zero_stage` argument.
    There is no option to disable ZeRO entirely, as it would be strictly worse in terms of performance.
    In general, training configurations should use the lowest value allowed by other memory constraints.

??? note "Recompute Level for MLPs"

    The MLP is the largest contributor to a transformer's activation memory (with Flash Attention),
    so recomputing its activations is a natural way to save memory.
    Fast-LLM offers three MLP recomputaton modes, set throught the `--mlp_recompute_level` argument:

    - **`none`** (default): All MLP activations are kept,
    allowing for the highest throughput at the highest memory cost.

    - **`activation`**: The MLP activation layer output (gelu, silu, etc.) is dropped and recomputed in the backward pass.
    This saves on activation memory (~20% for Mistral) with minimal impact on throughput.

    - **`full`**: Both the first dense layer and activation layer outputs are dropped and recomputed.
    This saves more activation memory (~60% for Mistral), but has a noticeable impact on throughput .

    For quantitative comparison, here are benchmarks for Mistral (using 4x A100 GPUs):

    | Recompute Level | Act. Memory (MiB) | Tokens/s/GPU | Model TFLOP/s/GPU |
    |-----------------|-------------------|--------------|---------------|
    | `none`          | 36515             | 4234.09      | 202.88        |
    | `activation`    | 29346             | 4218.63      | 202.14        |
    | `full`          | 15010             | 3804.49      | 182.29        |


# Monitoring and persistence parameters

Finally, we set up experiment monitoring and persistence
```bash
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
This setup includes:
- Creation of an experiment directory at `$EXP_BASE_DIR` to store checkpoints, logs, data cache and other artifacts.
- Validation for 25 steps every 1000 steps
- Logging of losses, metrics and other relevant quantities every 10 steps (from rank 0),
  both to stdout and the log file.
- Saving of a temporary checkpoint every 500 steps, and of a permanent checkpoint every 25000 steps.


??? note "More on Fast-LLM checkpointing"

    Fast-LLM provides two types of checkpoints:

    - `checkpoint`: temporary checkpoint saved at `[--experiment_dir]/checkpoints/[iter]`,
      to reload the experiment in case of a planned or unexpected shutdown.
      Only the `--max_checkpoints` most recent ones are kept to limit disk usage.
      Note that saving a checkpoint with Fast-LLM is relatively fast so can (and should) be done frequently.
    - `export`: permanent checkpoint saved at `[--experiment_dir]/export/[iter]`.
      This checkpoint type is typically intended for long-term storage, benchmarking, inference, etc.
      It should be saved less often to limit disk usage.


# (Optional) Set up wandb

Fast-LLM also support monitoring through [Weights and Biases](https://wandb.ai/).
This requires a valid API key,
passed through an environment variable rather than an explicit argument for security reasons.
It can be either contained in `$WANDB_API_KEY` or in a plain text file found at `$WANDB_API_KEY_PATH`.
Then, we set the Wandb username, project and version (Wandb group).
```bash
export WANDB_ARGS="\
--wandb_entity_name=$WANDB_ENTITY_NAME \
--wandb_project_name=$PROJECT_NAME \
--wandb_group_name=$PROJECT_VERSION \
"
```
The Wandb run will be set as the directory name of `$EXP_BASE_DIR`, or can be overriden through `--experiment_name`.
