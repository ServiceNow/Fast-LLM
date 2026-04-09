---
title: Parallelism Internals
---

This document describes how Fast-LLM's four parallelism strategies are implemented. It is aimed at contributors adding new layer types, modifying the distributed engine, or debugging performance issues.

For user-facing configuration, see the [Parallelism guide](../user_guide/parallelism.md).

---

## Rank Assignment

All rank arithmetic lives in `fast_llm/engine/distributed/config.py`. Given `world_size`, `tensor_parallel`, `pipeline_parallel`, and `sequence_data_parallel`, the derived dimensions are:

```python
data_parallel           = world_size // (tensor_parallel * pipeline_parallel)
batch_data_parallel     = data_parallel // sequence_data_parallel

tensor_rank             = rank % tensor_parallel
data_rank               = (rank // tensor_parallel) % data_parallel
pipeline_rank           = rank // (tensor_parallel * data_parallel)
batch_data_rank         = data_rank // sequence_data_parallel
sequence_data_rank      = data_rank % sequence_data_parallel
```

When `pipeline_first=True`, `data_rank` and `pipeline_rank` are swapped:

```python
pipeline_rank           = (rank // tensor_parallel) % pipeline_parallel
data_rank               = (rank // tensor_parallel) // pipeline_parallel
```

This alternative ordering places pipeline stages nearer in global rank space, which can improve NUMA locality when each node runs multiple pipeline stages.

---

## Process Groups

`fast_llm/engine/distributed/distributed.py` constructs the NCCL (or Gloo for CPU) process groups from the `DistributedConfig`. Groups are de-duplicated through `ProcessGroupPool` — if two parallelism dimensions happen to cover the same set of ranks, they share a single underlying `torch.distributed.ProcessGroup`.

The named groups used throughout the engine are:

| Name | Members | Primary use |
| --- | --- | --- |
| `world` | All ranks | Global barriers |
| `tensor` | Same `data_rank`, `pipeline_rank` | TP all-reduces |
| `data` | Same `tensor_rank`, `pipeline_rank` | ZeRO reduce-scatter / all-gather |
| `pipeline` | Same `tensor_rank`, `data_rank` | Pipeline send/recv |
| `sequence_data` | Same `tensor_rank`, `pipeline_rank`, `batch_data_rank` | Sequence-parallel reduction |
| `batch_data` | Same `tensor_rank`, `pipeline_rank`, `sequence_data_rank` | Non-sequence data reduction |
| `tensor_and_data` | Same `pipeline_rank` | ZeRO with TP |
| `tensor_and_sequence_data` | Same `pipeline_rank`, `batch_data_rank` | Sequence-TP activations |
| `model_and_sequence_data` | Same `batch_data_rank` | Cross-pipeline sequence gradient |

`Distributed.set_step(step, phase)` reseeds per-step generators (`pp_generator`, `tp_generator`) using large prime offsets per dimension, so dropout and other stochastic ops are deterministically reproducible across ranks and resumptions.

---

## Tensor Parallelism

### Sharded linear layers

Tensor parallelism is implemented by two linear layer variants in `fast_llm/layers/common/linear/linear.py`:

**`OutputParallelLinear`** (column split):

- Weight shape: `[output_dim / tensor_parallel, input_dim]`
- Each rank computes a different slice of the output columns
- Forward: local `Y_local = X @ W_local`; output stays partitioned — no communication on the output
- If `sequence_parallel`: input is first **all-gathered** across the tensor group before the matmul
- Backward: grad_input is **all-reduced** (or **reduce-scattered** with sequence-TP) across the tensor group
- Used for: Q/K/V projections, MLP gate/up projections

**`InputParallelLinear`** (row split):

- Weight shape: `[output_dim, input_dim / tensor_parallel]`
- Each rank holds a row slice of the weight (a slice of the input dimension)
- Forward: local `Y_local = X_local @ W_local`, then **all-reduce** output across the tensor group (so every rank has the full output)
- If `sequence_parallel`: output is **reduce-scattered** instead of all-reduced, so each rank ends up with a sequence slice
- Custom autograd via `input_parallel_linear_autograd` to correctly handle gradient flow
- Used for: attention output projection, MLP down projection

### Sequence-tensor parallelism

The standard (non-sequence-TP) TP pattern replicates the full sequence on every rank between layers. Sequence-tensor parallelism keeps activations distributed across the sequence dimension between layers, reducing activation memory by a factor of `tensor_parallel`.

At each transformer sub-layer (attention or MLP), the flow is:

- **`OutputParallelLinear`**: **all-gather** the sequence-chunked input → full sequence × partial output columns per rank
- Attention / elementwise ops: operate on full-sequence slices
- **`InputParallelLinear`**: matmul → **reduce-scatter** the output → each rank returns to holding `seq_len / tensor_parallel` rows

The total communication volume (all-gather + reduce-scatter) equals that of a single all-reduce, so there is no extra bandwidth cost. The benefit is smaller activation tensors between layers.

### Adding a new tensor-parallel layer

1. Declare weight dimensions using `TensorDim` objects from `fast_llm/engine/config_utils/tensor_dim.py`. Mark the sharded dimension with the appropriate `DistributedDim`.
2. Inherit from `OutputParallelLinear` or `InputParallelLinear`, or implement the custom forward/backward directly in `fast_llm/functional/`.
3. Ensure the layer's `forward()` uses the standard signature: `(input, kwargs, losses, metrics) → Tensor`.

---

## Pipeline Parallelism

### Stage splitting

`MultiStageModel._split_into_stages()` (in `fast_llm/engine/multi_stage/multi_stage.py`) partitions the flat list of `Layer` objects returned by `BaseModel.get_layers()`. Each stage holds `layers_per_stage` consecutive layers. The mapping from stage index to pipeline rank is:

```python
pipeline_rank = (stage_index // stages_per_pipeline_stage) % pipeline_parallel
```

Stages owned by this rank have full weights and compute. Stages on other pipeline ranks are metadata-only stubs (except for tied weights, see below).

### Tied weights

Embedding and LM-head weights are often shared. When these layers land on different pipeline stages, `Stage._tied_weight_copies` holds a reference-only copy:

- Forward pass: tied weights are **broadcast** from the owning stage to all stages that need them.
- Backward pass: gradients from non-owning stages are **all-reduced** back to the owning stage.

### Schedule

The schedule (`fast_llm/engine/schedule/`) builds a DAG of `ScheduleNode` operations (forward, backward, send, recv, optimizer step) and executes them across three CUDA streams (compute, send, recv). Pipeline communication uses PyTorch `isend` / `irecv` for overlap with compute.

`breadth_first_micro_batches` controls how many micro-batches are in-flight at once. With `N` pipeline stages and `breadth_first_micro_batches = N`, the pipeline bubble fraction approaches zero for large batches.

---

## Data Parallelism and ZeRO/FSDP

Data parallelism in Fast-LLM is inseparable from the ZeRO sharding implementation in `fast_llm/engine/multi_stage/fsdp.py`. The `FSDP` class owns the per-stage weight and gradient buffers and orchestrates all-gather / reduce-scatter across the data-parallel group.

### Buffer layout

Each `FSDP` instance maintains flat buffers for a pipeline stage's parameters:

```text
_weight_shard   : [total_params / data_parallel]    # local shard, always resident
_weight_buffer  : [total_params]                    # full weights, reconstructed on demand (ZeRO-3)
_grad_shard     : [total_params / data_parallel]    # local gradient shard
_grad_buffer    : [total_params]                    # full gradient accumulation buffer
```

Every parameter is a view into the appropriate buffer slice, so there are no copies during forward/backward — the buffer *is* the parameter storage.

Shards are padded to a multiple of `SHARD_PAD_TO_MULTIPLE` (32) per rank to ensure aligned communication.

### Forward pass (`restore_parameters`)

Before each forward pass through a stage:

1. Copy `_weight_shard` into the local slice of `_weight_buffer`
2. If ZeRO stage 3: `all_gather(_weight_buffer)` across the data-parallel group

With double-buffering (`num_weight_buffers=2`), the all-gather for stage `i+1` is issued asynchronously while stage `i` is computing.

### Backward pass (`reduce_gradients`)

After each backward pass:

1. If sequence-parallel: `all_reduce` sequence-parallel gradient contributions across the tensor-and-sequence-data group
2. `reduce_scatter(_grad_buffer → _grad_shard)` across the data-parallel group (average reduction)
3. If the gradient shard dtype differs from the buffer dtype (e.g. fp32 grad shard, bf16 buffer), copy and cast

With double-buffering (`num_grad_buffers=2`), the reduce-scatter for stage `i` is overlapped with the backward pass for stage `i-1`.

### ZeRO stage effect on buffers

| ZeRO stage | `_weight_buffer` | `_grad_buffer` | Communication |
| --- | --- | --- | --- |
| 1 | Not used (weights replicated) | Not used (grads replicated, then all-reduce) | All-reduce on grads |
| 2 | Not used | Used (grad reduce-scatter → shard) | Reduce-scatter on grads |
| 3 | Used (all-gather before forward) | Used | All-gather on weights + reduce-scatter on grads |

---

## Sequence Data Parallelism

Sequence data parallelism sub-divides the data-parallel group by the sequence dimension. The `sequence_data` process group covers ranks with the same `tensor_rank`, `pipeline_rank`, and `batch_data_rank`.

During the forward pass of sequence-parallel layers, each rank holds a contiguous chunk of the sequence. When a layer requires the full sequence (e.g. attention), an all-gather is performed across the `sequence_data` group. After the layer, a reduce-scatter returns each rank to its sequence chunk.

Gradient synchronization is handled in `FSDP.reduce_gradients`: gradients from the sequence-parallel group are all-reduced before the data-parallel reduce-scatter, so the sequence dimension is handled before any ZeRO sharding.

---

## Seeding and Reproducibility

`Distributed.set_step(step, phase)` is called at the start of each forward/backward pass. It reseeds two per-rank generators:

- `pp_generator`: seeded by `(step, phase, tensor_rank, data_rank)` — ensures dropout is identical across pipeline ranks within the same TP group.
- `tp_generator`: seeded by `(step, phase, pipeline_rank, data_rank)` — ensures TP ranks sample the same dropout mask.

Large prime offsets per dimension ensure seeds are distinct across all rank combinations. This guarantees deterministic training regardless of which GPU processes which micro-batch, and allows exact resumption from a checkpoint.

---

## Key Source Files

| File | Purpose |
| --- | --- |
| `fast_llm/engine/distributed/config.py` | `DistributedConfig`: rank arithmetic, derived fields |
| `fast_llm/engine/distributed/distributed.py` | `Distributed`: process group construction, `ProcessGroupPool`, seeding |
| `fast_llm/engine/multi_stage/fsdp.py` | `FSDP`: buffer management, all-gather, reduce-scatter, checkpoint loading |
| `fast_llm/engine/multi_stage/multi_stage.py` | `MultiStageModel`: stage splitting, tied weights |
| `fast_llm/engine/multi_stage/config.py` | `MultiStageConfig`: ZeRO stage, buffer counts |
| `fast_llm/layers/common/linear/linear.py` | `OutputParallelLinear`, `InputParallelLinear` |
| `fast_llm/functional/linear.py` | Functional forward/backward for TP linear ops |
| `fast_llm/engine/schedule/config.py` | `ScheduleConfig`: micro-batch and pipeline scheduling |
| `fast_llm/engine/schedule/runner.py` | `ScheduleRunner`: DAG execution, CUDA stream management |
| `tests/utils/distributed_configs.py` | 20+ reference configurations combining all strategies |
