---
title: Parallelism
---

Fast-LLM supports four complementary parallelism strategies that can be combined to train models at any scale. This guide explains each strategy, how to configure it, and when to use it.

For background on memory sharding (ZeRO), see the [Multi-Stage guide](multi-stage.md). The strategies below focus on how the computation itself is distributed.

---

## Overview

| Strategy | Config key | What it splits | Primary benefit |
| --- | --- | --- | --- |
| Data parallelism | `distributed.batch_data_parallel` (derived) | Batch across GPUs | Throughput |
| Tensor parallelism | `distributed.tensor_parallel` | Layers horizontally (weight matrices) | Model memory |
| Pipeline parallelism | `distributed.pipeline_parallel` | Layers vertically (by depth) | Model memory |
| Sequence data parallelism | `distributed.sequence_data_parallel` | Sequence dimension across GPUs | Activation memory |

These strategies compose: a 64-GPU run might use 4-way tensor parallelism, 4-way pipeline parallelism, and 4-way data parallelism simultaneously.

---

## Data Parallelism

Data parallelism replicates the full model on every GPU and processes different micro-batches in parallel. Gradients are averaged across all replicas before the optimizer step.

Fast-LLM infers the data-parallel degree automatically:

```text
data_parallel = world_size / (tensor_parallel × pipeline_parallel)
```

You do not set `data_parallel` directly — it falls out from the other settings.

Data parallelism is the baseline scaling strategy: it increases training throughput proportionally to the number of replicas (assuming sufficient batch size) and adds no memory pressure for the model itself. Any GPUs not used by tensor or pipeline parallelism are automatically used for data parallelism. Its only constraint is that the global batch size grows with the number of replicas.

---

## Tensor Parallelism

Tensor parallelism (sometimes called *intra-layer model parallelism*) shards individual weight matrices across GPUs within a group. Each GPU holds a slice of the weight and computes its portion of the output; an all-reduce synchronizes results.

```yaml
model:
  distributed:
    tensor_parallel: 4   # shard weights across 4 GPUs
```

Valid values are 1 (disabled) or any integer that divides `world_size`. In practice, powers of two work best (1, 2, 4, 8).

**When to use:** When a single model layer (e.g. attention projection or MLP) does not fit on one GPU, or when activation memory from large hidden dimensions is the bottleneck. Tensor parallelism requires high-bandwidth interconnects (NVLink within a node) because it adds an all-reduce communication on every forward *and* backward pass of every sharded layer.

**Rule of thumb:** Keep tensor parallelism within a node (≤ 8 GPUs). Crossing node boundaries with tensor parallelism incurs heavy inter-node communication overhead.

### Sequence-Tensor Parallelism

When tensor parallelism is active, you can enable an additional optimization that keeps activations distributed along the sequence dimension between layers, rather than replicating the full sequence on every GPU:

```yaml
model:
  distributed:
    tensor_parallel: 4
    sequence_tensor_parallel: true
```

With this enabled, each GPU holds only `1 / tensor_parallel` of the sequence at any given time. Activations are gathered before layers that need the full sequence, and scatter-reduced afterward. This reduces peak activation memory per GPU by a factor of `tensor_parallel`, at the same total communication cost as without the option. It is recommended whenever `tensor_parallel > 1`.

---

## Pipeline Parallelism

Pipeline parallelism splits the model by depth: each GPU holds a consecutive block of layers. Activations flow forward from stage to stage; gradients flow backward. Multiple micro-batches can be in-flight simultaneously to keep all stages busy.

```yaml
model:
  distributed:
    pipeline_parallel: 4   # split model across 4 GPUs
```

The number of layers per pipeline stage is controlled by how the total layer count divides across stages (see the [Multi-Stage guide](multi-stage.md) for `layers_per_stage`).

Pipeline parallelism works well across slow interconnects (e.g. InfiniBand between nodes) because point-to-point sends only occur at stage boundaries, and their volume is proportional to the activation size of a single layer rather than the full model.

### Scheduling micro-batches

To hide pipeline bubbles, Fast-LLM uses *breadth-first* scheduling: it keeps several micro-batches in flight simultaneously so each stage always has work to do.

```yaml
schedule:
  micro_batch_splits: 1          # sub-divide each micro-batch along the sequence
  breadth_first_micro_batches: 4 # interleave this many micro-batches across stages
  depth_first_micro_batches: 1   # gradient accumulation steps within one stage
```

A larger `breadth_first_micro_batches` reduces idle (bubble) time but increases activation memory, since activations from all in-flight micro-batches are held simultaneously. Start with a value equal to the number of pipeline stages.

!!! note
    The total number of micro-batches per step (`breadth_first_micro_batches × depth_first_micro_batches`) must be at least equal to `pipeline_parallel`. Otherwise some pipeline stages will be idle for the entire step.

**When to use:** When the model is too large to fit on a single node, or when you want to spread memory across nodes without incurring the per-layer all-reduce cost of tensor parallelism. Pipeline parallelism is naturally suited to slow cross-node links.

---

## Sequence Data Parallelism

Sequence data parallelism sub-divides the data-parallel group along the sequence dimension. Instead of each GPU processing an independent sequence in full, a group of GPUs collectively processes one sequence by splitting it into chunks.

```yaml
model:
  distributed:
    sequence_data_parallel: 2   # 2 GPUs share each sequence
```

`sequence_data_parallel` must divide `data_parallel`. The effective batch dimension is:

```text
batch_data_parallel = data_parallel / sequence_data_parallel
```

**When to use:** When training on very long sequences and activation memory is the primary constraint. Sequence data parallelism reduces per-GPU activation memory roughly in proportion to its value, at the cost of added gradient synchronization along the sequence dimension.

---

## Combining Strategies

All four strategies compose freely. A typical large-scale configuration looks like:

```yaml
model:
  distributed:
    tensor_parallel: 4              # within-node weight sharding
    sequence_tensor_parallel: true  # sequence-split activations
    pipeline_parallel: 8            # cross-node layer sharding
    sequence_data_parallel: 1       # each sequence lives on one GPU
    # data_parallel is inferred: world_size / (4 × 8) = e.g. 4 for a 128-GPU run

schedule:
  breadth_first_micro_batches: 8   # match pipeline depth
```

### Choosing a configuration

Start with the simplest setup that fits the model in memory and scale from there:

1. **Single GPU**: no parallelism needed.
2. **Multi-GPU, single node**: add `tensor_parallel` up to the number of GPUs (typically 8). Enable `sequence_tensor_parallel` alongside it.
3. **Multi-node**: add `pipeline_parallel` across nodes. Keep `tensor_parallel` within nodes.
4. **Very long sequences**: add `sequence_data_parallel` to reduce activation memory.
5. **Still out of memory**: increase `zero_stage` (see [Multi-Stage guide](multi-stage.md)).

### Rank ordering

By default, Fast-LLM assigns global ranks in tensor → data → pipeline order. If pipeline stages are on different sockets of the same machine, setting `pipeline_first: true` can improve NUMA locality:

```yaml
model:
  distributed:
    pipeline_first: true
```

---

## Configuration Reference

All distributed settings live under `model.distributed`:

| Field | Default | Description |
| --- | --- | --- |
| `tensor_parallel` | `1` | Size of the tensor-parallel group |
| `pipeline_parallel` | `1` | Number of pipeline stages |
| `sequence_data_parallel` | `1` | Sub-divide data-parallel group by sequence |
| `sequence_tensor_parallel` | `false` | Enable sequence-parallel activation splitting in TP layers |
| `pipeline_first` | `false` | Swap data and pipeline rank ordering for NUMA locality |

Schedule settings live under `schedule`:

| Field | Default | Description |
| --- | --- | --- |
| `breadth_first_micro_batches` | `1` | Micro-batches in flight simultaneously (reduces pipeline bubble) |
| `depth_first_micro_batches` | `1` | Gradient accumulation steps within a stage |
| `micro_batch_splits` | `1` | Sub-divide each micro-batch along the sequence dimension |
