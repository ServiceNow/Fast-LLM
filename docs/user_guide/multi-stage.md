# Multi-Stage Training in Fast-LLM

Fast-LLM trains large models by splitting them into *stages*, each running on a separate GPU or node. It reduces memory usage by distributing (or *sharding*) model state (weights, gradients, or optimizer states) across devices.

This guide explains how to configure multi-stage training for both common and advanced use cases.

## ZeRO-Stage Sharding

Fast-LLM uses ZeRO-style sharding to partition model state efficiently across GPUs. This differs from pipeline parallelism, which partitions model computation into sequential pipeline stages.

The primary setting for ZeRO sharding is `zero_stage` in your configuration:

```yaml
multi_stage:
  zero_stage: ...
```

The following table summarizes the behavior of `zero_stage`:

| `zero_stage`  | Weights    | Gradients  | Optimizer States | Communication overhead                                       |
| ------------- | ---------- | ---------- | ---------------- | ------------------------------------------------------------ |
| `1` (default) | Replicated | Replicated | Sharded          | Lowest, default choice                                       |
| `2`           | Replicated | Sharded    | Sharded          | Moderate, saves more memory at additional communication cost |
| `3`           | Sharded    | Sharded    | Sharded          | Highest, maximum memory saving with increased communication  |

Optimizer states are always sharded by default. ZeRO Stage 0 (full replication) is not supported.

In general, start with the default (`zero_stage: 1`) and verify if your model trains without memory errors. If you encounter out-of-memory issues, try increasing `zero_stage`:

```yaml
multi_stage:
  zero_stage: 2
```

Increased sharding reduces memory consumption but adds communication overhead between GPUs or nodes. Before increasing `zero_stage`, you might first try lowering the micro batch size or sequence length, since this usually incurs less overhead.

You'll likely iterate between adjusting `zero_stage`, micro batch size, and sequence length to find the optimal balance of memory usage and training throughput. If these adjustments don't resolve your issue, or you're unsatisfied with tradeoffs like sequence length versus throughput, you may need to reconsider your broader parallelism strategy. This includes adjusting tensor parallelism, pipeline parallelism, or sequence data parallelism. That topic is covered in greater depth in the [Parallelism Guide](parallelism.md).

## Expert Options

Beyond `zero_stage`, Fast-LLM offers additional multi-stage settings for fine-tuning. These advanced options typically don't need manual adjustment. Change them only if you're certain about your goals and tradeoffs.

### Buffers

When gradients or weights are sharded, Fast-LLM accumulates partial results in shared *buffers* during forward and backward passes. These buffers reduce communication overhead by batching gradient or weight updates across GPUs or nodes.

By default, Fast-LLM automatically determines buffer counts based on your `zero_stage` setting:

- `num_grad_buffers`:
    - `2` if `zero_stage >= 2`
    - `1` otherwise
- `num_weight_buffers`:
    - `2` if `zero_stage == 3`
    - `1` otherwise

If you want explicit control, you can override these values:

```yaml
multi_stage:
  num_grad_buffers: 3
  num_weight_buffers: 2
```

For example, increasing `num_grad_buffers` to `3` or `4` will decrease inter-GPU communication frequency, potentially improving throughput—provided sufficient GPU memory is available.

### Stage Layout Control

You can adjust how layers and pipeline stages map onto GPUs or nodes:

```yaml
multi_stage:
  layers_per_stage: 1.0
  stages_per_pipeline_stage: 1
```

Defaults work well in most cases:

- **`layers_per_stage`**: Determines the number of layers per stage. Defaults to `1.0` (one layer per stage). Increase it to reduce inter-stage communication or decrease it for better load balancing. Fractional values are allowed.

- **`stages_per_pipeline_stage`**: Specifies how many stages run per pipeline worker. This setting is relevant only when pipeline parallelism is active. Default is `1`. Increase to assign multiple stages to the same pipeline worker, potentially simplifying communication patterns at the cost of flexibility in load distribution.
