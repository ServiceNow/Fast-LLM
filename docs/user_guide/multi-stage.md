# Multi-Stage Training in Fast-LLM

Fast-LLM trains large models by splitting them into *stages*, each running on a separate GPU or node. It reduces memory usage by distributing (or *sharding*) model state (weights, gradients, or optimizer states) across devices.

A *stage* refers to a logical partition of a model, typically containing a subset of layers or computational steps. Each stage runs independently on its own GPU or node.

This guide explains how to configure multi-stage training for both common and advanced use cases.

## ZeRO-Stage Sharding

Fast-LLM uses ZeRO-style sharding to reduce memory usage by partitioning model state (such as weights, gradients, and optimizer states) across GPUs that would otherwise maintain full replicas in data parallelism. This is compatible with and complementary to model-parallel techniques like pipeline and tensor parallelism.

The primary setting for ZeRO sharding is `zero_stage` in your configuration:

```yaml
multi_stage:
  zero_stage: ...
```

The following table summarizes the behavior of `zero_stage`:

| `zero_stage`  | Weights    | Gradients  | Optimizer States | Communication overhead    |
| ------------- | ---------- | ---------- | ---------------- | ------------------------- |
| `1` (default) | Replicated | Replicated | Sharded          | Moderate, default choice  |
| `2`           | Replicated | Sharded    | Sharded          | Moderate[^1]              |
| `3`           | Sharded    | Sharded    | Sharded          | High[^2]                  |

[^1]: Communication overhead for ZeRO Stage 2 is similar to Stage 1, except during (depth-first) gradient accumulation when additional all-reduce operations occur.
[^2]: Communication overhead for ZeRO Stage 3 is higher than Stage 2, especially during (depth-first) gradient accumulation.

Optimizer states are always sharded by default. ZeRO Stage 0 (full replication) is not supported.

While ZeRO Stage 3 introduces the most communication overhead, the practical difference between Stages 1 and 2 is minimal except during gradient accumulation.

**Recommendation:**

- **ZeRO Stage 1 (default)**: Ideal for most training scenarios.
- **ZeRO Stage 2**: Useful if gradients cause memory pressure.
- **ZeRO Stage 3**: Useful for very large models exceeding GPU memory.

In general, start with the default (`zero_stage: 1`) and verify if your model trains without memory errors. If you encounter out-of-memory issues, try increasing `zero_stage`:

```yaml
multi_stage:
  zero_stage: 2
```

Increasing ZeRO-style sharding reduces memory consumption but may add communication overhead between GPUs or nodes, potentially slowing down training. Before increasing `zero_stage`, first try lowering the micro batch size or sequence length, as this typically incurs less overhead.

You'll likely iterate between adjusting `zero_stage`, micro batch size, and sequence length to find the optimal balance of memory usage and training throughput. If these adjustments don't resolve your issue, or you're unsatisfied with tradeoffs like sequence length versus throughput, reconsider your broader parallelism strategy. This includes adjusting tensor parallelism, pipeline parallelism, or sequence data parallelism, covered in greater depth in the [Parallelism Guide](parallelism.md).

## Expert Options

Beyond `zero_stage`, Fast-LLM offers additional multi-stage settings for fine-tuning. These advanced options typically don't need manual adjustment. Change them only if you're certain about your goals and tradeoffs.

### Buffers

When gradients or weights are sharded, Fast-LLM accumulates partial results in shared *buffers* during forward and backward passes, separately for gradients and weights. These buffers reduce communication overhead by batching gradient or weight updates across GPUs or nodes. The options `num_grad_buffers` and `num_weight_buffers` control the number of buffers used for gradients and weights, respectively.

By default, Fast-LLM assigns one gradient and weight buffer per stage, where the number of stages equals the total number of logical partitions (stages) of the model. This enables overlapping communication (e.g., data transfers between GPUs or nodes) with computation (actual processing done by each GPU or node). Lower values (e.g., 1) reduce this overlap, potentially increasing communication waiting times.

Increasing `num_grad_buffers` or `num_weight_buffers` provides more room for overlapping communication with compute. This can help in some setups, especially when stages are imbalanced, but generally isn't necessary. Note that this does not reduce total communication; it just shifts when it happens.

If you want explicit control, you can override these values:

```yaml
multi_stage:
  num_grad_buffers: 3
  num_weight_buffers: 2
```

Increasing `num_grad_buffers` to `3` or `4` decreases inter-GPU communication frequency, potentially improving throughput—provided sufficient GPU memory is available.

### Stage Layout Control

You can adjust how layers and pipeline stages map onto GPUs or nodes:

```yaml
multi_stage:
  layers_per_stage: 1.0
  stages_per_pipeline_stage: 1
```

Defaults work well in most cases:

- **`layers_per_stage`**: Determines the number of layers per stage. Defaults to `1.0` (one layer per stage). Increase to reduce inter-stage communication or decrease for better load balancing. Fractional values are allowed.

    !!! warning
        This setting is supported but hasn't been tested in recent versions. Use with caution.

- **`stages_per_pipeline_stage`**: Intended to specify how many stages run per pipeline worker when pipeline parallelism is active.

    !!! warning
        This feature is currently **not implemented**. Changing this value has no effect.
