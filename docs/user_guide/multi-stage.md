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

[^1]: Communication overhead for ZeRO Stage 2 is similar to Stage 1, except during (depth-first) gradient accumulation when additional reduce-scatter operations occur.
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

Fast-LLM streams sharded tensors through communication buffers, allowing network transfers to overlap with GPU computation. These buffers temporarily store gradient or weight shards during forward and backward passes, improving training throughput by hiding communication latency.

Buffers are only relevant when gradients or parameters are actually sharded, depending on your ZeRO stage:

| Buffer type      | Active when       | Config key           | Default |
| ---------------- | ----------------- | -------------------- | ------- |
| Gradient buffers | ZeRO stage 2 or 3 | `num_grad_buffers`   | `1`     |
| Weight buffers   | ZeRO stage 3 only | `num_weight_buffers` | `1`     |

- **Gradient buffers (`num_grad_buffers`)**:

    - Applies when gradients are sharded (ZeRO stages 2 and 3).
    - Default (`1`) means no overlap (gradients are communicated layer-by-layer).
    - Setting to `2` enables *double-buffering* (second buffer lets gradients transfer asynchronously while the GPU computes the next layer). Values of `3` or more add additional buffers, further increasing overlap at the cost of extra GPU memory per additional buffer.

- **Weight buffers (`num_weight_buffers`)**:

    - Applies only at ZeRO stage 3 when parameters (weights) are sharded.
    - Default (`1`) means no overlap (parameters communicated without asynchronous transfer).
    - Setting to `2` enables *double-buffering* for weights (second buffer lets parameter transfers overlap with GPU computation). Higher values add more overlap, consuming additional GPU memory per buffer.

These buffer settings have no effect when their respective tensors aren't sharded:

- At ZeRO stage **1**, gradients and parameters are fully replicated, so both `num_grad_buffers` and `num_weight_buffers` are ignored.
- At ZeRO stage **2**, parameters remain replicated; thus, only `num_grad_buffers` is relevant.

Buffers do not reduce the total amount of communication, Rather, they shift when communication occurs, improving throughput if your training is network-bound and you have spare GPU memory.

If you want explicit control, you can override these values in your configuration:

```yaml
multi_stage:
  num_grad_buffers: 3        # ZeRO 2 or 3
  num_weight_buffers: 2      # ZeRO 3 only
```

Adjust buffers only if you observe GPU utilization drops due to frequent waiting for network transfers, and have GPU memory to spare. Start with defaults (`1`) and tune upward cautiously.

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
        This feature is currently **not implemented**. Changing this value will currently cause a validation error.
