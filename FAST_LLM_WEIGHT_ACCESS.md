# How to Access Weights in Fast-LLM

## The Problem
When you load a Fast-LLM model and try to access weights through normal parameter attributes like `model.layer.weight` or `model.layer.bias`, they appear to be all zeros. This is misleading!

## The Root Cause: FSDP Weight Management

Fast-LLM uses a sophisticated FSDP (Fully Sharded Data Parallel) system for memory efficiency:

1. **Weight Shard**: The actual weights are stored in a flat 1D tensor called `_weight_shard`
2. **Weight Buffer**: Parameters are views into `_weight_buffer` (starts as zeros)
3. **Restore Step**: `restore_parameters()` copies from shard to buffer before forward pass

### Architecture

```
_weight_shard (1D tensor with actual data)
      ↓ restore_parameters()
_weight_buffer (flat buffer, initially zeros)
      ↓ views
parameters (.weight, .bias - show zeros until restored!)
```

## The Solution - Method 1: Access the Shard Directly

```python
from fast_llm.engine.multi_stage.config import ShardName

# Load model
model = GPTModel.from_pretrained(CheckpointLoadConfig(...))

# Get the actual weights (NOT through .weight or .bias!)
weights_shard = model.get_shard(ShardName.weights)  # Returns a 1D tensor with ALL weights

# weights_shard is a flat tensor containing all model weights
print(weights_shard.shape)  # e.g., torch.Size([2804643712])
print(weights_shard.sum())   # Non-zero!
```

## The Solution - Method 2: Restore Parameters First

```python
# Load model
model = GPTModel.from_pretrained(CheckpointLoadConfig(...))

# Parameters show zeros BEFORE restore
print(model.base_model.decoder[0].mlp.router.bias.sum())  # 0.0

# Restore parameters from shard to buffer
for stage in model._stages:
    stage.restore_parameters()

# Parameters show actual weights AFTER restore
print(model.base_model.decoder[0].mlp.router.bias.sum())  # Non-zero!
```

## Why Parameters Show Zeros

Fast-LLM's FSDP implementation:
- Creates parameters as **views into `_weight_buffer`** (see `fsdp.py:82-90`)
- The buffer starts empty (zeros) for memory efficiency
- `restore_parameters()` copies from `_weight_shard` to `_weight_buffer` (see `fsdp.py:181-189`)
- This happens automatically during forward pass (see `stage.py:121` - asserts `_is_restored`)

## Important Notes

1. **Forward pass calls restore automatically**: When you call `model(input)`, Fast-LLM internally calls `restore_parameters()` first
2. **Parameters are views**: Modifying parameters after restore modifies the buffer
3. **Chunking parameters**: If you chunk `.weight` or `.bias` before restore, you'll chunk zeros!

## Verification Examples

```python
# ❌ WRONG - will show zeros (before restore)
bias = model.decoder[0].mlp.layer_1.bias
print(bias[0, :10])  # All zeros!

# ✅ CORRECT - access through shard
weights = model.get_shard(ShardName.weights)
print(weights.sum())  # Non-zero!
print(weights.count_nonzero())  # Many non-zero elements

# ✅ ALSO CORRECT - restore first, then access parameters
for stage in model._stages:
    stage.restore_parameters()
bias = model.decoder[0].mlp.layer_1.bias
print(bias.sum())  # Non-zero!
```

## References
- `fast_llm/engine/multi_stage/fsdp.py:82-90` - Parameter buffer creation
- `fast_llm/engine/multi_stage/fsdp.py:181-189` - `restore_parameters()` implementation
- `fast_llm/engine/multi_stage/stage.py:121` - Forward pass asserts `_is_restored`
- `tests/models/test_checkpoint.py:227` - Shard access example
