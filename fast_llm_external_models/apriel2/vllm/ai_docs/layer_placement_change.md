# Layer Placement Change in Apriel2 vLLM

## Architecture Overview

Apriel2 is a **hybrid model** with heterogeneous decoder layers. Some layers are "stochastic" — they contain **multiple mixer sub-modules** (e.g., attention, GDN, KDA) but only run **one at a time**. The placement system lets you dynamically switch which mixer is active per layer at runtime.

## Key Components

### 1. `Apriel2StochasticMixer` (line ~2344)

- Contains an `nn.ModuleDict` called `self.mixers` with all sub-mixer instances (e.g., attention, GDN, KDA)
- Tracks `self.active_mixer_name` — which mixer is currently active
- All sub-mixers have their weights loaded, but only one runs during forward pass
- Each sub-mixer gets a **virtual layer index** (`layer_idx + (mixer_index+1) * num_layers`) so they each get separate KV cache allocations without collisions

### 2. `Apriel2StochasticDecoderLayer` (line ~2513)

- Wraps `Apriel2StochasticMixer` + MLP + layer norms
- Exposes `set_active_mixer(name)` / `get_active_mixer()` which delegate to the mixer

### 3. Dynamic dispatch via custom op (line ~870)

- `stochastic_mixer_dispatch` is registered as a `vllm::stochastic_mixer_dispatch` custom op
- This op is added to vLLM's `_attention_ops` splitting ops list, causing **graph breaks** in torch.compile
- At runtime, it looks up the `Apriel2StochasticMixer` from `forward_context.no_compile_layers[layer_name]`, reads `active_mixer_name`, and forwards to that mixer
- The fake impl just copies input→output to satisfy the compiler's data dependency analysis

## The Placement Change Call Chain

From the debug script (`debug_offline.py`):

```python
llm.collective_rpc("set_layer_placements", args=(placement,))
```

1. **Worker monkey-patch** (line ~2962): `_patch_worker_for_placement_switching()` runs at import time and adds `set_layer_placements`/`get_layer_placements`/`get_mixer_names` methods to `vllm.v1.worker.gpu_worker.Worker`

2. **`Worker._set_layer_placements`** (line ~3003):
   - First calls `_clear_kv_cache(self)` — zeroes out **all** KV cache tensors to prevent stale data from a different mixer type causing NaN errors
   - Then calls `self.get_model().set_layer_placements(placement)`

3. **`Apriel2ForCausalLM.set_layer_placements`** (line ~2896):
   - Iterates through all layers
   - For each layer that is an `Apriel2StochasticDecoderLayer`, calls `layer.set_active_mixer(mixer_name)` with the corresponding entry from the placement list

4. **`Apriel2StochasticMixer.set_active_mixer`** (line ~2454):
   - Simply sets `self.active_mixer_name = name` (after validation)

5. On the **next `llm.generate()` call**, the forward pass hits `stochastic_mixer_dispatch` which reads the updated `active_mixer_name` and routes to the new mixer.

## Summary Diagram

```
debug_offline.py
  |
  +-- llm.collective_rpc("get_mixer_names")
  |     -> Worker.get_mixer_names -> model.get_mixer_names
  |     -> returns ("attention", "gdn", ...) from first stochastic layer
  |
  +-- llm.collective_rpc("get_layer_placements")
  |     -> Worker.get_layer_placements -> model.get_layer_placements
  |     -> returns {layer_idx: active_mixer_name} for stochastic layers
  |
  +-- llm.collective_rpc("set_layer_placements", args=(placement,))
  |     -> Worker._set_layer_placements
  |         +-- _clear_kv_cache()     <- zero all cache tensors
  |         +-- model.set_layer_placements(placement)
  |              +-- for each stochastic layer:
  |                   layer.mixer.active_mixer_name = new_name
  |
  +-- llm.generate(prompts, ...)
        -> forward pass per layer:
            -> stochastic_mixer_dispatch (custom op, graph break)
                -> looks up self.active_mixer_name
                -> calls active_mixer.forward(hidden_states, output, positions)
```

## Key Insight

All mixer weights are **always loaded** — switching is just flipping `active_mixer_name` and clearing the cache. The custom op mechanism ensures this dynamic routing works even with torch.compile/CUDA graphs by forcing graph breaks at dispatch points.
