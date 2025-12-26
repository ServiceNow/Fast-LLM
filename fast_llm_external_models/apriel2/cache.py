from __future__ import annotations
import torch
from transformers.cache_utils import Cache


class _AttentionCache:
    __slots__ = ["key", "value", "window", "cumulative_length"]

    def __init__(self, window=None):
        self.key = None
        self.value = None
        self.window = window
        self.cumulative_length = 0

    def update(self, key, value):
        new_tokens = key.shape[-2]
        self.cumulative_length += new_tokens

        if self.key is None:
            if self.window and key.shape[-2] > self.window:
                self.key = key[..., -self.window :, :].contiguous()
                self.value = value[..., -self.window :, :].contiguous()
            else:
                self.key = key.contiguous()
                self.value = value.contiguous()
        else:
            if self.window:
                self.key = self._window(self.key, key)
                self.value = self._window(self.value, value)
            else:
                self.key = torch.cat([self.key, key], -2)
                self.value = torch.cat([self.value, value], -2)
        return self.key, self.value

    def _window(self, cache, new):
        if cache.shape[-2] == self.window and new.shape[-2] == 1:
            cache = cache.roll(-1, -2)
            cache[..., -1:, :] = new
            return cache
        return torch.cat([cache, new], -2)[..., -self.window :, :].contiguous()

    def reset(self):
        self.key = None
        self.value = None
        self.cumulative_length = 0

    def reorder(self, beam_idx):
        if self.key is not None:
            self.key = self.key.index_select(0, beam_idx.to(self.key.device))
            self.value = self.value.index_select(0, beam_idx.to(self.value.device))

    def crop(self, max_length):
        if self.key is not None:
            self.key = self.key[..., :max_length, :]
            self.value = self.value[..., :max_length, :]
            self.cumulative_length = self.key.shape[-2]

    def batch_repeat(self, repeats):
        if self.key is not None:
            self.key = self.key.repeat_interleave(repeats, dim=0)
            self.value = self.value.repeat_interleave(repeats, dim=0)

    def batch_select(self, indices):
        if self.key is not None:
            self.key = self.key.index_select(0, indices.to(self.key.device))
            self.value = self.value.index_select(0, indices.to(self.value.device))

    @property
    def is_initialized(self):
        return self.key is not None

    @property
    def batch_size(self):
        return self.key.shape[0] if self.key is not None else None


class _SSMCache:
    __slots__ = ["conv", "recurrent"]

    def __init__(self):
        self.conv = None
        self.recurrent = None

    def reset(self):
        self.conv = None
        self.recurrent = None

    def reorder(self, beam_idx):
        if self.conv is not None:
            if isinstance(self.conv, tuple):
                self.conv = tuple(c.index_select(0, beam_idx.to(c.device)) for c in self.conv)
            else:
                self.conv = self.conv.index_select(0, beam_idx.to(self.conv.device))
        if self.recurrent is not None:
            self.recurrent = self.recurrent.index_select(0, beam_idx.to(self.recurrent.device))

    def crop(self, max_length):
        pass  # SSM caches don't have sequence dimension to crop

    def batch_repeat(self, repeats):
        if self.conv is not None:
            if isinstance(self.conv, tuple):
                self.conv = tuple(c.repeat_interleave(repeats, dim=0) for c in self.conv)
            else:
                self.conv = self.conv.repeat_interleave(repeats, dim=0)
        if self.recurrent is not None:
            self.recurrent = self.recurrent.repeat_interleave(repeats, dim=0)

    def batch_select(self, indices):
        if self.conv is not None:
            if isinstance(self.conv, tuple):
                self.conv = tuple(c.index_select(0, indices.to(c.device)) for c in self.conv)
            else:
                self.conv = self.conv.index_select(0, indices.to(self.conv.device))
        if self.recurrent is not None:
            self.recurrent = self.recurrent.index_select(0, indices.to(self.recurrent.device))

    @property
    def is_initialized(self):
        return self.conv is not None

    @property
    def batch_size(self):
        if self.conv is None:
            return None
        if isinstance(self.conv, tuple):
            return self.conv[0].shape[0]
        return self.conv.shape[0]


class _DummyCacheLayer:
    pass


class Apriel2Cache(Cache):

    def __init__(self, config):
        super().__init__(layer_class_to_replicate=_DummyCacheLayer)
        self.config = config
        n = config.decoder["num_blocks"]
        self.layers = []
        self.mixer_types = []
        self.active_mixers = [None] * n

        for i in range(n):
            block = config.get_block_config(i)
            mixer = block.get("mixer", {})
            mtype = mixer.get("type", "attention")

            if mtype == "stochastic":
                sub = {}
                main = mixer.get("main_mixer_name")
                for name, cfg in mixer.get("mixers", {}).items():
                    if cfg.get("type") == "attention":
                        sub[name] = _AttentionCache(cfg.get("window_size"))
                    else:
                        sub[name] = _SSMCache()
                self.layers.append(sub)
                self.mixer_types.append(mixer["mixers"][main].get("type") if main else "attention")
            elif mtype == "attention":
                self.layers.append(_AttentionCache(mixer.get("window_size")))
                self.mixer_types.append("attention")
            else:
                self.layers.append(_SSMCache())
                self.mixer_types.append(mtype)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        layer = self.layers[layer_idx]
        if isinstance(layer, dict):
            mixer = self.active_mixers[layer_idx]
            if mixer is None:
                raise RuntimeError(f"Stochastic layer {layer_idx} needs active_mixer set")
            return layer[mixer].update(key_states, value_states)
        return layer.update(key_states, value_states)

    def set_active_mixer(self, layer_idx, mixer_name):
        self.active_mixers[layer_idx] = mixer_name

    def get_seq_length(self, layer_idx=0):
        """Returns the cumulative sequence length of tokens seen by the cache.

        For sliding window caches, this returns the total tokens seen (not just cached).
        This matches HuggingFace's DynamicSlidingWindowLayer behavior.
        """
        layer = self.layers[layer_idx]
        if isinstance(layer, dict):
            mixer = self.active_mixers[layer_idx]
            if mixer and isinstance(layer[mixer], _AttentionCache):
                return layer[mixer].cumulative_length
            return 0
        if isinstance(layer, _AttentionCache):
            return layer.cumulative_length
        return 0

    def get_max_cache_shape(self, layer_idx=0):
        layer = self.layers[layer_idx]
        if isinstance(layer, dict):
            mixer = self.active_mixers[layer_idx]
            if mixer and isinstance(layer[mixer], _AttentionCache):
                return layer[mixer].window
        elif isinstance(layer, _AttentionCache):
            return layer.window
        return None

    def get_mask_sizes(self, cache_position, layer_idx):
        """Return the length and offset of the cache, used to generate the attention mask.

        For standard (non-sliding) attention:
            kv_offset = 0 (KV[0] corresponds to sequence position 0)
            kv_length = cumulative_length + query_length

        For sliding window attention:
            kv_offset = max(cumulative_length - window + 1, 0)
            kv_length = min(cumulative_length, window - 1) + query_length

        For SSM/linear layers:
            kv_offset = 0, kv_length = query_length (no KV cache to attend to)
        """
        query_length = cache_position.shape[0]
        layer = self.layers[layer_idx]

        # Handle stochastic layers by getting the active mixer's cache
        if isinstance(layer, dict):
            mixer = self.active_mixers[layer_idx]
            if mixer is None:
                # No active mixer set, return defaults
                return query_length, 0
            cache = layer[mixer]
        else:
            cache = layer

        # SSM layers don't have KV cache for attention mask purposes
        if isinstance(cache, _SSMCache):
            return query_length, 0

        # Attention cache - check if sliding window
        if isinstance(cache, _AttentionCache):
            cumulative = cache.cumulative_length
            window = cache.window

            if window is not None:
                # Sliding window attention
                kv_offset = max(cumulative - window + 1, 0)
                if cumulative >= window:
                    kv_length = window - 1 + query_length
                else:
                    kv_length = cumulative + query_length
            else:
                # Full attention
                kv_offset = 0
                kv_length = cumulative + query_length

            return kv_length, kv_offset

        # Fallback
        return query_length, 0

    @property
    def has_previous_state(self):
        return any(isinstance(cache, _SSMCache) and cache.conv is not None for cache in self._iter_caches())

    @property
    def key_cache(self):
        return _LayerListAccessor(self, "key")

    @property
    def value_cache(self):
        return _LayerListAccessor(self, "value")

    @property
    def conv_states(self):
        return _LayerListAccessor(self, "conv")

    @property
    def recurrent_states(self):
        return _LayerListAccessor(self, "recurrent")

    def _iter_caches(self):
        """Iterate over all leaf cache objects (flattening stochastic layer dicts)."""
        for layer in self.layers:
            if isinstance(layer, dict):
                yield from layer.values()
            else:
                yield layer

    def reorder_cache(self, beam_idx):
        for cache in self._iter_caches():
            cache.reorder(beam_idx)

    def reset(self):
        for cache in self._iter_caches():
            cache.reset()

    def crop(self, max_length):
        for cache in self._iter_caches():
            cache.crop(max_length)

    def batch_repeat_interleave(self, repeats):
        for cache in self._iter_caches():
            cache.batch_repeat(repeats)

    def batch_select_indices(self, indices):
        for cache in self._iter_caches():
            cache.batch_select(indices)

    @property
    def is_compileable(self):
        return False

    @property
    def is_initialized(self):
        return any(cache.is_initialized for cache in self._iter_caches())

    @property
    def is_sliding(self):
        result = []
        for layer in self.layers:
            if isinstance(layer, dict):
                has_sliding = any(
                    isinstance(cache, _AttentionCache) and cache.window is not None for cache in layer.values()
                )
                result.append(has_sliding)
            elif isinstance(layer, _AttentionCache):
                result.append(layer.window is not None)
            else:
                result.append(False)
        return result

    @property
    def max_batch_size(self):
        for cache in self._iter_caches():
            bs = cache.batch_size
            if bs is not None:
                return bs
        return None

    @property
    def max_cache_len(self):
        windows = [
            cache.window
            for cache in self._iter_caches()
            if isinstance(cache, _AttentionCache) and cache.window is not None
        ]
        return min(windows) if windows else None

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, idx):
        layer = self.layers[idx]
        if isinstance(layer, dict):
            mixer = self.active_mixers[idx]
            if mixer and isinstance(layer[mixer], _AttentionCache):
                c = layer[mixer]
                if c.key is not None:
                    return c.key, c.value
        elif isinstance(layer, _AttentionCache):
            if layer.key is not None:
                return layer.key, layer.value

        for i, l in enumerate(self.layers):
            if isinstance(l, _AttentionCache) and l.key is not None:
                return torch.empty((0,), device=l.key.device, dtype=l.key.dtype), torch.empty(
                    (0,), device=l.key.device, dtype=l.key.dtype
                )
            elif isinstance(l, dict):
                for c in l.values():
                    if isinstance(c, _AttentionCache) and c.key is not None:
                        return torch.empty((0,), device=c.key.device, dtype=c.key.dtype), torch.empty(
                            (0,), device=c.key.device, dtype=c.key.dtype
                        )
        return torch.empty((0,)), torch.empty((0,))


class _LayerListAccessor:
    __slots__ = ["cache", "attr"]

    def __init__(self, cache, attr):
        self.cache = cache
        self.attr = attr

    def __getitem__(self, idx):
        layer = self.cache.layers[idx]
        if isinstance(layer, dict):
            mixer = self.cache.active_mixers[idx]
            if mixer is None:
                raise RuntimeError(
                    f"Stochastic layer {idx} requires set_active_mixer() to be called before accessing cache. "
                    f"Available mixers: {list(layer.keys())}"
                )
            return getattr(layer[mixer], self.attr)
        return getattr(layer, self.attr, None)

    def __setitem__(self, idx, value):
        layer = self.cache.layers[idx]
        if isinstance(layer, dict):
            mixer = self.cache.active_mixers[idx]
            if mixer is None:
                raise RuntimeError(
                    f"Stochastic layer {idx} requires set_active_mixer() to be called before accessing cache. "
                    f"Available mixers: {list(layer.keys())}"
                )
            setattr(layer[mixer], self.attr, value)
        elif hasattr(layer, self.attr):
            setattr(layer, self.attr, value)
