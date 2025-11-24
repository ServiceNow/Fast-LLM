from __future__ import annotations
import torch
from transformers.cache_utils import Cache


class _AttentionCache:
    __slots__ = ["key", "value", "window"]

    def __init__(self, window=None):
        self.key = None
        self.value = None
        self.window = window

    def update(self, key, value):
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


class _SSMCache:
    __slots__ = ["conv", "recurrent"]

    def __init__(self):
        self.conv = None
        self.recurrent = None


class _DummyCacheLayer:
    pass


class Apriel2Cache(Cache):

    def __init__(self, config):
        super().__init__(layer_class_to_replicate=_DummyCacheLayer)
        self.config = config
        n = config.num_hidden_layers
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
                        sub[name] = _AttentionCache(cfg.get("sliding_window"))
                    else:
                        sub[name] = _SSMCache()
                self.layers.append(sub)
                self.mixer_types.append(mixer["mixers"][main].get("type") if main else "attention")
            elif mtype == "attention":
                self.layers.append(_AttentionCache(mixer.get("sliding_window")))
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
        layer = self.layers[layer_idx]
        if isinstance(layer, dict):
            mixer = self.active_mixers[layer_idx]
            if mixer and isinstance(layer[mixer], _AttentionCache):
                return layer[mixer].key.shape[-2] if layer[mixer].key is not None else 0
            return 0
        if isinstance(layer, _AttentionCache):
            return layer.key.shape[-2] if layer.key is not None else 0
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
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        kv_offset = past_seen_tokens
        return kv_length, kv_offset

    @property
    def has_previous_state(self):
        for layer in self.layers:
            if isinstance(layer, dict):
                for cache in layer.values():
                    if isinstance(cache, _SSMCache) and cache.conv is not None:
                        return True
            elif isinstance(layer, _SSMCache) and layer.conv is not None:
                return True
        return False

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

    def reorder_cache(self, beam_idx):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, dict):
                for cache in layer.values():
                    self._reorder_cache_obj(cache, beam_idx)
            else:
                self._reorder_cache_obj(layer, beam_idx)

    def _reorder_cache_obj(self, cache, beam_idx):
        if isinstance(cache, _AttentionCache):
            if cache.key is not None:
                cache.key = cache.key.index_select(0, beam_idx.to(cache.key.device))
                cache.value = cache.value.index_select(0, beam_idx.to(cache.value.device))
        elif isinstance(cache, _SSMCache):
            if cache.conv is not None:
                cache.conv = cache.conv.index_select(0, beam_idx.to(cache.conv.device))
            if cache.recurrent is not None:
                cache.recurrent = cache.recurrent.index_select(0, beam_idx.to(cache.recurrent.device))

    def reset(self):
        for layer in self.layers:
            if isinstance(layer, dict):
                for cache in layer.values():
                    self._reset_cache_obj(cache)
            else:
                self._reset_cache_obj(layer)

    def _reset_cache_obj(self, cache):
        if isinstance(cache, _AttentionCache):
            cache.key = None
            cache.value = None
        elif isinstance(cache, _SSMCache):
            cache.conv = None
            cache.recurrent = None

    def crop(self, max_length):
        for layer in self.layers:
            if isinstance(layer, dict):
                for cache in layer.values():
                    if isinstance(cache, _AttentionCache) and cache.key is not None:
                        cache.key = cache.key[..., :max_length, :]
                        cache.value = cache.value[..., :max_length, :]
            elif isinstance(layer, _AttentionCache) and layer.key is not None:
                layer.key = layer.key[..., :max_length, :]
                layer.value = layer.value[..., :max_length, :]

    def batch_repeat_interleave(self, repeats):
        for layer in self.layers:
            if isinstance(layer, dict):
                for cache in layer.values():
                    self._batch_repeat_cache_obj(cache, repeats)
            else:
                self._batch_repeat_cache_obj(layer, repeats)

    def _batch_repeat_cache_obj(self, cache, repeats):
        if isinstance(cache, _AttentionCache):
            if cache.key is not None:
                cache.key = cache.key.repeat_interleave(repeats, dim=0)
                cache.value = cache.value.repeat_interleave(repeats, dim=0)
        elif isinstance(cache, _SSMCache):
            if cache.conv is not None:
                cache.conv = cache.conv.repeat_interleave(repeats, dim=0)
            if cache.recurrent is not None:
                cache.recurrent = cache.recurrent.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices):
        for layer in self.layers:
            if isinstance(layer, dict):
                for cache in layer.values():
                    self._batch_select_cache_obj(cache, indices)
            else:
                self._batch_select_cache_obj(layer, indices)

    def _batch_select_cache_obj(self, cache, indices):
        if isinstance(cache, _AttentionCache):
            if cache.key is not None:
                cache.key = cache.key.index_select(0, indices.to(cache.key.device))
                cache.value = cache.value.index_select(0, indices.to(cache.value.device))
        elif isinstance(cache, _SSMCache):
            if cache.conv is not None:
                cache.conv = cache.conv.index_select(0, indices.to(cache.conv.device))
            if cache.recurrent is not None:
                cache.recurrent = cache.recurrent.index_select(0, indices.to(cache.recurrent.device))

    @property
    def is_compileable(self):
        return False

    @property
    def is_initialized(self):
        for layer in self.layers:
            if isinstance(layer, dict):
                for cache in layer.values():
                    if isinstance(cache, _AttentionCache) and cache.key is not None:
                        return True
                    if isinstance(cache, _SSMCache) and cache.conv is not None:
                        return True
            else:
                if isinstance(layer, _AttentionCache) and layer.key is not None:
                    return True
                if isinstance(layer, _SSMCache) and layer.conv is not None:
                    return True
        return False

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
        for layer in self.layers:
            if isinstance(layer, dict):
                for cache in layer.values():
                    if isinstance(cache, _AttentionCache) and cache.key is not None:
                        return cache.key.shape[0]
                    if isinstance(cache, _SSMCache) and cache.conv is not None:
                        return cache.conv.shape[0]
            else:
                if isinstance(layer, _AttentionCache) and layer.key is not None:
                    return layer.key.shape[0]
                if isinstance(layer, _SSMCache) and layer.conv is not None:
                    return layer.conv.shape[0]
        return None

    @property
    def max_cache_len(self):
        max_len = None
        for layer in self.layers:
            if isinstance(layer, dict):
                for cache in layer.values():
                    if isinstance(cache, _AttentionCache):
                        if cache.window is not None:
                            max_len = cache.window if max_len is None else min(max_len, cache.window)
            elif isinstance(layer, _AttentionCache):
                if layer.window is not None:
                    max_len = layer.window if max_len is None else min(max_len, layer.window)
        return max_len

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
