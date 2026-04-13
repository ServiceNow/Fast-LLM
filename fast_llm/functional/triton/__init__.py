import torch

from fast_llm.utils import InvalidObject, try_decorate

try:
    import triton
    import triton.knobs
    import triton.language as tl

    tl_constexpr = tl.constexpr
    TritonConfig = triton.Config
    # Use `TRITON_INTERPRET=1` to enable triton on CPU.
    triton_interpret = triton.knobs.runtime.interpret
    triton_available = torch.cuda.is_available() or triton_interpret
except ImportError as e:
    triton = InvalidObject(e)
    tl = triton
    tl_constexpr = None
    TritonConfig = lambda *args, **kwargs: None
    triton_interpret = False
    triton_available = False

triton_jit = try_decorate(lambda: triton.jit)
triton_autotune = try_decorate(lambda: triton.autotune)

if not triton_available:
    tl_arange = None
    tl_full = None
elif triton_interpret:
    # Workaround for a triton interpreter bug: constexpr int arguments to device functions
    # arrive as 1-d numpy arrays rather than scalars. The interpreter's _patch_lang_tensor sets
    # tensor.__index__ = lambda self: int(self.handle.data), which fails for 1-d arrays.
    # Patch _patch_lang_tensor to use .item() instead, which works for both 0-d and 1-d arrays.
    import triton.runtime.interpreter as _triton_interpreter

    _orig_patch_lang_tensor = _triton_interpreter._patch_lang_tensor

    def _fixed_patch_lang_tensor(tensor):
        _orig_patch_lang_tensor(tensor)
        tensor.__index__ = lambda self: self.handle.data.item()

    _triton_interpreter._patch_lang_tensor = _fixed_patch_lang_tensor

    @triton_jit
    def tl_arange(start, end):
        return tl.arange(int(start), int(end))

    @triton_jit
    def tl_full(shape, value, dtype):
        return tl.full(tuple(int(x) for x in shape), value, dtype)

else:
    tl_arange = tl.arange
    tl_full = tl.full
