import torch

from fast_llm.utils import InvalidObject, try_decorate

try:
    import triton
    import triton.knobs
    import triton.language as tl

    tl_constexpr = tl.constexpr
    TritonConfig = triton.Config
    triton_available = torch.cuda.is_available() or triton.knobs.runtime.interpret
except ImportError as e:
    triton = InvalidObject(e)
    tl = triton
    tl_constexpr = None
    TritonConfig = lambda *args, **kwargs: None
    triton_available = False

triton_jit = try_decorate(lambda: triton.jit)
triton_autotune = try_decorate(lambda: triton.autotune)


if not triton_available:
    tl_arange = None
elif triton.knobs.runtime.interpret:
    # Workaround for a triton bug.
    @triton_jit
    def tl_arange(start, end):
        return tl.arange(int(start), int(end))

else:
    tl_arange = tl.arange
