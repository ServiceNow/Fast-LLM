from fast_llm.utils import InvalidObject, try_decorate

try:
    import triton
    import triton.language as tl

    tl_constexpr = tl.constexpr
    TritonConfig = triton.Config
except ImportError as e:
    triton = InvalidObject(e)
    tl = triton
    tl_constexpr = None
    TritonConfig = lambda *args, **kwargs: None

triton_jit = try_decorate(lambda: triton.jit)
triton_autotune = try_decorate(lambda: triton.autotune)
