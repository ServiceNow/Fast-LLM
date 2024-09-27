"""
A collection of external kernels and kernel wrappers used within Fast llm.
TODO: Is is still useful to keep them in one place? (Some like layer norm are added elsewhere.)
Todo: Move all core methods elsewhere (functional?).
"""

import torch

from fast_llm.core.distributed import set_generator

try:
    from amp_C import multi_tensor_adam as _multi_tensor_adam  # noqa
    from amp_C import multi_tensor_l2norm as _multi_tensor_l2norm  # noqa
    from amp_C import multi_tensor_scale as _multi_tensor_scale  # noqa
    from apex.multi_tensor_apply import multi_tensor_applier as _multi_tensor_applier  # noqa

    _apex_available = True
except ImportError:
    _apex_available = False

try:
    from flash_attn.flash_attn_interface import flash_attn_func as _flash_attn_func  # noqa

    _flash_available = True
except ImportError:
    _flash_available = False


def l2_norm(tensors: list[torch.Tensor], noop_flag: torch.Tensor):
    assert _apex_available
    norm, _ = _multi_tensor_applier(
        _multi_tensor_l2norm,
        noop_flag,
        [tensors],
        False,  # no per-parameter norm
    )
    return norm


def scale_(tensors: list[torch.Tensor], noop_flag: torch.Tensor, scale: torch.Tensor | float):
    assert _apex_available
    _multi_tensor_applier(
        _multi_tensor_scale,
        noop_flag,
        [tensors, tensors],
        scale,
    )


# TODO: Same as torch._fused_adam_?
def fused_adam(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    exp_avgs: list[torch.Tensor],
    exp_avg_sqs: list[torch.Tensor],
    noop_flag: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    wd: float,
    eps: float,
    step: int,
):
    _multi_tensor_applier(
        _multi_tensor_adam,
        noop_flag,
        [grads, params, exp_avgs, exp_avg_sqs],
        lr,
        beta1,
        beta2,
        eps,
        step,
        1,  # adamw
        1,  # bias correction
        wd,
    )


def flash_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dropout_p: float,
    *,
    window_size: int | None,
    causal: bool = False,
    generator: torch.Generator | None,
    softmax_scale: float | None = None,
):
    assert _flash_available
    with set_generator(generator):
        return _flash_attn_func(
            query,
            key,
            value,
            window_size=(-1, -1) if window_size is None else (window_size - 1, 0),
            dropout_p=dropout_p,
            causal=causal,
            softmax_scale=softmax_scale,
        )
