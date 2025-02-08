"""
An experimental triton implementation of Adam.
Not currently is use.
Simpler and faster than the apex implementation, but doesn't have a multi-tensor version.
"""

import torch
from torch.optim.adamw import adamw  # noqa

from fast_llm.functional.config import TritonConfig
from fast_llm.functional.triton import tl, tl_constexpr, triton, triton_jit


@triton_jit()
def triton_adam_kernel(
    params_ptr,
    grads_ptr,
    exp_avgs_ptr,
    exp_avg_sqs_ptr,
    noop_flag_ptr,
    scale_ptr,
    step_size,  # lr / (1 - beta1 ** step)
    beta1,
    beta2,
    bias_correction,  # (1 - beta2 ** step)**0.5
    decay_factor,  # (1 - lr * weight_decay)
    epsilon,
    numel: tl_constexpr,
    block_size: tl_constexpr,
):
    noop_flag = tl.load(noop_flag_ptr)
    # TODO: Does location matter?
    if noop_flag != 0:
        return

    scale = tl.load(scale_ptr)

    # TODO: Int64 ptr only if needed?
    block_start = tl.program_id(axis=0).to(tl.int64) * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < numel

    params = tl.load(params_ptr + offsets, mask=mask)
    grads = tl.load(grads_ptr + offsets, mask=mask)

    grads = scale * grads

    exp_avgs = tl.load(exp_avgs_ptr + offsets, mask=mask)
    exp_avgs = beta1 * exp_avgs + (1 - beta1) * grads
    tl.store(exp_avgs_ptr + offsets, exp_avgs, mask=mask)
    # tl.device_print("exp_avgs",exp_avgs)

    exp_avg_sqs = tl.load(exp_avg_sqs_ptr + offsets, mask=mask)
    exp_avg_sqs = beta2 * exp_avg_sqs + (1 - beta2) * grads * grads
    tl.store(exp_avg_sqs_ptr + offsets, exp_avg_sqs, mask=mask)
    # tl.device_print("exp_avg_sqs",exp_avg_sqs)
    # tl.device_print("update",params,step_size, bias_correction, epsilon)

    params = decay_factor * params - step_size * exp_avgs / (tl.sqrt(exp_avg_sqs) / bias_correction + epsilon)
    tl.store(params_ptr + offsets, params, mask=mask)


def triton_adam(
    params: torch.Tensor,
    grads: torch.Tensor,
    exp_avgs: torch.Tensor,
    exp_avg_sqs: torch.Tensor,
    noop_flag: torch.Tensor,
    grad_scale: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    step: int,
    weight_decay: float,
    epsilon: float,
    use_triton=True,
) -> None:
    if not use_triton or (use_triton is None and TritonConfig.TRITON_ENABLED):
        if noop_flag.item() == 0:
            return adamw(
                [params],
                [grad_scale * grads],
                [exp_avgs],
                [exp_avg_sqs],
                [],
                [params.new_full((1,), step)],
                amsgrad=False,
                beta1=beta1,
                beta2=beta2,
                lr=lr,
                weight_decay=weight_decay,
                eps=epsilon,
                maximize=False,
            )
    # TODO: Improve assumptions.
    assert params.is_contiguous()
    assert grads.is_contiguous()
    assert exp_avgs.is_contiguous()
    assert exp_avg_sqs.is_contiguous()
    numel = params.numel()
    triton_adam_kernel[lambda meta: (triton.cdiv(numel, meta["block_size"]),)](
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        noop_flag,
        grad_scale,
        lr / (1 - beta1**step),
        beta1,
        beta2,
        (1 - beta2**step) ** 0.5,
        (1 - lr * weight_decay),
        epsilon,
        numel,  # noqa
        block_size=TritonConfig.POINTWISE_BLOCK_SIZE,  # noqa
    )
