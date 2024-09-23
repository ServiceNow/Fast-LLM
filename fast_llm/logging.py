import datetime
import logging
import math
import typing

import torch
import torch._dynamo  # noqa

from fast_llm.engine.base_model.base_model import LossDef
from fast_llm.engine.distributed.config import PhaseType
from fast_llm.tensor import TensorMeta
from fast_llm.utils import format_number

if typing.TYPE_CHECKING:
    from fast_llm.core.distributed import ProcessGroup
    from fast_llm.engine.distributed.distributed import Distributed

logger = logging.getLogger(__name__)


def log(*message, log_fn=logger.info):
    return log_fn(", ".join([str(m() if callable(m) else m) for m in message]))


_MEMORY_METRIC_FORMAT_KEYS = {
    "allocated",
    "max_allocated",
    "reserved",
    "max_reserved",
    "global_max_reserved",
}


_MEMORY_METRIC_FORMAT = (
    " allocated {allocated:,.2f} MiB"
    " | max allocated {max_allocated:,.2f} MiB"
    " | reserved {reserved:,.2f} MiB"
    " | max reserved {max_reserved:,.2f} MiB"
    " | global max reserved {global_max_reserved:,.2f} MiB"
)

_VALIDATION_METRIC_FORMAT_KEYS = _MEMORY_METRIC_FORMAT_KEYS | {
    "iteration",
    "train_iters",
    "consumed_samples",
    "consumed_tokens",
    "step_time_ms",
    "batch_size",
    "model_tflops",
    "hardware_tflops",
    "tokens_per_sec_per_gpu",
}

_VALIDATION_METRIC_FORMATS = (
    "{phase} @ iteration {iteration:6.0f}/{train_iters:6.0f}"
    " | consumed samples: {consumed_samples:12,.0f}"
    " | consumed tokens: {consumed_tokens:16,.0f}"
    " | batch size: {batch_size:3.0f}"
    " | step time: {step_time_ms:.2f} ms"
    " | throughput: {model_tflops:.2f} tflop/s (model)"
    " | {hardware_tflops:.2f} tflop/s (hardware)"
    " | {tokens_per_sec_per_gpu:.2f} tokens/s/gpu"
    " | Memory"
) + _MEMORY_METRIC_FORMAT


_TRAINING_METRIC_FORMAT_KEYS = _VALIDATION_METRIC_FORMAT_KEYS | {
    "learning_rate",
    "loss_scale",
    "grad_norm",
    "skipped_iters",
    "nan_iters",
    "step_time_average_ms",
    "remaining_time",
    "completion_time",
    "percent_done",
}

_TRAINING_METRIC_FORMATS = _VALIDATION_METRIC_FORMATS + (
    " | learning rate: {learning_rate:.3e}"
    " | loss scale: {loss_scale:5.0f}"
    " | grad norm: {grad_norm:.4f}"
    " | skipped iterations: {skipped_iters:3.0f}"
    " | nan iterations: {nan_iters:3.0f}"
    " | average step time {step_time_average_ms:.2f} ms"
    " | remaining {remaining_time} "
    " | completion {completion_time} ({percent_done:.2f} %)"
)
_NAN = float("nan")

_FORMAT_MAP = {
    "remaining_time": lambda x: str(datetime.timedelta(seconds=round(x))),
    "completion_time": lambda x: str(datetime.datetime.fromtimestamp(round(x))),
}

_METRIC_FORMATS_KEYS = {
    PhaseType.training: _TRAINING_METRIC_FORMAT_KEYS,
    PhaseType.validation: _VALIDATION_METRIC_FORMAT_KEYS,
    PhaseType.test: _VALIDATION_METRIC_FORMAT_KEYS,
}
_METRIC_FORMATS = {
    PhaseType.training: _TRAINING_METRIC_FORMATS,
    PhaseType.validation: _VALIDATION_METRIC_FORMATS,
    PhaseType.test: _VALIDATION_METRIC_FORMATS,
}

_max_logged_elements = 8


def set_max_logged_elements(value: int):
    global _max_logged_elements
    _max_logged_elements = value


def format_metrics(metrics: dict[str, float | int], loss_defs: list[LossDef], phase: PhaseType):
    # TODO: Improve, add flexibility.
    metrics = {key: _FORMAT_MAP[key](value) if key in _FORMAT_MAP else value for key, value in metrics.items()}

    outputs = [
        _METRIC_FORMATS[phase].format(
            phase=phase, **{key: metrics.pop(key, _NAN) for key in _METRIC_FORMATS_KEYS[phase]}
        )
    ]
    outputs.extend([f"{loss_def.formatted_name}: {metrics.pop(loss_def.name, _NAN):.5f}" for loss_def in loss_defs])
    if metrics:
        outputs.extend([f"{key}: {value}" for key, value in metrics.items()])

    return " | ".join(outputs)


# A global buffer for holding logged tensor stats.
_tensor_log_stats: list | None = None
_tensor_log_show: bool = True


def set_show_tensor_logs(enabled=True):
    global _tensor_log_show
    _tensor_log_show = enabled


def reset_tensor_stats_logging(enabled=True):
    global _tensor_log_stats
    _tensor_log_stats = [] if enabled else None


def get_logged_tensor_stats():
    return _tensor_log_stats


@torch._dynamo.disable  # noqa
def log_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    scale: float = 1.0,
    level: int = 2,
    storage: bool = False,
    log_fn: typing.Callable[[str], typing.Any] | None = logger.info,
):
    if level < 1:
        return
    save_stats = _tensor_log_stats is not None
    shape = tuple(tensor.shape)
    _, dtype = str(tensor.dtype).split("torch.")
    txt = [
        (None, name, 50),
        ("shape", shape, 18),
        ("dtype", dtype, 9),
        ("device", tensor.device, 7),
    ]
    stats = dict(
        name=name,
        shape=list(shape),
        dtype=dtype,
        device=str(tensor.device),
    )
    if level >= 2 and tensor.device.type != "meta":
        v_float = tensor.float()

        stats.update(
            mu=v_float.mean().item(),
            std=v_float.std().item() if v_float.numel() > 1 else math.nan,
            stride=tensor.stride(),  # noqa
        )
        if save_stats:
            stats.update(
                min=v_float.min().item(),
                max=v_float.max().item(),
            )
        txt.extend(
            [
                ("mu", format_number(stats["mu"] * scale), 10),
                ("std", format_number(stats["std"] * scale), 10),
                ("stride", stats["stride"], 20),
            ]
        )
        if storage:
            storage = tensor.untyped_storage()
            storage_float = torch.tensor(storage, dtype=tensor.dtype, device=tensor.device).float()
            stats.update(
                storage=str(storage.data_ptr())[-8:],
                storage_size=storage.size(),
                storage_mu=storage_float.mean().item() * scale,
                storage_std=storage_float.std().item() * scale,
            )
            txt.extend(
                [
                    (f"storage", stats["storage"], 8),
                    (f"s size", f"{stats['storage_size']:,d}", 12),
                    (f"s mu", format_number(stats["storage_mu"]), 10),
                    (f"s std", format_number(stats["storage_std"]), 10),
                ]
            )
        if level >= 3:
            target_samples = 2 ** (level - 3)
            step = max(tensor.numel() // target_samples, 1)
            while step > 1 and any(step % s == 0 and s > 1 for s in shape):
                step -= 1
            samples = tensor.flatten()[: target_samples * step : step].cpu()
            stats.update(samples=samples, step=step)
            # Crop the list in the logs. The full tensor is still in stats.
            samples = [format_number(x) for x in samples.tolist()[:_max_logged_elements]]
            num_logged_elements = len(samples)
            samples = ",".join(f"{sample:10s}" for sample in samples)
            txt.append((f"{f'samples (step={step})':21s}", f" ({samples})", num_logged_elements * 11 + 3))
    out, len_ = "", 0
    if save_stats:
        stats.update(
            min=v_float.min().item(),  # noqa
            max=v_float.max().item(),  # noqa
        )
        _tensor_log_stats.append(stats)
    for prefix, val, col_len in txt:
        prefix = "" if prefix is None else f" {prefix}="
        len_ += col_len + len(prefix) + 1
        out = f"{f'{out}{prefix}{str(val)}':{len_}s}"
    if _tensor_log_show:
        return log_fn(out)


@torch._dynamo.disable  # noqa
def log_grad(
    name: str,
    tensor: torch.Tensor,
    *,
    scale: float = 1.0,
    level: int = 2,
    storage: bool = False,
    grad_fn: typing.Callable[[torch.Tensor], torch.Tensor] | None = None,
    log_fn: typing.Callable[[str], typing.Any] | None = logger.info,
):
    tensor.register_hook(
        lambda grad: log_tensor(
            name,
            grad if grad_fn is None else grad_fn(grad),
            scale=scale,
            level=level,
            storage=storage,
            log_fn=log_fn,
        )
    )


@torch._dynamo.disable  # noqa
def log_distributed_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    scale: float = 1.0,
    level: int = 2,
    storage: bool = False,
    distributed: "Distributed",
    duplicate_groups: tuple[typing.Optional["ProcessGroup"], ...] = (),
    global_: bool = True,
    log_fn: typing.Callable[[str], typing.Any] | None = logger.info,
    meta: TensorMeta,
):
    if level <= 0:
        return
    if global_:
        tensor, is_first_rank = meta.local_to_global(tensor, distributed=distributed)
        storage = False
        is_first_rank = is_first_rank and all(group.rank() == 0 for group in duplicate_groups if group)
        if not is_first_rank:
            log_fn = None
    if log_fn is not None:
        return log_tensor(
            f"{'Global' if global_ else 'Local'} {name}: {meta.tensor_name}",
            tensor,
            level=level,
            scale=scale,
            storage=storage,
            log_fn=log_fn,
        )


@torch._dynamo.disable  # noqa
def log_distributed_grad(
    name: str,
    tensor: torch.Tensor,
    *,
    scale: float = 1.0,
    level: int = 2,
    storage: bool = False,
    distributed: "Distributed",
    duplicate_groups: tuple[typing.Optional["ProcessGroup"], ...] = (),
    grad_fn: typing.Callable[[torch.Tensor], torch.Tensor] | None = None,
    global_: bool = True,
    log_fn: typing.Callable[[str], None] | None = logger.info,
    meta: TensorMeta,
):
    if level <= 0:
        return
    tensor.register_hook(
        lambda grad: log_distributed_tensor(
            name,
            grad if grad_fn is None else grad_fn(grad),
            scale=scale,
            level=level,
            storage=storage,
            distributed=distributed,
            duplicate_groups=duplicate_groups,
            global_=global_,
            log_fn=log_fn,
            meta=meta,
        )
    )


@torch._dynamo.disable  # noqa
def log_generator(
    name,
    generator: torch.Tensor | torch.Generator | None = None,
    log_fn: typing.Callable[[str], typing.Any] | None = logger.info,
):
    if generator is None:
        generator = torch.cuda.default_generators[torch.cuda.current_device()]
    tensor = generator.get_state() if isinstance(generator, torch.Generator) else generator
    return log_fn(f"{name} {tensor.view(dtype=torch.int64)[-8:].tolist()}")


_global_max_reserved = 0


def get_memory_usage_mib(reset_stats: bool = True, relative_to: dict[str, int] | None = None):
    global _global_max_reserved
    max_reserved = torch.cuda.max_memory_reserved() / 2**20
    _global_max_reserved = max(max_reserved, _global_max_reserved)
    out = {
        "allocated": torch.cuda.memory_allocated() / 2**20,
        "max_allocated": torch.cuda.max_memory_allocated() / 2**20,
        "reserved": torch.cuda.memory_reserved() / 2**20,
        "max_reserved": max_reserved,
        "global_max_reserved": _global_max_reserved,
    }
    if relative_to:
        out = {key: value - relative_to.get(key, 0) for key, value in out.items()}
    if reset_stats:
        torch.cuda.reset_peak_memory_stats()
    return out


def log_memory_usage(
    header: str | None = None,
    log_fn: typing.Callable[[str], typing.Any] | None = logger.info,
    reset_stats: bool = True,
    stats: dict[str, int] | None = None,
    relative_to: dict[str, int] | None = None,
):
    if stats is None:
        stats = get_memory_usage_mib(reset_stats, relative_to)
    formatted = _MEMORY_METRIC_FORMAT.format(**stats)
    if header is not None:
        formatted = f"{header}: {formatted}"
    return log_fn(formatted)
