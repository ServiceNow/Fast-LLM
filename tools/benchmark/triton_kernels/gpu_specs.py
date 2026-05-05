"""
Peak memory bandwidth and compute for high-end datacenter GPUs, used to report
%-of-peak. Values are vendor-published nominal dense peaks (no 2:4 sparsity);
real-world achievable is typically ~80-90% of nominal for bandwidth and
~70-85% for compute.
"""

import dataclasses

import torch


@dataclasses.dataclass(frozen=True)
class GpuSpec:
    name: str
    peak_bandwidth_gbps: float
    peak_tflops_fp32: float
    peak_tflops_tf32: float
    peak_tflops_fp16: float
    peak_tflops_bf16: float

    def peak_tflops(self, dtype: torch.dtype) -> float | None:
        if dtype == torch.float32:
            return self.peak_tflops_fp32
        if dtype == torch.float16:
            return self.peak_tflops_fp16
        if dtype == torch.bfloat16:
            return self.peak_tflops_bf16
        return None


# Name-substring match → spec. First match wins. Dense (non-sparse) peaks.
_GPU_SPECS: list[tuple[str, GpuSpec]] = [
    (
        "B200",
        GpuSpec(
            name="NVIDIA B200 SXM",
            peak_bandwidth_gbps=8000.0,
            peak_tflops_fp32=80.0,
            peak_tflops_tf32=1100.0,
            peak_tflops_fp16=2250.0,
            peak_tflops_bf16=2250.0,
        ),
    ),
    (
        "B100",
        GpuSpec(
            name="NVIDIA B100 SXM",
            peak_bandwidth_gbps=8000.0,
            peak_tflops_fp32=60.0,
            peak_tflops_tf32=880.0,
            peak_tflops_fp16=1800.0,
            peak_tflops_bf16=1800.0,
        ),
    ),
    (
        "H200",
        GpuSpec(
            name="NVIDIA H200 SXM",
            peak_bandwidth_gbps=4800.0,
            peak_tflops_fp32=67.0,
            peak_tflops_tf32=494.0,
            peak_tflops_fp16=989.0,
            peak_tflops_bf16=989.0,
        ),
    ),
    (
        "H100",
        GpuSpec(
            name="NVIDIA H100 SXM",
            peak_bandwidth_gbps=3350.0,
            peak_tflops_fp32=67.0,
            peak_tflops_tf32=494.0,
            peak_tflops_fp16=989.0,
            peak_tflops_bf16=989.0,
        ),
    ),
    (
        "A100",
        GpuSpec(
            name="NVIDIA A100 80GB SXM",
            peak_bandwidth_gbps=2039.0,
            peak_tflops_fp32=19.5,
            peak_tflops_tf32=156.0,
            peak_tflops_fp16=312.0,
            peak_tflops_bf16=312.0,
        ),
    ),
]


def detect_gpu_spec() -> GpuSpec | None:
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name()
    for needle, spec in _GPU_SPECS:
        if needle in name:
            return spec
    return None
