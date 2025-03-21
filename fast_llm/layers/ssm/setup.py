import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def append_nvcc_threads(nvcc_extra_args):
    """Append nvcc thread configuration."""
    return nvcc_extra_args + ["--threads", "4"]

def get_cuda_compute_capability():
    """Get CUDA compute capability flags."""
    import torch
    # Get CUDA version and capabilities
    cuda_version = torch.version.cuda.split('.')
    major, minor = int(cuda_version[0]), int(cuda_version[1])
    
    compute_capabilities = []
    for device in range(torch.cuda.device_count()):
        major_minor = torch.cuda.get_device_capability(device)
        if major_minor not in compute_capabilities:
            compute_capabilities.append(major_minor)
    
    # Generate capability flags
    cc_flag = []
    for major, minor in compute_capabilities:
        cc_flag.extend(['-gencode', f'arch=compute_{major}{minor},code=sm_{major}{minor}'])
    
    return cc_flag

# Get the directory containing the setup.py
this_dir = os.path.dirname(os.path.abspath(__file__))
# Get the directory containing the CUDA source files
csrc_dir = os.path.join(this_dir, "csrc", "selective_scan")

setup(
    name='selective_scan_cuda',
    ext_modules=[
        CUDAExtension(
            name="selective_scan_cuda",
            sources=[
                os.path.join(csrc_dir, "selective_scan.cpp"),
                os.path.join(csrc_dir, "selective_scan_fwd_fp32.cu"),
                os.path.join(csrc_dir, "selective_scan_fwd_fp16.cu"),
                os.path.join(csrc_dir, "selective_scan_fwd_bf16.cu"),
                os.path.join(csrc_dir, "selective_scan_bwd_fp32_real.cu"),
                os.path.join(csrc_dir, "selective_scan_bwd_fp32_complex.cu"),
                os.path.join(csrc_dir, "selective_scan_bwd_fp16_real.cu"),
                os.path.join(csrc_dir, "selective_scan_bwd_fp16_complex.cu"),
                os.path.join(csrc_dir, "selective_scan_bwd_bf16_real.cu"),
                os.path.join(csrc_dir, "selective_scan_bwd_bf16_complex.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": append_nvcc_threads([
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v",
                    "-lineinfo",
                ] + get_cuda_compute_capability()),
            },
            include_dirs=[csrc_dir],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)