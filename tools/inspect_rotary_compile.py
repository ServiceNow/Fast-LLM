"""
Dump the Triton kernel that torch.compile generates for the rotary embedding,
so we can compare it to our hand-written fast_llm kernel.

Run on a GPU node:
    python tools/inspect_rotary_compile.py
Output lands in /tmp/torchinductor_*/  (one subdir per compiled function).
This script also prints the path and first 200 lines of each .py file found.
"""

import os

import torch
import torch._inductor.config as inductor_config

# Route torch.compile output to a known directory.
_OUT = "/tmp/torchinductor_rotary_inspect"
os.makedirs(_OUT, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = _OUT


inductor_config.debug = True  # writes generated Triton .py files alongside the cache

tokens, num_heads, head_size = 4096, 32, 128
rotary_dim = head_size // 2
dtype = torch.bfloat16
device = "cuda"

input_ = torch.randn(tokens, num_heads, head_size, dtype=dtype, device=device)
frequencies = torch.randn(tokens, 2 * rotary_dim, dtype=torch.float32, device=device)


def _rotary_eager(input_: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    rotary_dim = frequencies.shape[-1] // 2
    freq_re = frequencies[:, :rotary_dim].unsqueeze(1)
    freq_im = frequencies[:, rotary_dim:].unsqueeze(1)
    x_re, x_im = input_.chunk(2, dim=-1)
    out_re = x_re * freq_re - x_im * freq_im
    out_im = x_im * freq_re + x_re * freq_im
    return torch.cat([out_re, out_im], dim=-1)


compiled = torch.compile(_rotary_eager, mode="default", dynamic=False)

# Trigger compilation.
out = compiled(input_, frequencies)
torch.cuda.synchronize()
print(f"Output shape: {out.shape}, dtype: {out.dtype}")
print(f"\nInductor cache / debug output dir: {_OUT}")

# Find and print the generated Triton kernel files.
for root, dirs, files in os.walk(_OUT):
    for fname in sorted(files):
        if fname.endswith(".py"):
            path = os.path.join(root, fname)
            print(f"\n{'='*80}")
            print(f"FILE: {path}")
            print("=" * 80)
            with open(path) as f:
                lines = f.readlines()
            print("".join(lines[:300]))
            if len(lines) > 300:
                print(f"... ({len(lines) - 300} more lines)")
