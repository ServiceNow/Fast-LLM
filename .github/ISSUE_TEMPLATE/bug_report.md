---
name: Bug report
about: Create a report to help us improve
title: "[bug] Brief description of the issue"
labels: bug
assignees: jlamypoirier

---

# 🐞 Describe the Bug

Provide a clear and concise description of the bug.

# 🔄 Steps to Reproduce

Steps to reproduce the behavior:

1. **Get the relevant Fast-LLM version** (e.g., git commit hash or Docker image tag) that you encountered the issue with.
2. **Run the following command** (modify or redact as needed):

    ```bash
    torchrun --rdzv_backend=static \
             --rdzv_endpoint=[insert DNS name]:[insert port] \
             --node_rank=[insert node rank] \
             --nproc_per_node=[insert number of processes per node] \
             --nnodes=[insert number of nodes] \
             --max_restarts=0 \
             --rdzv_conf=timeout=3600 \
             --no_python \
             fast-llm train gpt \
             --config /path/to/your/config.yaml
    ```

3. **Include relevant log excerpts** to help us diagnose the issue, with `NCCL_DEBUG=INFO` (or higher) enabled. Make sure the logs contain the full configuration of the run.
4. **Provide the configuration YAML** used for the Fast-LLM setup if logs are unavailable.

# 🎯 Expected Behavior

Describe what you expected to happen.

# 📜 Environment Information

Run the following script in your environment and paste its output here:

```bash
#!/bin/bash

# Script to gather environment details for Fast-LLM bug reporting

# Hardware and system information
echo "=== SYSTEM INFORMATION ==="
echo "Operating System:"
uname -a

echo "CPU Information:"
lscpu

echo "Memory Information:"
free -h

# GPU-related information
echo "=== GPU INFORMATION ==="
echo "NVIDIA System Management Interface (nvidia-smi):"
nvidia-smi

echo "CUDA Version:"
nvcc --version

# Torch and CUDA details
echo "=== PYTHON AND TORCH INFORMATION ==="
echo "Python Version:"
python --version

echo "Torch Version:"
python -c "import torch; print(torch.__version__)"

echo "CUDA Available (in Torch):"
python -c "import torch; print(torch.cuda.is_available())"

echo "CUDA Version (in Torch):"
python -c "import torch; print(torch.version.cuda)"

echo "NCCL Version (in Torch):"
python -c "import torch; print(torch.cuda.nccl.version())"

# Check for Flash Attention version, if installed
echo "=== FLASH ATTENTION INFORMATION ==="
if python -c "import flash_attn" &> /dev/null; then
    echo "Flash Attention Version:"
    python -c "import flash_attn; print(flash_attn.__version__)"
else
    echo "Flash Attention is not installed."
fi

# Check for APEX version, if installed
echo "=== APEX INFORMATION ==="
if python -c "import apex" &> /dev/null; then
    echo "APEX Version:"
    python -c "import apex; print(apex.__version__ if hasattr(apex, '__version__') else 'APEX version not specified')"
else
    echo "APEX is not installed."
fi

# End of script
echo "=== END OF ENVIRONMENT INFORMATION ==="
```

# 📝 Additional Context

Include any other information that may help us understand the issue, such as:
- Recent changes to the configuration or code.
- Whether the issue occurs consistently or intermittently.
- Any troubleshooting steps you have already tried.
