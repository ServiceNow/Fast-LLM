# syntax=docker/dockerfile:1.7-labs
FROM nvcr.io/nvidia/pytorch:25.05-py3

# Install dependencies.
RUN apt-get update \
    && apt-get install --no-install-recommends -y acl git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Set the working directory.
WORKDIR /app
# Set the permission to 777 for all files and directories in `/app`, `/home` and python install directories:
#   1. Create directories explicitly because docker use the wrong permission for explicit creation.
#   2. For the rest, set the default ACL to 777 for all users.
RUN mkdir -m 777 /app/Megatron-LM /app/examples /app/fast_llm /app/tests /app/tools \
    && setfacl -m d:u::rwx,d:g::rwx,d:o::rwx,u::rwx,g::rwx,o::rwx \
      /app \
      /home \
      /usr \
      /usr/local \
      /usr/local/bin \
      /usr/local/lib \
      /usr/local/lib/python3.12 \
      /usr/local/lib/python3.12/dist-packages \
      /usr/local/lib/python3.12/dist-packages/__pycache__

# The base image enforces versions for things like pytest for no good reason.
ENV PIP_CONSTRAINT=""
# There is no pre-build mamba image for pytorch 2.8, we build it before the rest to avoid rebuilds.
# We need to compile from the repo because of https://github.com/state-spaces/mamba/issues/720 (same for causal-conv1d)
# We set the number of workers to avoid OOM when compiling on laptop. (TODO: Can we make it configurable?)
RUN MAX_JOBS=4 pip install --no-build-isolation  "causal-conv1d@git+https://github.com/Dao-AILab/causal-conv1d@2a288a1"
RUN MAX_JOBS=4 pip install --no-build-isolation "mamba_ssm[causal-conv1d]@git+https://github.com/state-spaces/mamba@74729d0"
# Copy dependency files with universal write permissions for all users.
COPY --chmod=777 setup.py setup.cfg pyproject.toml ./
COPY --chmod=777 ./fast_llm/__init__.py fast_llm/
COPY --chmod=777 ./fast_llm/csrc/ fast_llm/csrc/

# Install dependencies within the virtual environment.
RUN pip install --no-cache-dir --no-build-isolation -e ".[CORE,OPTIONAL,HUGGINGFACE,SSM,DEV]"

# Copy the remaining source code with universal write permissions.
COPY --chmod=777 ./Megatron-LM Megatron-LM
COPY --chmod=777 ./examples examples
COPY --chmod=777 ./tests tests
COPY --chmod=777 ./tools tools
COPY --chmod=777 --exclude=./fast_llm/csrc/ ./fast_llm/ fast_llm/
