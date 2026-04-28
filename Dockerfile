# syntax=docker/dockerfile:1.7-labs
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Install dependencies.
RUN apt-get update \
    && apt-get install --no-install-recommends -y acl git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Set the working directory.
WORKDIR /app
# Make /app and /home world-rwx so the cluster's dynamically-assigned UIDs can write to them.
# Subdirs are created explicitly because Docker uses the wrong permissions for implicit creation;
# the default ACL then propagates to files copied or created under each path. Our primary cluster
# mounts /home from the outside, but other deployments may run the image as-is with the
# in-image /home, so we keep it permissive here. /usr/local/lib/python3.12 deliberately stays at
# default perms so a compromised process can't tamper with installed Python packages or system
# binaries.
RUN mkdir -m 777 /app/Megatron-LM /app/examples /app/fast_llm /app/tests /app/tools \
    && setfacl -m d:u::rwx,d:g::rwx,d:o::rwx,u::rwx,g::rwx,o::rwx /app /home

# The base image enforces versions for things like pytest for no good reason.
ENV PIP_CONSTRAINT=""
# There is no pre-build mamba image for pytorch 2.8, we build it before the rest to avoid rebuilds.
# We need to compile from the repo because of https://github.com/state-spaces/mamba/issues/720 (same for causal-conv1d)
# We set the number of workers to avoid OOM when compiling on laptop. (TODO: Can we make it configurable?)
RUN MAX_JOBS=2 pip install --no-build-isolation "causal-conv1d @ git+https://github.com/Dao-AILab/causal-conv1d@v1.5.4"
RUN MAX_JOBS=2 pip install --no-build-isolation mamba-ssm==2.2.6.post3
RUN MAX_JOBS=2 pip install --no-build-isolation "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention@67eee20c8503cd19eeb52aa1b99821308e9260c5"
# Copy dependency files with universal write permissions for all users.
COPY --chmod=777 setup.py setup.cfg pyproject.toml ./
COPY --chmod=777 ./fast_llm_external_models/__init__.py fast_llm_external_models/
COPY --chmod=777 ./fast_llm/__init__.py fast_llm/
COPY --chmod=777 ./fast_llm/csrc/ fast_llm/csrc/

# Install dependencies within the virtual environment.
RUN pip install --no-cache-dir --no-build-isolation -e ".[CORE,OPTIONAL,HUGGINGFACE,SSM,VISION,GENERATION,STREAMING,DEV]" triton==3.5.1 "transformers>=5.0.0,<6.0.0"

# Copy the remaining source code with universal write permissions.
COPY --chmod=777 ./Megatron-LM Megatron-LM
COPY --chmod=777 ./examples examples
COPY --chmod=777 ./tests tests
COPY --chmod=777 ./tools tools
COPY --chmod=777 ./fast_llm_external_models fast_llm_external_models
COPY --chmod=777 --exclude=./fast_llm/csrc/ ./fast_llm/ fast_llm/

# Set a dummy default user so we don't run in root by default.
# The image is still compatible with any user id.
RUN useradd user
USER user
