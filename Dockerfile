# syntax=docker/dockerfile:1.7-labs
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Install dependencies.
RUN apt-get update \
    && apt-get install --no-install-recommends -y acl python3.10-venv git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Set the permission to 777 for all files and directories in `/app` and `/home`:
#   1. Create directories explicitly because docker use the wrong permission for explicit creation.
#   2. For the rest, set the default ACL to 777 for all users.
RUN mkdir -m 777 /app /app/Megatron-LM /app/examples /app/fast_llm /app/tests /app/tools \
    && chmod -R 777 /home \
    && setfacl -d -m u::rwx,g::rwx,o::rwx /app /home

# Set the working directory.
WORKDIR /app

# Environment settings for the virtual environment.
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Create the virtual environment with system site packages.
RUN python3 -m venv $VIRTUAL_ENV --system-site-packages

# Copy dependency files with universal write permissions for all users.
COPY --chmod=777 setup.py setup.cfg pyproject.toml ./
COPY --chmod=777 ./fast_llm/csrc/ fast_llm/csrc/

# Install dependencies within the virtual environment.
RUN pip install --no-cache-dir --no-build-isolation -e ".[CORE,OPTIONAL,DEV]"

# Copy the remaining source code with universal write permissions.
COPY --chmod=777 ./Megatron-LM Megatron-LM
COPY --chmod=777 ./examples examples
COPY --chmod=777 ./tests tests
COPY --chmod=777 ./tools tools
COPY --chmod=777 --exclude=./fast_llm/csrc/ ./fast_llm/ fast_llm/
