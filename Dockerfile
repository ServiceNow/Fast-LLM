# syntax=docker/dockerfile:1.7-labs
FROM nvcr.io/nvidia/pytorch:24.07-py3 as base

# Install dependencies
RUN apt-get update \
    && apt-get install --no-install-recommends -y acl python3.10-venv git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Set the working directory
WORKDIR /app

# Set the default ACL for /app to rwx for all users
RUN setfacl -d -m u::rwx,g::rwx,o::rwx /app

# Environment settings for the virtual environment
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Create the virtual environment with system site packages
RUN python3 -m venv $VIRTUAL_ENV --system-site-packages

# Copy dependency files with universal write permissions for all users
COPY --chmod=777 setup.py setup.cfg pyproject.toml ./
COPY --chmod=777 ./fast_llm/csrc/ fast_llm/csrc/

# Install dependencies within the virtual environment
RUN pip install --no-cache-dir --no-build-isolation -e ".[CORE,OPTIONAL,DEV]"

# Copy remaining source code with universal write permissions
COPY --chmod=777 ./Megatron-LM Megatron-LM
COPY --chmod=777 ./examples examples
COPY --chmod=777 ./tests tests
COPY --chmod=777 ./tools tools
COPY --chmod=777 --exclude=./fast_llm/csrc/ ./fast_llm/ fast_llm/
