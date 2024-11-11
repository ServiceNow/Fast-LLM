# syntax=docker/dockerfile:1.7-labs
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Install git-lfs for Huggingface hub interaction and sudo for system adjustments
RUN apt-get update \
    && apt-get install --no-install-recommends -y git-lfs sudo util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Set the working directory
WORKDIR /app

# Environment settings for Python and the user
ENV PYTHONPATH=/app:/app/Megatron-LM

# Copy the dependency files and install dependencies globally
COPY setup.py setup.cfg pyproject.toml ./
COPY ./fast_llm/csrc/ fast_llm/csrc/
RUN PIP_NO_INPUT=1 pip3 install --no-cache-dir --no-build-isolation -e ".[CORE,OPTIONAL,DEV]"

# Copy the rest of the code
COPY ./Megatron-LM Megatron-LM
COPY ./examples examples
COPY ./tests tests
COPY ./tools tools

# Copy the main source code
COPY --exclude=./fast_llm/csrc/ ./fast_llm/ fast_llm/

# Ensure the source code files are writable
RUN chmod -R a+w /app
