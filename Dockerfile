# syntax=docker/dockerfile:1.7-labs
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Install git-lfs for Huggingface hub interaction and sudo for system adjustments
RUN apt-get update \
    && apt-get install --no-install-recommends -y git-lfs sudo util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Add a user for Fast-LLM with sudo privileges for runtime adjustments
ARG FAST_LLM_USER_ID=1000
RUN useradd -m -u $FAST_LLM_USER_ID -s /bin/bash fast_llm \
    && echo 'fast_llm ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers

USER fast_llm
WORKDIR /app

# Environment settings for Python and PATH
ENV PYTHONPATH=/app:/app/Megatron-LM \
    PATH=$PATH:/home/fast_llm/.local/bin/

# Copy the dependency files and install dependencies
COPY --chown=fast_llm setup.py setup.cfg pyproject.toml ./
COPY --chown=fast_llm ./fast_llm/csrc/ fast_llm/csrc/
RUN PIP_NO_INPUT=1 pip3 install --no-cache-dir --no-build-isolation -e ".[CORE,OPTIONAL,DEV]"

# Copy the rest of the code
COPY --chown=fast_llm ./Megatron-LM Megatron-LM
COPY --chown=fast_llm ./examples examples
COPY --chown=fast_llm ./tests tests
COPY --chown=fast_llm ./tools tools

# Copy the main source code for Fast-LLM
COPY --exclude=./fast_llm/csrc/ --chown=fast_llm ./fast_llm/ fast_llm/
