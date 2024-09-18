FROM nvcr.io/nvidia/pytorch:24.07-py3

# git-lfs is needed to interact with the huggingface hub
RUN apt-get update \
    && apt-get install git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

ARG FAST_LLM_USER_ID=1000

RUN useradd -m -u $FAST_LLM_USER_ID -s /bin/bash -d /home/fast_llm fast_llm
USER fast_llm
WORKDIR /app
ENV PYTHONPATH=/app:/app:/app/Megatron-LM
ENV PATH=$PATH:/app:/home/fast_llm/.local/bin/

COPY --chown=fast_llm setup.py setup.cfg ./
COPY --chown=fast_llm fast_llm/__init__.py ./fast_llm/

RUN PIP_NO_INPUT=1 pip3 install --no-cache-dir -e ".[CUDA]"

COPY --chown=fast_llm fast_llm/csrc/ ./fast_llm/csrc/
RUN make -C ./fast_llm/csrc/

COPY --chown=fast_llm ./Megatron-LM Megatron-LM
COPY --chown=fast_llm ./examples examples
COPY --chown=fast_llm ./tests tests
COPY --chown=fast_llm ./tools tools
COPY --chown=fast_llm ./fast_llm fast_llm
COPY --chown=fast_llm fast_llm/tools/train.py pyproject.toml ./
