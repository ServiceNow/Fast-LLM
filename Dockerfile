# syntax=docker/dockerfile:1.7-labs
FROM nvcr.io/nvidia/pytorch:24.07-py3 as base

# Install dependencies
RUN apt-get update \
    && apt-get install --no-install-recommends -y acl python3.10-venv git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Set the working directory
WORKDIR /app

# Set the setgid bit and default ACL for /app
RUN chmod g+s /app && \
    setfacl -d -m u::rwx,g::rwx,o::rwx /app && \
    setfacl -d -m u::rw-,g::rw-,o::rw- /app

# Environment settings for the virtual environment
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Create the virtual environment with system site packages
RUN python3 -m venv $VIRTUAL_ENV --system-site-packages

# Copy dependency files with universal write permissions for all users
COPY --chmod=666 setup.py setup.cfg pyproject.toml ./
COPY --chmod=666 ./fast_llm/csrc/ fast_llm/csrc/

# Install dependencies within the virtual environment
RUN pip install --no-cache-dir --no-build-isolation -e ".[CORE,OPTIONAL,DEV]"

# Use intermediate build stage to copy the remaining source code
FROM alpine as copy_source

# Set the working directory
WORKDIR /app

# Copy remaining source code with universal write permissions
COPY ./Megatron-LM Megatron-LM
COPY ./examples examples
COPY ./tests tests
COPY ./tools tools
COPY --exclude=./fast_llm/csrc/ ./fast_llm/ fast_llm/

RUN find Megatron-LM -type f -exec chmod 666 {} \; && \
    find examples -type f -exec chmod 666 {} \; && \
    find tests -type f -exec chmod 666 {} \; && \
    find tools -type f -exec chmod 666 {} \; && \
    find fast_llm -type f -exec chmod 666 {} \; && \
    find . -type d -exec chmod 777 {} \;

# Create a tar archive of /app with permissions preserved
RUN tar -cf /app.tar -C /app .

# Continue with the base stage
FROM base

# Copy the remaining source code from the intermediate build stage
COPY --from=copy_source /app.tar /
RUN tar -xf /app.tar -C /app && rm /app.tar
