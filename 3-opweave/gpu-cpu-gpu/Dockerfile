FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
    CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

WORKDIR /workspace

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    vim \
    less \
    make \
    build-essential \
    pkg-config \
    cmake \
    python3 \
    python3-pip \
    python3-venv && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip && \
    python -m pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple \
    "torch>=2.2.0" \
    "numpy>=1.26.0,<2.0.0" \
    "nvidia-ml-py==12.560.30"

RUN mkdir -p /tmp/gpu_uds_benchmark /tmp/nvidia-mps /tmp/nvidia-log

# Clean development container: no project files copied at build time.
CMD ["bash"]
