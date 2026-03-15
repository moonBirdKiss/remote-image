FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
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

COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

COPY . /workspace

RUN chmod +x /workspace/scripts/mps_control.sh && \
    mkdir -p /workspace/logs /tmp/gpu_uds_benchmark /tmp/nvidia-mps /tmp/nvidia-log

# Development container: no project build/compile at image build time.
CMD ["bash"]
