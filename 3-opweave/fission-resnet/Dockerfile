FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # MPS defaults (overridable at runtime via --env)
    CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
    CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

# System dependencies: Python 3.11 + minimal tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && python -m pip install --no-cache-dir --upgrade pip setuptools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# PyTorch + dependencies (CUDA 12.1 wheels)
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple \
    "torch>=2.2.0" \
    "torchvision>=0.17.0" \
    "numpy>=1.26.0,<2.0.0" \
    "nvidia-ml-py==12.560.30"


# MPS pipe/log directories
RUN mkdir -p /tmp/nvidia-mps /tmp/nvidia-log

ENTRYPOINT ["bash"]
