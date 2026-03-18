FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
ENV CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

WORKDIR /build/

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    vim \
    python3-pip \
    python3-venv \
    python-is-python3 \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch>=2.2.0 torchvision torchaudio && \
    python3 -m pip install -r requirements.txt

COPY . .

RUN chmod +x scripts/*.sh && \
    mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"


CMD ["bash"]
