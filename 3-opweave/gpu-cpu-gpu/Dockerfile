FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps \
    CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ca-certificates \
    bash && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Install CUDA-enabled PyTorch wheels (cu121) and remaining Python deps.
RUN python -m pip install --upgrade pip && \
    python -m pip install \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY README.md ./README.md
COPY readme_zh.md ./readme_zh.md

RUN chmod +x /app/scripts/mps_control.sh && \
    mkdir -p /app/logs /tmp/gpu_uds_benchmark /tmp/nvidia-mps /tmp/nvidia-log

CMD ["python", "src/orchestrator.py"]
