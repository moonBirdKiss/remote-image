FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda

# ── 系统依赖 ──────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3.10-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

# ── PyTorch + 核心数值库（用户指定源） ────────────────────────────────────────
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple \
    "torch>=2.2.0" \
    "torchvision>=0.17.0" \
    "numpy>=1.26.0,<2.0.0" \
    "nvidia-ml-py==12.560.30"

# ── 其余 Python 依赖 ─────────────────────────────────────────────────────────
RUN python -m pip install --no-cache-dir \
    "Pillow>=10.0.0" \
    "fastapi>=0.109.0" \
    "uvicorn[standard]>=0.27.0" \
    "numba>=0.58.0" \
    "accelerate>=0.26.0"

# ── 复制项目代码 ──────────────────────────────────────────────────────────────
WORKDIR /app
COPY . .

RUN mkdir -p logs results

EXPOSE 8000

CMD ["bash"]
