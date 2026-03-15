FROM python:3.11-slim

# Install system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    make \
    git \
    curl \
    wget \
    bash \
    procps \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies inline (no file copy into image)
# PyTorch is installed with CUDA 12.1 support to match RTX 4090 requirements
RUN pip install --no-cache-dir \
        numpy>=1.24.0 \
        torch>=2.1.0 \
        torchvision>=0.16.0 \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir \
        transformers>=4.40.0 \
        huggingface-hub>=0.23.0 \
        tqdm>=4.66.0 \
        sentencepiece>=0.1.99 \
        safetensors>=0.4.0

WORKDIR /workspace

CMD ["bash"]
