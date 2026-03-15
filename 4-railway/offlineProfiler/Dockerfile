# 使用官方 Python 镜像（ML 推理场景中常见的稳定版本）
FROM python:3.10-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 使用 bash + pipefail，提升 RUN 命令健壮性
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# 安装常用开发工具（不安装多余推荐包以减小镜像体积）
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    bash \
    vim \
    make \
    git \
    curl \
    ca-certificates \
    tzdata \
    less \
    procps \
    iproute2 \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# 配置 pip 镜像源加速（可按需改为你自己的镜像）
RUN mkdir -p /etc \
    && cat > /etc/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
timeout = 120
EOF

# 升级基础 Python 打包工具
RUN python -m pip install --upgrade pip setuptools wheel

# 直接在镜像内安装项目依赖（不通过 COPY requirements.txt）
# 说明：这里使用 CUDA 12.1 的 PyTorch 官方源，便于在 4090 环境使用 GPU 推理
RUN python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    "numpy>=1.24.0" \
    "torch>=2.1.0" \
    "torchvision>=0.16.0" \
    "transformers>=4.40.0" \
    "huggingface-hub>=0.23.0" \
    "tqdm>=4.66.0" \
    "sentencepiece>=0.1.99" \
    "safetensors>=0.4.0"

ARG USERNAME=ubuntu
ARG UID=1000
ARG GID=1000

# 创建低权限 ubuntu 用户用于开发，避免长期使用 root
RUN groupadd --gid "${GID}" "${USERNAME}" \
    && useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash "${USERNAME}"

# 约定工作目录；项目代码通过 bind mount 挂载，不在镜像里 COPY
WORKDIR /workspace
RUN chown -R "${USERNAME}:${USERNAME}" /workspace

USER ${USERNAME}

# 默认进入 bash，便于交互开发
CMD ["/bin/bash"]
