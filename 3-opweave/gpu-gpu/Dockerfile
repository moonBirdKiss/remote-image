# =============================================================================
# Dockerfile — GPU IPC 开发环境
#
# 纯净开发容器，不包含任何项目文件。
# 通过 bind mount 将宿主机项目目录挂载进容器使用。
#
# ── 构建镜像 ──────────────────────────────────────────────────────────────────
#
#   docker build -t gpu-ipc-dev .
#
# ── 启动开发环境 ───────────────────────────────────────────────────────────────
#
#   docker run --rm -it \
#       --gpus all \
#       --ipc=host \
#       -v "$(pwd):/workspace" \
#       gpu-ipc-dev
#
# 进入容器后：
#   cd /workspace
#   make SM=89 all          # 编译（SM=89 对应 RTX 4090）
#   python3 python/benchmark.py --pairs 1 2
#
# ── 必要的运行时参数 ───────────────────────────────────────────────────────────
#
#   --gpus all   通过 NVIDIA Container Runtime 将 GPU 透传给容器
#                （宿主机须安装 nvidia-container-toolkit）
#
#   --ipc=host   与宿主机共享 IPC 命名空间。benchmark.py 会 fork 多个子进程，
#                各子进程之间通过 CUDA IPC 共享显存；Docker 默认的 /dev/shm
#                仅 64 MB 且命名空间隔离，会导致 cudaIpcOpenMemHandle 失败。
#
# ── MPS（多对并发场景建议开启）────────────────────────────────────────────────
#
#   在宿主机上执行（容器启动前）：
#     sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
#     sudo nvidia-cuda-mps-control -d
#
#   验证：
#     echo "get_server_list" | sudo nvidia-cuda-mps-control
#
#   关闭：
#     echo "quit" | sudo nvidia-cuda-mps-control
#     sudo nvidia-smi -i 0 -c DEFAULT
#
# =============================================================================

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        # C/C++ 编译工具链
        make              \
        gcc               \
        g++               \
        # Python 运行时
        python3           \
        python3-pip       \
        # 常用调试 / 开发工具
        vim               \
        gdb               \
        strace            \
        # 进程 / 系统观测
        htop              \
        procps            \
        # 网络工具（可选）
        curl              \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

CMD ["bash"]
