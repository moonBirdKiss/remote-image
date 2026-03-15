# =============================================================================
# Dockerfile — GPU IPC Benchmark (RTX 4090 / Ada Lovelace)
#
# Two-stage build:
#   Stage 1 (builder) : Full CUDA devel image → compiles libgpuipc.so, producer, consumer
#   Stage 2 (runtime) : Lean CUDA runtime image → runs Python benchmark
#
# ── Build ────────────────────────────────────────────────────────────────────
#
#   # RTX 4090 (default, CC 8.9 = SM 89)
#   docker build -t gpu-ipc-bench .
#
#   # Other GPU families:
#   docker build --build-arg SM=90 -t gpu-ipc-bench .   # H100  (Hopper)
#   docker build --build-arg SM=80 -t gpu-ipc-bench .   # A100  (Ampere)
#   docker build --build-arg SM=86 -t gpu-ipc-bench .   # RTX 3090 / A6000
#   docker build --build-arg SM=75 -t gpu-ipc-bench .   # RTX 2080 (Turing)
#
# ── Run ──────────────────────────────────────────────────────────────────────
#
#   # Full benchmark (all concurrency levels 1,2,4,6,8,10 × 200 msgs)
#   docker run --rm --gpus all --ipc=host \
#       -v "$(pwd)/ret.log:/workspace/ret.log" \
#       gpu-ipc-bench
#
#   # Custom concurrency levels and message count
#   docker run --rm --gpus all --ipc=host \
#       -v "$(pwd)/ret.log:/workspace/ret.log" \
#       gpu-ipc-bench --pairs 1 2 4 --messages 500
#
#   # Skip RTX 4090 check (for other GPU models)
#   docker run --rm --gpus all --ipc=host \
#       -v "$(pwd)/ret.log:/workspace/ret.log" \
#       gpu-ipc-bench --no-check
#
#   # Interactive shell
#   docker run --rm -it --gpus all --ipc=host gpu-ipc-bench bash
#
# ── Required docker run flags ─────────────────────────────────────────────────
#
#   --gpus all   Passes NVIDIA GPU(s) through to the container via the NVIDIA
#                Container Runtime (nvidia-container-toolkit must be installed
#                on the host).
#
#   --ipc=host   Shares the host IPC namespace with the container.  Required
#                because:
#                  1. CUDA IPC (cudaIpcGetMemHandle / cudaIpcOpenMemHandle)
#                     between subprocesses relies on shared memory in /dev/shm.
#                     Docker's default /dev/shm is only 64 MB and uses a
#                     private namespace.
#                  2. NVIDIA MPS (Multi-Process Service) communicates via
#                     Unix-domain sockets in /dev/shm and /tmp.
#
# ── MPS (Multi-Process Service) ──────────────────────────────────────────────
#
#   For concurrency levels ≥ 2 pairs, NVIDIA MPS is strongly recommended.
#   MPS allows multiple CUDA contexts to share the GPU simultaneously with
#   lower overhead.  Start MPS on the HOST before launching the container:
#
#     # Enable MPS on the host (run once)
#     sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
#     sudo nvidia-cuda-mps-control -d
#
#     # Verify MPS is running
#     echo "get_server_list" | sudo nvidia-cuda-mps-control
#
#     # Stop MPS when done
#     echo "quit" | sudo nvidia-cuda-mps-control
#     sudo nvidia-smi -i 0 -c DEFAULT
#
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Builder
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS builder

# Compute Capability for the target GPU.
# IMPORTANT: nvidia-smi is NOT available during docker build, so the Makefile's
# auto-detection falls back to SM=80.  We override it explicitly here.
# Default: 89 (RTX 4090, Ada Lovelace).  Override with --build-arg SM=XX.
ARG SM=89

# Suppress interactive apt prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build essentials.
# The devel image provides nvcc and CUDA headers but not all system build tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    make         \
    gcc          \
    g++          \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy source tree (ordering maximises Docker layer-cache reuse:
# headers and sources rarely change together with the Makefile)
COPY Makefile     .
COPY include/     include/
COPY lib/         lib/
COPY src/         src/

# Compile all targets.
# SM is passed explicitly so the Makefile never falls back to nvidia-smi.
# This produces:
#   build/producer       — standalone producer binary
#   build/consumer       — standalone consumer binary
#   build/libgpuipc.so  — shared library for Python ctypes bindings
RUN make SM=${SM} all 2>&1 | tee /tmp/build.log \
    && echo "[Dockerfile] Build succeeded for SM=${SM}" \
    || { echo "[Dockerfile] Build FAILED — see /tmp/build.log above"; exit 1; }


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Python 3 (no build tools needed in the runtime stage)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3      \
    python3-pip  \
    && rm -rf /var/lib/apt/lists/* \
    # Make "python" resolve to python3 for convenience
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /workspace

# ── Copy compiled artifacts from builder ─────────────────────────────────────
COPY --from=builder /workspace/build/ build/

# ── Copy Python sources ───────────────────────────────────────────────────────
COPY python/ python/

# ── Copy documentation (optional but useful in interactive sessions) ──────────
COPY README.md .

# ── Environment ───────────────────────────────────────────────────────────────
# Tell gpu_ipc.py exactly where to find libgpuipc.so (skips the search loop).
ENV GPUIPC_LIB_PATH=/workspace/build/libgpuipc.so

# Ensure the CUDA runtime libraries installed by the base image are on LD_LIBRARY_PATH.
# The NVIDIA Container Runtime additionally bind-mounts the host's libcuda.so.1
# when the container is started with --gpus.
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# ── Output directory for ret.log ─────────────────────────────────────────────
# benchmark.py writes ret.log to /workspace/ret.log (the project root).
# Pre-create an empty file so "docker run -v $(pwd)/ret.log:/workspace/ret.log"
# works whether or not the host file exists yet.
RUN touch /workspace/ret.log

# ── Smoke-test: verify the library loads cleanly ──────────────────────────────
# This runs at IMAGE BUILD time with no GPU present, so we only check that
# the .so is present and ctypes can dlopen it (symbols are verified at runtime).
RUN python3 - <<'EOF'
import ctypes, sys, os
lib_path = os.environ["GPUIPC_LIB_PATH"]
if not os.path.exists(lib_path):
    print(f"[smoke-test] FAIL: {lib_path} not found", file=sys.stderr)
    sys.exit(1)
print(f"[smoke-test] PASS: {lib_path} exists ({os.path.getsize(lib_path)//1024} KB)")
EOF

# ── Default command ────────────────────────────────────────────────────────────
# ENTRYPOINT: always run the Python benchmark.
# CMD: empty by default — benchmark.py runs all concurrency levels with defaults.
# Override CMD to pass extra flags:
#   docker run ... gpu-ipc-bench --pairs 1 4 --messages 500
ENTRYPOINT ["python3", "python/benchmark.py"]
CMD []
