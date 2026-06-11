#!/usr/bin/env bash
# start.sh — wrapper for transcribe.py on this Linux box.
#
# What it does:
#   1. Make sure uv is available (installs it on first run if missing).
#   2. Create/sync the .venv with `uv sync` (no-op if already up to date).
#   3. Make sure the bundled cu12 libs (libcublas.so.12, libcudnn.so.9) are on
#      LD_LIBRARY_PATH so faster-whisper / ctranslate2 can find them. The
#      system has CUDA 13 but the cu12 wheel of ctranslate2 needs cu12.
#   4. Run transcribe.py with all extra args forwarded.
#
# Usage:
#   ./start.sh recording.m4a
#   ./start.sh recording.m4a --whisper-model medium --skip-cleanup
#   ./start.sh --from-raw ./output/old/raw.txt
#
# All transcribe.py flags are passed through unchanged.

set -euo pipefail

# --- locate repo root (the dir this script lives in) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- 1. ensure uv ---
if ! command -v uv >/dev/null 2>&1; then
    echo "[start.sh] uv not found — installing via the official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installs to ~/.local/bin by default; pick it up for the rest of this script
    export PATH="$HOME/.local/bin:$PATH"
fi

# --- 2. sync the venv (idempotent: fast on subsequent runs) ---
echo "[start.sh] syncing .venv with uv sync..."
uv sync --quiet

# --- 3. ensure bundled cu12 cuBLAS / cuDNN are present, then wire LD_LIBRARY_PATH ---
# faster-whisper's ctranslate2 wheel was built against CUDA 12, but the system
# has CUDA 13. pip bundles the matching cu12 libs in the nvidia-cublas-cu12 /
# nvidia-cudnn-cu12 packages — install them if missing, then point
# LD_LIBRARY_PATH at them.
#
# macOS is skipped entirely: faster-whisper uses the Accelerate framework /
# Metal on Apple Silicon, and the nvidia-* pip packages only ship Linux x86_64
# wheels — installing them on a Mac either errors or pulls a useless
# manylinux wheel.
if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "[start.sh] macOS detected — skipping cu12 libs (Apple Silicon uses Metal/Accelerate)."
else
    VENV_NVIDIA="$SCRIPT_DIR/.venv/lib/python3.12/site-packages/nvidia"
    CUBLAS_SO="$VENV_NVIDIA/cublas/lib/libcublas.so.12"
    CUDNN_SO="$VENV_NVIDIA/cudnn/lib/libcudnn.so.9"

    if [[ ! -f "$CUBLAS_SO" || ! -f "$CUDNN_SO" ]]; then
        echo "[start.sh] installing nvidia-cublas-cu12 and nvidia-cudnn-cu12 into the venv..."
        uv pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 >&2
    fi

    if [[ ! -f "$CUBLAS_SO" ]]; then
        echo "[start.sh] ERROR: $CUBLAS_SO still missing after install." >&2
        echo "          faster-whisper will not run on GPU." >&2
        exit 1
    fi
    if [[ ! -f "$CUDNN_SO" ]]; then
        echo "[start.sh] ERROR: $CUDNN_SO still missing after install." >&2
        echo "          faster-whisper will not run on GPU." >&2
        exit 1
    fi

    LIB_PATHS=()
    [[ -d "$VENV_NVIDIA/cublas/lib"    ]] && LIB_PATHS+=("$VENV_NVIDIA/cublas/lib")
    [[ -d "$VENV_NVIDIA/cudnn/lib"     ]] && LIB_PATHS+=("$VENV_NVIDIA/cudnn/lib")
    [[ -d "$VENV_NVIDIA/cuda_nvrtc/lib" ]] && LIB_PATHS+=("$VENV_NVIDIA/cuda_nvrtc/lib")
    JOINED=$(IFS=:; echo "${LIB_PATHS[*]}")
    export LD_LIBRARY_PATH="$JOINED${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[start.sh] cu12 libs on LD_LIBRARY_PATH: $JOINED"
fi

# --- 4. run transcribe.py with forwarded args ---
echo "[start.sh] running: uv run transcribe.py $*"
exec uv run transcribe.py "$@"
