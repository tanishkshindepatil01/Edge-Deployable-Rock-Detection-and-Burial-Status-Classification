#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-$HOME/rock_edge_project}"
VENV_DIR="${2:-$HOME/rock_edge_venv}"

echo "Project root: $PROJECT_ROOT"
echo "Virtual environment: $VENV_DIR"

sudo apt update
sudo apt install -y python3-full python3-venv python3-pip libgl1 libglib2.0-0 git cmake build-essential

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r "$PROJECT_ROOT/05_raspberry_pi_scripts/requirements_raspberry_pi.txt"

# NCNN is required only for NCNN inference. Prefer the exact version tested in Colab,
# then fall back to the latest compatible wheel available for this Raspberry Pi OS.
python -m pip install --no-deps "ncnn==1.0.20260526" || python -m pip install --no-deps ncnn

# LiteRT Python wheels vary by Raspberry Pi OS/Python version. Try the modern package,
# then the lightweight runtime. The verification script reports what is available.
python -m pip install ai-edge-litert || python -m pip install tflite-runtime || true

python "$PROJECT_ROOT/05_raspberry_pi_scripts/verify_environment.py" --project-root "$PROJECT_ROOT"
echo "Setup completed. Activate with: source $VENV_DIR/bin/activate"
