#!/usr/bin/env bash
# One-shot dependency installer for the YOLOv8 + TensorRT defect detection prototype.
# Usage: bash setup.sh [cu118|cu121|cu124]   (default: cu121)

set -e

CUDA_TAG="${1:-cu121}"

echo "=== 1. Checking NVIDIA driver ==="
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Install/verify your NVIDIA GPU driver first."
    exit 1
fi
nvidia-smi

echo "=== 2. Upgrading pip ==="
pip install --upgrade pip

echo "=== 3. Installing PyTorch (CUDA: $CUDA_TAG) ==="
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo "=== 4. Installing Ultralytics (YOLOv8) ==="
pip install ultralytics

echo "=== 5. Installing ONNX export/runtime tooling ==="
pip install onnx onnxsim onnxruntime-gpu

echo "=== 6. Installing TensorRT Python wheels ==="
pip install tensorrt tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs || \
    echo "WARNING: pip TensorRT install failed — install via NVIDIA's official package instead: https://developer.nvidia.com/tensorrt"

echo "=== 7. Installing PyCUDA (for running TensorRT engines) ==="
pip install pycuda

echo "=== 8. Installing remaining utilities ==="
pip install opencv-python numpy pillow pyyaml matplotlib

echo "=== 9. Verifying installation ==="
python3 -c "import torch; print('Torch CUDA available:', torch.cuda.is_available())"
python3 -c "from ultralytics import YOLO; print('Ultralytics OK')"
python3 -c "import tensorrt as trt; print('TensorRT version:', trt.__version__)" || \
    echo "TensorRT import failed - see note above about installing via NVIDIA's official package."

echo "=== Setup complete ==="
