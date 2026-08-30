# Glass & Metal Defect Detection — YOLOv8 + TensorRT Prototype

A prototype pipeline for detecting surface defects (scratches, cracks, bubbles,
dents, inclusions, stains, chips, etc.) on glass and metal parts using
**YOLOv8** for training and **TensorRT** for accelerated inference.

Pipeline:
```
Label images (Roboflow/LabelImg/CVAT)
        │
        ▼
Train YOLOv8 (train.py)
        │
        ▼
Export to ONNX → TensorRT engine (export_tensorrt.py)
        │
        ▼
Real-time inference with TensorRT (infer_trt.py)
```

---

## 0. Hardware / Software Requirements

- NVIDIA GPU (GTX 1660 / RTX 20-series or newer recommended)
- NVIDIA driver installed (`nvidia-smi` should work)
- CUDA 12.x + cuDNN (TensorRT install below pulls compatible wheels)
- Ubuntu 20.04/22.04 or WSL2, Python 3.9–3.11

Check your GPU/driver first:
```bash
nvidia-smi
```
If this fails, install/update your NVIDIA driver before continuing — nothing
below will work without it.

---

## 1. Environment Setup

```bash
# Create an isolated environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

Run the provided setup script (installs PyTorch w/ CUDA, Ultralytics YOLOv8,
TensorRT, ONNX tooling, and OpenCV):

```bash
bash setup.sh
```

Or install manually, step by step:

```bash
# 1. PyTorch with CUDA 12.1 support (adjust cu121 to match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Ultralytics (YOLOv8)
pip install ultralytics

# 3. ONNX export/runtime tooling
pip install onnx onnxsim onnxruntime-gpu

# 4. TensorRT (NVIDIA's Python wheels)
pip install tensorrt tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs

# 5. PyCUDA — needed to run TensorRT engines in infer_trt.py
pip install pycuda

# 6. Utilities
pip install opencv-python numpy pillow pyyaml matplotlib
```

> **Note on TensorRT install:** the `pip install tensorrt` wheels work on most
> recent setups, but if you hit version-mismatch issues, install TensorRT via
> NVIDIA's official `.deb`/`.tar` package instead (matched exactly to your CUDA
> version): https://developer.nvidia.com/tensorrt — this is the more reliable
> path on production machines / Jetson devices.

For **Jetson devices** (Orin/Xavier for edge deployment on the factory line),
TensorRT and CUDA come preinstalled with JetPack — skip the pip TensorRT
install and just do:
```bash
sudo apt install python3-libnvinfer python3-libnvinfer-dev
pip install ultralytics onnx onnxsim
```

Verify everything:
```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python3 -c "import tensorrt as trt; print('TensorRT:', trt.__version__)"
```

---

## 2. Project Structure

```
defect_detection_prototype/
├── README.md
├── setup.sh                 # one-shot dependency installer
├── requirements.txt
├── data/
│   └── data.yaml             # dataset config (edit paths + class names)
├── train.py                  # fine-tune YOLOv8 on your defect dataset
├── export_tensorrt.py        # .pt -> ONNX -> TensorRT .engine
├── infer_trt.py               # run inference with the TensorRT engine (image/video/webcam)
└── infer_quick_pt.py          # quick sanity-check inference using plain .pt model (no TensorRT)
```

---

## 3. Prepare Your Dataset

YOLOv8 expects YOLO-format labels (one `.txt` per image, normalized
`class x_center y_center width height`).

Recommended fastest path: label ~200-500 images per defect class in
[Roboflow](https://roboflow.com) (free tier) — it exports directly in
YOLOv8 format and does augmentation for you. Alternatives: `LabelImg`, `CVAT`.

Folder layout expected by `data/data.yaml`:
```
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Example defect classes for glass/metal manufacturing (edit to match your line):
```yaml
names:
  0: scratch
  1: crack
  2: bubble
  3: dent
  4: inclusion
  5: stain
  6: chip
  7: burr
```
Edit `data/data.yaml` with your actual paths and class names before training.

If you don't have a labeled dataset yet and just want to test the pipeline
end-to-end, search Roboflow Universe for public "surface defect detection"
or "metal defect" / "glass defect" datasets to get started quickly.

---

## 4. Train YOLOv8

```bash
python3 train.py --data data/data.yaml --model yolov8s.pt --epochs 100 --imgsz 640 --batch 16
```

This fine-tunes a pretrained YOLOv8 checkpoint on your defect dataset.
Outputs land in `runs/detect/train/weights/best.pt`.

Model size guide (speed vs. accuracy tradeoff):
- `yolov8n.pt` — fastest, lowest accuracy, good for edge/Jetson
- `yolov8s.pt` — good default balance (recommended starting point)
- `yolov8m.pt` / `yolov8l.pt` — higher accuracy, slower, needs a stronger GPU

---

## 5. Export to TensorRT

```bash
python3 export_tensorrt.py --weights runs/detect/train/weights/crack_seg_yolov8n.pt --imgsz 640 --half
```

This produces `best.engine` — a TensorRT engine optimized for your specific
GPU (engines are **not portable** across different GPU models; re-export on
the deployment machine).

Flags:
- `--half` → FP16 precision (2x+ speedup, minimal accuracy loss, recommended)
- `--int8` → INT8 quantization (fastest, needs a small calibration dataset — see script)
- `--workspace` → GPU memory (GB) TensorRT can use while optimizing (default 4)

---

## 6. Run Real-Time Inference

On an image:
```bash
python3 infer_trt.py --engine runs/detect/train/weights/best.engine --source path/to/image.jpg --data data/data.yaml
```

On a video file:
```bash
python3 infer_trt.py --engine best.engine --source path/to/video.mp4 --data data/data.yaml
```

On a live camera feed (e.g. inline inspection camera at index 0):
```bash
python3 infer_trt.py --engine best.engine --source 0 --data data/data.yaml
```

The script draws bounding boxes + defect class + confidence, prints FPS, and
saves annotated output to `outputs/`.

---

## 7. Quick Sanity Check (skip TensorRT)

Before bothering with TensorRT export, you can quickly confirm your trained
model works at all:
```bash
python3 infer_quick_pt.py --weights runs/detect/train/weights/crack_seg_yolov8n.pt --source path/to/image.jpg
```

---

## 8. Next Steps for a Production System

- Trigger frames from a PLC/line-sensor signal instead of polling a webcam
- Add a "reject" GPIO/relay signal when a defect above confidence threshold is found
- Log detections (image + metadata) to a DB for traceability/quality reports
- Retrain periodically as new defect types/lighting conditions appear
- Consider DeepStream (NVIDIA) instead of raw TensorRT if you need multi-camera
  pipelines at scale
