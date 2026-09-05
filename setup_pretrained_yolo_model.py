import os
import argparse
import urllib.request
from ultralytics import YOLO

# ============================================================
# HOW TO RUN
# ============================================================
#
# From /workspace/QMSInspection-POC:
#
# Basic:
#   python3 infer.py --source /workspace/QMSInspection-POC/input
#
# With custom confidence:
#   python3 infer.py --source /workspace/QMSInspection-POC/input --conf 0.5
#
# Input can be:
#   - A folder containing images
#   - A single image
#   - A video file
#
# Output:
#   /workspace/QMSInspection-POC/outputs/
#
# Example:
#   python3 infer.py --source /workspace/QMSInspection-POC/input
#
# ============================================================


# Command-line arguments
parser = argparse.ArgumentParser(
    description="YOLO crack segmentation inference"
)

parser.add_argument(
    "--source",
    required=True,
    help="Input image, folder, or video path"
)

parser.add_argument(
    "--conf",
    type=float,
    default=0.25,
    help="Confidence threshold (default: 0.25)"
)

parser.add_argument(
    "--out-dir",
    default="/workspace/QMSInspection-POC/outputs",
    help="Output directory"
)

args = parser.parse_args()


# ============================================================
# Model configuration
# ============================================================

weight_filename = "crack_seg_yolov8n.pt"

pretrained_url = "https://huggingface.co/OpenSistemas/YOLOv8-crack-seg/resolve/main/yolov8n/weights/best.pt"


# ============================================================
# Download model weights if not present
# ============================================================

if not os.path.exists(weight_filename):
    print("Downloading pretrained crack segmentation model...")

    urllib.request.urlretrieve(
        pretrained_url,
        weight_filename
    )

    print(f"Model saved as '{weight_filename}'.")
else:
    print(f"'{weight_filename}' already exists.")


# ============================================================
# Load model
# ============================================================

print("Loading model...")

model = YOLO(weight_filename)


# ============================================================
# Run inference
# ============================================================

print(f"Running inference on: {args.source}")
print(f"Confidence threshold: {args.conf}")
print(f"Output directory: {args.out_dir}")

model.predict(
    source=args.source,
    conf=args.conf,
    save=True,
    project=args.out_dir
)


# ============================================================
# Completed
# ============================================================

print("Inference completed.")
print(f"Results saved to: {args.out_dir}")
