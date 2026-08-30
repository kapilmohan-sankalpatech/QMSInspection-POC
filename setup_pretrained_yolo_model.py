import os
import urllib.request
from ultralytics import YOLO

# Define the filename and official Hugging Face URL for the weights
weight_filename = "crack_seg_yolov8n.pt"
pretrained_url = "https://huggingface.co/OpenSistemas/YOLOv8-crack-seg/resolve/main/yolov8n/weights/best.pt"

# Download the model weights if they do not exist locally
if not os.path.exists(weight_filename):
    print("Downloading pretrained crack segmentation model...")
    urllib.request.urlretrieve(pretrained_url, weight_filename)
    print(f"Model saved as '{weight_filename}'.")
else:
    print(f"'{weight_filename}' already exists.")

# Load the downloaded model
model = YOLO(weight_filename)