import base64
import os
from pathlib import Path

import cv2
import numpy as np
import runpod
from ultralytics import YOLO

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/best.pt")
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.35"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Load once when the worker starts; do not load the model per request.
model = YOLO(MODEL_PATH, task="detect")


def decode_image(image_base64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(image_base64)
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as exc:
        raise ValueError("Invalid base64 image") from exc

    if image is None:
        raise ValueError("Could not decode image")
    return image


def handler(job):
    job_input = job.get("input", {}) or {}
    image_base64 = job_input.get("image_base64")

    if not image_base64:
        return {"status": "error", "error": "input.image_base64 is required"}

    try:
        image = decode_image(image_base64)
        confidence = float(job_input.get("confidence", CONFIDENCE))
        imgsz = int(job_input.get("imgsz", IMAGE_SIZE))

        results = model.predict(
            source=image,
            conf=confidence,
            imgsz=imgsz,
            verbose=False,
        )
        result = results[0]

        defects = []
        for box in result.boxes:
            cls_id = int(box.cls.item())
            defects.append({
                "class_id": cls_id,
                "class_name": result.names.get(cls_id, str(cls_id)),
                "confidence": float(box.conf.item()),
                "box": [float(v) for v in box.xyxy[0].tolist()],
            })

        return {
            "status": "success",
            "defect_count": len(defects),
            "defects": defects,
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


runpod.serverless.start({"handler": handler})
