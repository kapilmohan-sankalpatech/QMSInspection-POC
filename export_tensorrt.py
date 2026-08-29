"""
Export a trained YOLOv8 .pt model to a TensorRT .engine file.

Uses Ultralytics' built-in exporter (handles .pt -> ONNX -> TensorRT internally).

Example:
    python3 export_tensorrt.py --weights runs/detect/train/weights/best.pt --imgsz 640 --half
    python3 export_tensorrt.py --weights best.pt --imgsz 640 --int8 --data data/data.yaml
"""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to TensorRT")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained .pt weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="Export in FP16 (recommended)")
    parser.add_argument("--int8", action="store_true",
                         help="Export in INT8 (fastest, requires --data for calibration)")
    parser.add_argument("--data", type=str, default=None,
                         help="data.yaml, required for INT8 calibration")
    parser.add_argument("--workspace", type=float, default=4.0, help="TensorRT workspace size in GB")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic input shapes")
    args = parser.parse_args()

    if args.int8 and args.data is None:
        raise ValueError("--int8 export requires --data pointing to your data.yaml for calibration images")

    model = YOLO(args.weights)

    export_kwargs = dict(
        format="engine",       # TensorRT
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        workspace=args.workspace,
        batch=args.batch,
        dynamic=args.dynamic,
        device=0,
    )
    if args.int8:
        export_kwargs["data"] = args.data

    engine_path = model.export(**export_kwargs)
    print(f"\nTensorRT engine saved to: {engine_path}")
    print("Note: this engine is optimized for THIS GPU model only — re-export if you deploy to a different GPU.")


if __name__ == "__main__":
    main()
