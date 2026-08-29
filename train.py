"""
Fine-tune YOLOv8 on a glass/metal defect detection dataset.

Example:
    python3 train.py --data data/data.yaml --model yolov8s.pt --epochs 100 --imgsz 640 --batch 16
"""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for defect detection")
    parser.add_argument("--data", type=str, default="data/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                         help="Pretrained checkpoint to fine-tune (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0", help="GPU id, or 'cpu'")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="train")
    args = parser.parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        # Defect-detection-friendly augmentation tweaks:
        # small/subtle defects benefit from less aggressive geometric distortion
        degrees=5.0,
        shear=2.0,
        perspective=0.0,
        mosaic=1.0,
        mixup=0.0,
        hsv_h=0.01,   # keep hue mostly stable — defect color often matters
        hsv_s=0.5,
        hsv_v=0.4,
    )

    # Validate on the val split and print metrics
    metrics = model.val()
    print(f"\nmAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"\nBest weights saved to: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
