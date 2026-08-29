"""
Quick sanity check: run inference with the raw .pt model (no TensorRT export
needed). Use this first to confirm training worked, before bothering with the
TensorRT export step.

Example:
    python3 infer_quick_pt.py --weights runs/detect/train/weights/best.pt --source part.jpg
"""
import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    model = YOLO(args.weights)
    results = model.predict(source=args.source, conf=args.conf, save=True, project=args.out_dir)

    r = results[0]
    print(f"Defects found: {len(r.boxes)}")
    for box in r.boxes:
        print(f"  - class {int(box.cls.item())}: conf {float(box.conf.item()):.2f}")
    print(f"Annotated result saved under: {args.out_dir}/")


if __name__ == "__main__":
    main()
