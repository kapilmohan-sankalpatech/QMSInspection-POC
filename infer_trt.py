"""
Run real-time defect detection inference using a TensorRT engine.

Uses Ultralytics' YOLO class, which can load .engine files directly and
handles TensorRT execution, pre/post-processing internally.

Examples:
    # single image
    python3 infer_trt.py --engine best.engine --source part.jpg --data data/data.yaml

    # video file
    python3 infer_trt.py --engine best.engine --source line_footage.mp4 --data data/data.yaml

    # live camera (index 0)
    python3 infer_trt.py --engine best.engine --source 0 --data data/data.yaml
"""
import argparse
import time
from pathlib import Path

import cv2
import yaml
from ultralytics import YOLO


def load_class_names(data_yaml):
    if data_yaml is None:
        return None
    with open(data_yaml, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("names")


def main():
    parser = argparse.ArgumentParser(description="TensorRT defect detection inference")
    parser.add_argument("--engine", type=str, required=True, help="Path to .engine file")
    parser.add_argument("--source", type=str, required=True,
                         help="Image/video path, or camera index (e.g. 0)")
    parser.add_argument("--data", type=str, default=None, help="data.yaml (for class names)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--save", action="store_true", default=True, help="Save annotated output")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names(args.data)

    print(f"Loading TensorRT engine: {args.engine}")
    model = YOLO(args.engine, task="detect")

    # Determine if source is a live camera (int index) vs file
    is_camera = args.source.isdigit()
    source = int(args.source) if is_camera else args.source

    if is_camera or str(args.source).lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        run_stream(model, source, args, class_names, out_dir, is_camera)
    else:
        run_image(model, source, args, class_names, out_dir)


def run_image(model, source, args, class_names, out_dir):
    t0 = time.time()
    results = model.predict(source=source, conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)
    elapsed = time.time() - t0

    r = results[0]
    n_defects = len(r.boxes)
    print(f"Inference time: {elapsed*1000:.1f} ms | Defects found: {n_defects}")

    for box in r.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        label = class_names[cls_id] if class_names else str(cls_id)
        print(f"  - {label}: {conf:.2f}")

    annotated = r.plot()
    out_path = out_dir / f"annotated_{Path(str(source)).name}"
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved annotated result to: {out_path}")


def run_stream(model, source, args, class_names, out_dir, is_camera):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    writer = None
    if not is_camera:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = out_dir / "annotated_output.mp4"
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        print(f"Writing annotated video to: {out_path}")

    frame_count = 0
    t_start = time.time()

    print("Press 'q' to quit (if a display is available).")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()
        results = model.predict(source=frame, conf=args.conf, iou=args.iou,
                                 imgsz=args.imgsz, verbose=False)
        infer_ms = (time.time() - t0) * 1000
        r = results[0]
        annotated = r.plot()

        fps_display = 1000.0 / infer_ms if infer_ms > 0 else 0
        cv2.putText(annotated, f"FPS: {fps_display:.1f} | Defects: {len(r.boxes)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if writer is not None:
            writer.write(annotated)

        try:
            cv2.imshow("Defect Detection (TensorRT)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except cv2.error:
            pass  # no display available (headless server) - just keep processing

        frame_count += 1

    total_time = time.time() - t_start
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\nProcessed {frame_count} frames in {total_time:.1f}s | Avg FPS: {avg_fps:.1f}")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
