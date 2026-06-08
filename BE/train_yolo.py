import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> int:
    parser = argparse.ArgumentParser(description="Train or fine-tune a YOLO detection model.")
    parser.add_argument("--data", default="DataSet/data.yaml", help="Path to YOLO data.yaml")
    parser.add_argument("--model", default="BE/yolov8n.pt", help="Base model, e.g. yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="train")
    args = parser.parse_args()

    data_path = Path(args.data)
    model_path = Path(args.model)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")
    if model_path.exists() and model_path.stat().st_size == 0:
        raise ValueError(f"Model file is empty and cannot be used: {model_path}")

    model = YOLO(str(model_path))
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )

    print("Training finished.")
    print(f"Best weights should be at: {args.project}/{args.name}/weights/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
