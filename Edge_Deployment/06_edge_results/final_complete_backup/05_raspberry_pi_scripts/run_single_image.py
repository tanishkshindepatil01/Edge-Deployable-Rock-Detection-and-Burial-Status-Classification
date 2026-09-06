#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO


def discover_model(folder: Path) -> Path:
    files = [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in {".pt", ".onnx", ".tflite"}]
    if files:
        return sorted(files)[0]
    dirs = [path for path in folder.rglob("*") if path.is_dir() and any(path.glob("*.param")) and any(path.glob("*.bin"))]
    if dirs:
        return sorted(dirs)[0]
    raise FileNotFoundError(folder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("single_image_result"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = discover_model(args.model_folder)
    model = YOLO(str(model_path), task="detect")
    result = model.predict(str(args.image), imgsz=args.imgsz, conf=args.conf, iou=args.iou, batch=1, device="cpu", verbose=False)[0]
    counts = Counter(result.boxes.cls.cpu().numpy().astype(int).tolist() if result.boxes is not None else [])
    plotted = result.plot()
    output_image = args.output_dir / f"{args.image.stem}_annotated.jpg"
    cv2.imwrite(str(output_image), plotted)
    payload = {
        "model": str(model_path),
        "source_image": str(args.image),
        "total_rocks": int(sum(counts.values())),
        "rock_completely_exposed": int(counts.get(0, 0)),
        "rock_half_buried": int(counts.get(1, 0)),
    }
    output_json = args.output_dir / f"{args.image.stem}_result.json"
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("Annotated image:", output_image)
    print("JSON result:", output_json)


if __name__ == "__main__":
    main()
