#!/usr/bin/env python3
import argparse
import math
import os
import subprocess
import time
from pathlib import Path

import pandas as pd
import psutil
import matplotlib.pyplot as plt
from ultralytics import YOLO


def discover_model(folder: Path) -> Path:
    files = [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in {".pt", ".onnx", ".tflite"}]
    if files:
        return sorted(files)[0]
    dirs = [path for path in folder.rglob("*") if path.is_dir() and any(path.glob("*.param")) and any(path.glob("*.bin"))]
    if dirs:
        return sorted(dirs)[0]
    raise FileNotFoundError(folder)


def temperature_c():
    try:
        return float(Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()) / 1000
    except Exception:
        return math.nan


def throttled():
    try:
        return subprocess.check_output(["vcgencmd", "get_throttled"], text=True).strip()
    except Exception:
        return "unavailable"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-folder", required=True, help="Example: 05_ncnn_fp32")
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    root = args.project_root.resolve()
    model_path = discover_model(root / "02_selected_model" / args.model_folder)
    images = sorted((root / "03_edge_test_dataset" / "02_preprocessed_640x640_90" / "images").glob("*.jpg"))
    if not images:
        raise FileNotFoundError("No 640x640 test images found")
    model = YOLO(str(model_path), task="detect")
    process = psutil.Process(os.getpid())
    deadline = time.monotonic() + args.minutes * 60
    rows = []
    index = 0
    while time.monotonic() < deadline:
        image = images[index % len(images)]
        start = time.perf_counter()
        model.predict(str(image), imgsz=args.imgsz, conf=args.conf, batch=1, device="cpu", verbose=False)
        latency_ms = (time.perf_counter() - start) * 1000
        rows.append({
            "iteration": index + 1,
            "elapsed_minutes": (args.minutes * 60 - max(0, deadline - time.monotonic())) / 60,
            "image": image.name,
            "latency_ms": latency_ms,
            "fps": 1000 / latency_ms if latency_ms > 0 else math.nan,
            "cpu_percent": psutil.cpu_percent(interval=None),
            "rss_mb": process.memory_info().rss / (1024 ** 2),
            "temperature_c": temperature_c(),
            "throttling": throttled(),
        })
        index += 1

    output = root / "06_edge_results" / args.model_folder / "stability_test_30_minutes.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    stability_df = pd.DataFrame(rows)
    stability_df.to_csv(output, index=False)

    stability_visual_dir = (
        root / "visualization" / "08_raspberry_pi_runtime"
        / args.model_folder / "stability_test"
    )
    stability_visual_dir.mkdir(parents=True, exist_ok=True)

    stability_plots = [
        (
            "latency_ms",
            "Latency (ms/image)",
            "RP08_stability_latency_over_time.png",
            "Stability test: latency over time",
        ),
        (
            "temperature_c",
            "Temperature (°C)",
            "RP09_stability_temperature_over_time.png",
            "Stability test: temperature over time",
        ),
        (
            "rss_mb",
            "Process RAM (MB)",
            "RP10_stability_ram_over_time.png",
            "Stability test: RAM over time",
        ),
    ]
    for metric, ylabel, filename, title in stability_plots:
        if metric not in stability_df.columns or stability_df[metric].dropna().empty:
            continue
        plt.figure(figsize=(10, 5))
        plt.plot(stability_df["elapsed_minutes"], stability_df[metric])
        plt.xlabel("Elapsed time (minutes)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(stability_visual_dir / filename, dpi=180, bbox_inches="tight")
        plt.close()

    print("Saved:", output)
    print("Stability visualizations:", stability_visual_dir)
    print(stability_df.tail().to_string(index=False))


if __name__ == "__main__":
    main()
