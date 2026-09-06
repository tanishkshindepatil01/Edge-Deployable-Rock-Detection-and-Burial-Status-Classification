#!/usr/bin/env python3
import argparse
import gc
import json
import math
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import psutil
import yaml
import matplotlib.pyplot as plt
from ultralytics import YOLO

FORMAT_FOLDERS = [
    ("01_pytorch_fp32", "PyTorch FP32"),
    ("02_onnx_fp32", "ONNX FP32"),
    ("03_tflite_fp32", "LiteRT/TFLite FP32"),
    ("04_tflite_int8", "LiteRT/TFLite INT8"),
    ("05_ncnn_fp32", "NCNN FP32"),
    ("06_ncnn_fp16", "NCNN FP16"),
]


def discover_model(folder: Path) -> Path:
    files = [
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pt", ".onnx", ".tflite"}
    ]
    if files:
        files.sort()
        return files[0]
    ncnn_dirs = [
        path for path in folder.rglob("*")
        if path.is_dir() and any(path.glob("*.param")) and any(path.glob("*.bin"))
    ]
    if ncnn_dirs:
        ncnn_dirs.sort()
        return ncnn_dirs[0]
    raise FileNotFoundError(f"No model artifact found inside {folder}")


def read_temperature_c():
    thermal = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        return float(thermal.read_text().strip()) / 1000.0
    except Exception:
        try:
            text = subprocess.check_output(["vcgencmd", "measure_temp"], text=True).strip()
            return float(text.split("=")[1].replace("'C", ""))
        except Exception:
            return math.nan


def throttling_status():
    try:
        return subprocess.check_output(["vcgencmd", "get_throttled"], text=True).strip()
    except Exception:
        return "unavailable"


def label_count(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    return sum(1 for line in label_path.read_text().splitlines() if line.strip())


def prediction_payload(result):
    detections = []
    counts = Counter()
    if result.boxes is not None:
        xyxy = result.boxes.xyxy.cpu().numpy().tolist()
        classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
        confidences = result.boxes.conf.cpu().numpy().tolist()
        for box, class_id, confidence in zip(xyxy, classes, confidences):
            counts[class_id] += 1
            detections.append({
                "box_xyxy": [float(value) for value in box],
                "class_id": int(class_id),
                "confidence": float(confidence),
            })
    return detections, counts


def main():
    parser = argparse.ArgumentParser(description="Benchmark all six rock detector formats on Raspberry Pi")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--formats", nargs="*", default=None, help="Optional folder names, e.g. 02_onnx_fp32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    models_root = project_root / "02_selected_model"
    test_root = project_root / "03_edge_test_dataset" / "02_preprocessed_640x640_90"
    image_dir = test_root / "images"
    label_dir = test_root / "labels"
    results_root = project_root / "06_edge_results"
    summary_root = results_root / "07_summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    runtime_visual_root = project_root / "visualization" / "08_raspberry_pi_runtime"
    runtime_visual_root.mkdir(parents=True, exist_ok=True)
    data_yaml = summary_root / "runtime_edge_test_data_640.yaml"
    data_yaml.write_text(
        yaml.safe_dump({
            "path": str(test_root),
            "train": "images",
            "val": "images",
            "test": "images",
            "nc": 2,
            "names": {0: "rock_completely_exposed", 1: "rock_half_buried"},
        }, sort_keys=False),
        encoding="utf-8",
    )

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No test images found in {image_dir}")

    process = psutil.Process(os.getpid())
    all_inference_rows = []
    all_summary_rows = []

    selected_formats = [
        item for item in FORMAT_FOLDERS
        if args.formats is None or item[0] in set(args.formats)
    ]

    for folder_name, display_name in selected_formats:
        print("\n" + "=" * 90)
        print("FORMAT:", display_name)
        model_folder = models_root / folder_name
        output_root = results_root / folder_name
        annotation_dir = output_root / "annotated_images"
        json_dir = output_root / "prediction_json"
        format_visual_dir = runtime_visual_root / folder_name / "annotated_images"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        format_visual_dir.mkdir(parents=True, exist_ok=True)
        model_path = discover_model(model_folder)
        print("Model:", model_path)

        load_start = time.perf_counter()
        model = YOLO(str(model_path), task="detect")
        load_ms = (time.perf_counter() - load_start) * 1000

        for _ in range(args.warmup):
            model.predict(
                str(image_paths[0]), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                batch=1, device="cpu", verbose=False
            )

        format_rows = []
        start_temperature = read_temperature_c()
        start_throttling = throttling_status()

        for pass_index in range(1, args.passes + 1):
            for image_index, image_path in enumerate(image_paths, start=1):
                psutil.cpu_percent(interval=None)
                before_rss = process.memory_info().rss
                start = time.perf_counter()
                result = model.predict(
                    str(image_path), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                    batch=1, device="cpu", verbose=False
                )[0]
                latency_ms = (time.perf_counter() - start) * 1000
                cpu_percent = psutil.cpu_percent(interval=None)
                after_rss = process.memory_info().rss
                temperature_c = read_temperature_c()

                detections, class_counts = prediction_payload(result)
                gt_count = label_count(label_dir / f"{image_path.stem}.txt")
                pred_count = len(detections)
                record = {
                    "format_folder": folder_name,
                    "format": display_name,
                    "model_path": str(model_path),
                    "pass": pass_index,
                    "image_index": image_index,
                    "image": image_path.name,
                    "latency_ms": latency_ms,
                    "fps": 1000.0 / latency_ms if latency_ms > 0 else math.nan,
                    "cpu_percent": cpu_percent,
                    "rss_before_mb": before_rss / (1024 ** 2),
                    "rss_after_mb": after_rss / (1024 ** 2),
                    "temperature_c": temperature_c,
                    "ground_truth_count": gt_count,
                    "predicted_count": pred_count,
                    "absolute_count_error": abs(gt_count - pred_count),
                    "completely_exposed_count": int(class_counts.get(0, 0)),
                    "half_buried_count": int(class_counts.get(1, 0)),
                }
                format_rows.append(record)
                all_inference_rows.append(record)

                if pass_index == 1:
                    plotted = result.plot()
                    cv2.imwrite(str(annotation_dir / image_path.name), plotted)
                    visual_name = f"RP_{folder_name}_IMG{image_index:03d}_{image_path.stem}_annotated.jpg"
                    cv2.imwrite(str(format_visual_dir / visual_name), plotted)
                    payload = {
                        "format": display_name,
                        "model_path": str(model_path),
                        "source_image": image_path.name,
                        "ground_truth_count": gt_count,
                        "predicted_count": pred_count,
                        "class_counts": {
                            "rock_completely_exposed": int(class_counts.get(0, 0)),
                            "rock_half_buried": int(class_counts.get(1, 0)),
                        },
                        "detections": detections,
                    }
                    (json_dir / f"{image_path.stem}.json").write_text(
                        json.dumps(payload, indent=2), encoding="utf-8"
                    )

        end_temperature = read_temperature_c()
        end_throttling = throttling_status()
        format_df = pd.DataFrame(format_rows)
        format_df.to_csv(output_root / "per_inference_metrics.csv", index=False)

        accuracy = {
            "precision": math.nan,
            "recall": math.nan,
            "f1": math.nan,
            "mAP50": math.nan,
            "mAP50_95": math.nan,
            "validation_error": "",
        }
        try:
            metrics = model.val(
                data=str(data_yaml), split="test", imgsz=args.imgsz, batch=1,
                device="cpu", workers=0, plots=True, project=str(output_root),
                name="accuracy_validation", exist_ok=True, verbose=False
            )
            precision = float(getattr(metrics.box, "mp", np.mean(metrics.box.p)))
            recall = float(getattr(metrics.box, "mr", np.mean(metrics.box.r)))
            accuracy.update({
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / max(precision + recall, 1e-12),
                "mAP50": float(metrics.box.map50),
                "mAP50_95": float(metrics.box.map),
            })
        except Exception as exc:
            accuracy["validation_error"] = repr(exc)

        summary = {
            "format_folder": folder_name,
            "format": display_name,
            "model_path": str(model_path),
            "model_load_ms": load_ms,
            "unique_test_images": len(image_paths),
            "timed_passes": args.passes,
            "timed_inferences": len(format_df),
            "median_latency_ms": float(format_df.latency_ms.median()),
            "mean_latency_ms": float(format_df.latency_ms.mean()),
            "p95_latency_ms": float(format_df.latency_ms.quantile(0.95)),
            "median_fps": float(format_df.fps.median()),
            "mean_cpu_percent": float(format_df.cpu_percent.mean()),
            "peak_rss_mb": float(format_df.rss_after_mb.max()),
            "mean_count_mae": float(format_df.absolute_count_error.mean()),
            "start_temperature_c": start_temperature,
            "maximum_temperature_c": float(format_df.temperature_c.max()),
            "end_temperature_c": end_temperature,
            "start_throttling": start_throttling,
            "end_throttling": end_throttling,
            **accuracy,
        }
        all_summary_rows.append(summary)
        pd.DataFrame([summary]).to_csv(output_root / "format_summary.csv", index=False)
        print(pd.DataFrame([summary]).to_string(index=False))

        del model
        gc.collect()

    all_inference_df = pd.DataFrame(all_inference_rows)
    all_summary_df = pd.DataFrame(all_summary_rows)
    all_inference_df.to_csv(summary_root / "all_formats_per_inference_metrics.csv", index=False)
    all_summary_df.to_csv(summary_root / "edge_benchmark_summary.csv", index=False)

    visualization_records = []

    def save_bar_plot(metric, ylabel, filename, title):
        if metric not in all_summary_df.columns:
            return
        plot_df = all_summary_df[["format", metric]].dropna()
        if plot_df.empty:
            return
        plt.figure(figsize=(10, 6))
        plt.bar(plot_df["format"], plot_df[metric])
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        output_path = runtime_visual_root / filename
        plt.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close()
        visualization_records.append((filename, str(output_path), title))

    save_bar_plot(
        "median_latency_ms",
        "Median latency (ms/image)",
        "RP01_format_median_latency_ms.png",
        "Raspberry Pi format comparison: median latency",
    )
    save_bar_plot(
        "median_fps",
        "Median FPS",
        "RP02_format_median_fps.png",
        "Raspberry Pi format comparison: median FPS",
    )
    save_bar_plot(
        "peak_rss_mb",
        "Peak process RAM (MB)",
        "RP03_format_peak_ram_mb.png",
        "Raspberry Pi format comparison: peak RAM",
    )
    save_bar_plot(
        "maximum_temperature_c",
        "Maximum temperature (°C)",
        "RP04_format_maximum_temperature_c.png",
        "Raspberry Pi format comparison: maximum temperature",
    )
    save_bar_plot(
        "mAP50_95",
        "mAP50-95",
        "RP05_format_map50_95.png",
        "Raspberry Pi format comparison: detection accuracy",
    )
    save_bar_plot(
        "mean_count_mae",
        "Mean absolute rock-count error",
        "RP06_format_count_mae.png",
        "Raspberry Pi format comparison: rock-count error",
    )

    tradeoff_df = all_summary_df[["format", "median_latency_ms", "mAP50_95"]].dropna()
    if not tradeoff_df.empty:
        plt.figure(figsize=(9, 6))
        plt.scatter(tradeoff_df["median_latency_ms"], tradeoff_df["mAP50_95"], s=80)
        for row in tradeoff_df.itertuples(index=False):
            plt.annotate(
                row.format,
                (row.median_latency_ms, row.mAP50_95),
                xytext=(5, 5),
                textcoords="offset points",
            )
        plt.xlabel("Median latency (ms/image)")
        plt.ylabel("mAP50-95")
        plt.title("Raspberry Pi accuracy–latency trade-off")
        plt.tight_layout()
        tradeoff_path = runtime_visual_root / "RP07_accuracy_latency_tradeoff.png"
        plt.savefig(tradeoff_path, dpi=180, bbox_inches="tight")
        plt.close()
        visualization_records.append((
            tradeoff_path.name,
            str(tradeoff_path),
            "Raspberry Pi accuracy–latency trade-off",
        ))

    visualization_list_path = runtime_visual_root / "RASPBERRY_PI_VISUALIZATION_LIST.txt"
    visualization_list_path.write_text(
        "\n".join(
            f"{index:02d}. {name} | {description} | {path}"
            for index, (name, path, description) in enumerate(visualization_records, start=1)
        ) + "\n",
        encoding="utf-8",
    )

    print("\nSaved combined summary:", summary_root / "edge_benchmark_summary.csv")
    print("Raspberry Pi visualizations:", runtime_visual_root)
    print(all_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
