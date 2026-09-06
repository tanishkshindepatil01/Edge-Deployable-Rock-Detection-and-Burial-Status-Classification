#!/usr/bin/env python3
import argparse
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path


def import_status(module_name):
    try:
        module = importlib.import_module(module_name)
        return {"available": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    project_root = args.project_root.resolve()

    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: import_status(name)
            for name in ["ultralytics", "onnxruntime", "cv2", "numpy", "pandas", "psutil"]
        },
        "litert_backends": {
            name: import_status(name)
            for name in ["ai_edge_litert.interpreter", "tflite_runtime.interpreter", "tensorflow.lite"]
        },
    }
    try:
        report["vcgencmd_version"] = subprocess.check_output(
            ["vcgencmd", "version"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as exc:
        report["vcgencmd_error"] = repr(exc)

    output = project_root / "06_edge_results" / "07_summary" / "raspberry_pi_environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()
