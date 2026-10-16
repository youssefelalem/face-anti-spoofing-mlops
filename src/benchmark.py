"""Mesures de l'application reelle (taille du modele, latence) pour le poster et le README.

Usage : python -m src.benchmark
Sortie : reports/app_benchmark.json
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time

import cv2

from app.inference import AntiSpoofingEngine
from src import config
from src.preprocessing.webcam_manifest import read_webcam_manifest


def count_tests() -> int:
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:warnings"],
                         capture_output=True, text=True, cwd=config.ROOT).stdout
    total = 0
    for line in out.splitlines():                  # formats "fichier::test" ou "fichier: N"
        if "::" in line:
            total += 1
        elif line.startswith("tests/") and line.rsplit(":", 1)[-1].strip().isdigit():
            total += int(line.rsplit(":", 1)[-1])
    return total


def main() -> None:
    engine = AntiSpoofingEngine(config.ONNX_MODEL, config.MODEL_META, config.MODELS_DIR / "lbp_svm.joblib",
                                config.MODELS_DIR / "fusion.json")
    # Images de taille webcam (640x480) : visage colle sur un fond, comme dans l'application
    faces = [r for r in read_webcam_manifest() if r["split"] == "test"][:60]
    frames = []
    for r in faces:
        face = cv2.resize(cv2.imread(str(config.WEBCAM_DIR / r["path"])), (240, 240))
        frame = cv2.copyMakeBorder(face, 120, 120, 200, 200, cv2.BORDER_REPLICATE)
        frames.append(frame)
    for f in frames[:5]:
        engine.predict(f)                                   # echauffement
    times = []
    for f in frames:
        t0 = time.perf_counter()
        engine.predict(f)
        times.append((time.perf_counter() - t0) * 1000)
    dq = json.loads((config.REPORTS_DIR / "data_quality_report.json").read_text(encoding="utf-8"))
    res = {
        "mode": engine.mode,
        "onnx_mb": config.ONNX_MODEL.stat().st_size / 1e6,
        "lbp_svm_mb": (config.MODELS_DIR / "lbp_svm.joblib").stat().st_size / 1e6,
        "latency_ms_median": statistics.median(times),
        "latency_ms_p95": sorted(times)[int(0.95 * (len(times) - 1))],
        "n_frames": len(times),
        "n_tests": count_tests(),
        "n_quality_checks": len(dq["checks"]),
    }
    (config.REPORTS_DIR / "app_benchmark.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
