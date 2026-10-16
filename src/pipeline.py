"""Orchestration du pipeline complet, etape par etape, avec arret au premier echec (fail fast).

Chaque etape est un module independant ; l'ordre reproduit le DAG du projet :
manifestes -> controle qualite -> CNN -> evaluation -> LBP-SVM -> fusion -> export -> mesures.

Usage : python -m src.pipeline            (tout)
        python -m src.pipeline --from fusion   (reprendre a une etape)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

STEPS = [
    ("manifest", ["-m", "src.preprocessing.build_manifest"]),
    ("webcam", ["-m", "src.preprocessing.webcam_manifest"]),
    ("quality", ["-m", "src.preprocessing.validate_data"]),
    ("train", ["-m", "src.train", "--name", "pipeline"]),
    ("evaluate", ["-m", "src.evaluate"]),
    ("lbp", ["-m", "src.baseline_lbp_svm"]),
    ("fusion", ["-m", "src.fusion"]),
    ("export", ["-m", "src.export_onnx"]),
    ("benchmark", ["-m", "src.benchmark"]),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=[s for s, _ in STEPS], default=STEPS[0][0])
    args = ap.parse_args()
    names = [s for s, _ in STEPS]
    for name, cmd in STEPS[names.index(args.start):]:
        t0 = time.time()
        print(f"\n===== [{name}] =====", flush=True)
        code = subprocess.call([sys.executable, "-u", *cmd])
        if code != 0:
            print(f"ECHEC de l'etape {name} (code {code}) : arret du pipeline.", flush=True)
            return code
        print(f"===== [{name}] OK en {time.time() - t0:.0f} s =====", flush=True)
    archive_version()
    print("\nPIPELINE TERMINE", flush=True)
    return 0


def archive_version() -> None:
    """Registre de modeles minimal : chaque execution est conservee dans models/saved/versions/<date>/."""
    import shutil

    from src import config

    dest = config.MODELS_DIR / "versions" / time.strftime("%Y%m%d-%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("mobilenetv2_fas.pt", "mobilenetv2_fas.onnx", "model_meta.json", "lbp_svm.joblib",
                 "fusion.json", "train_summary.json"):
        if (config.MODELS_DIR / name).is_file():
            shutil.copy2(config.MODELS_DIR / name, dest / name)
    for name in ("final_metrics.json", "test_metrics.json", "baseline_lbp_svm.json", "app_benchmark.json"):
        if (config.REPORTS_DIR / name).is_file():
            shutil.copy2(config.REPORTS_DIR / name, dest / name)
    print(f"Version archivee : {dest}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
