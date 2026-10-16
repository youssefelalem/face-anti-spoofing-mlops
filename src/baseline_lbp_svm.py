"""Modele classique : descripteurs de texture LBP + SVM (Maatta et al., 2011).

C'est du Machine Learning supervise "classique" (caracteristiques manuelles +
SVM), compare au Deep Learning (MobileNetV2) et fusionne avec lui dans l'application.

Le score SVM est calibre en probabilite (regression logistique sur la
validation, methode de Platt) pour pouvoir etre combine avec le CNN.

Usage : python -m src.baseline_lbp_svm
Sorties : models/saved/lbp_svm.joblib, reports/baseline_lbp_svm.json
"""

from __future__ import annotations

import json

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config
from src.preprocessing.build_manifest import read_manifest
from src.preprocessing.dataset import image_path
from src.preprocessing.webcam_manifest import read_webcam_manifest
from src.utils.lbp import lbp_features
from src.utils.metrics import auc, eer_threshold, pad_metrics

LBP_MODEL = config.MODELS_DIR / "lbp_svm.joblib"


def load(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([lbp_features(cv2.imread(str(image_path(r)))) for r in rows])
    return x, np.array([r["label"] for r in rows])


def summary(scores, labels, threshold) -> dict:
    eer, _ = eer_threshold(scores, labels)
    return {"n": int(len(labels)), "auc": auc(scores, labels), "eer": eer,
            "at_validation_threshold": pad_metrics(scores, labels, threshold)}


def main() -> None:
    nuaa = read_manifest()
    webcam = read_webcam_manifest() if config.USE_WEBCAM else []
    train = [r for r in nuaa + webcam if r["split"] == "train"]
    val = [r for r in nuaa + webcam if r["split"] == "val"]
    x_tr, y_tr = load(train)
    x_va, y_va = load(val)

    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale"))
    svm.fit(x_tr, y_tr)
    calibrator = LogisticRegression().fit(svm.decision_function(x_va)[:, None], y_va)
    proba = lambda x: calibrator.predict_proba(svm.decision_function(x)[:, None])[:, 1]  # noqa: E731
    p_va = proba(x_va)
    val_eer, thr = eer_threshold(p_va, y_va)

    results = {"model": "LBP(8,1) global + 3x3, LBP(8,2) global, SVM RBF (C=10)",
               "n_features": int(x_tr.shape[1]), "n_train": len(train),
               "validation": {"auc": auc(p_va, y_va), "eer": val_eer, "threshold": thr}, "test": {}}
    test_sets = {"nuaa": [r for r in nuaa if r["split"] == "test"],
                 "webcam": [r for r in webcam if r["split"] == "test"]}
    for name, rows in test_sets.items():
        if rows:
            x_te, y_te = load(rows)
            results["test"][name] = summary(proba(x_te), y_te, thr)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"svm": svm, "calibrator": calibrator, "threshold": thr}, LBP_MODEL)
    (config.REPORTS_DIR / "baseline_lbp_svm.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"LBP+SVM  val: AUC={results['validation']['auc']:.4f} EER={val_eer * 100:.2f} %")
    for name, m in results["test"].items():
        t = m["at_validation_threshold"]
        print(f"LBP+SVM test {name}: AUC={m['auc']:.4f} EER={m['eer'] * 100:.2f} % "
              f"ACER={t['ACER'] * 100:.2f} % "
              f"(APCER={t['APCER'] * 100:.2f} %, BPCER={t['BPCER'] * 100:.2f} %)")


if __name__ == "__main__":
    main()
