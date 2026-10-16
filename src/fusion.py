"""Fusion CNN (MobileNetV2) + LBP-SVM, et comparaison finale des trois approches.

1. Chaque modele donne un score d'attaque ; chacun est calibre en probabilite
   sur la VALIDATION (regression logistique, methode de Platt).
2. Score fusionne = moyenne des deux probabilites calibrees.
3. Le seuil de decision est fixe a l'EER de la validation.
4. Comparaison sur les deux jeux de test : NUAA session 03, et webcam session s2.

Usage : python -m src.fusion   (apres src.train et src.baseline_lbp_svm)
Sorties : models/saved/fusion.json, reports/final_metrics.json, reports/figures/*.png
"""

from __future__ import annotations

import json
import os

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src import config  # noqa: E402
from src.baseline_lbp_svm import LBP_MODEL  # noqa: E402
from src.models.mobilenet import build_model, predict_proba  # noqa: E402
from src.preprocessing.build_manifest import read_manifest  # noqa: E402
from src.preprocessing.dataset import NuaaDataset, image_path  # noqa: E402
from src.preprocessing.webcam_manifest import read_webcam_manifest  # noqa: E402
from src.utils.lbp import lbp_features  # noqa: E402
from src.utils.metrics import auc, eer_threshold, pad_metrics  # noqa: E402

FUSION_META = config.MODELS_DIR / "fusion.json"
COLORS = {"CNN (MobileNetV2)": "#1f5fa8", "LBP + SVM": "#7a8696", "Fusion": "#d9822b"}
EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def cnn_scores(model, cfg: dict, rows: list[dict]) -> np.ndarray:
    ds = NuaaDataset(rows, train=False, grayscale=cfg.get("grayscale", False),
                     photo_norm=cfg.get("photo_norm", False))
    return predict_proba(model, DataLoader(ds, batch_size=128))[0].numpy()


def lbp_scores(lbp: dict, rows: list[dict]) -> np.ndarray:
    x = np.stack([lbp_features(cv2.imread(str(image_path(r)))) for r in rows])
    return lbp["calibrator"].predict_proba(lbp["svm"].decision_function(x)[:, None])[:, 1]


def report(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    eer, _ = eer_threshold(scores, labels)
    return {"n": int(len(labels)), "auc": auc(scores, labels), "eer": eer,
            "at_validation_threshold": pad_metrics(scores, labels, threshold)}


def plot_roc(curves: dict, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 4.3), dpi=220)
    short = {"CNN (MobileNetV2)": "CNN", "LBP + SVM": "LBP-SVM", "Fusion": "Fusion"}
    for name, (s, y) in curves.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, lw=2.6 if name == "Fusion" else 2.0, color=COLORS[name],
                label=f"{short[name]}  {auc(s, y):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1)
    ax.set_xlabel("BPCER", fontsize=11)
    ax.set_ylabel("1 - APCER", fontsize=11)
    ax.set_title(title, fontsize=11.5)
    ax.legend(loc="lower right", fontsize=10.5, title="AUC", title_fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_bars(results: dict, path) -> None:
    sets = [s for s in ("nuaa", "webcam") if s in results["test"]]
    names = list(COLORS)
    labels = {"nuaa": "NUAA test (session 3)", "webcam": "Own webcam test (session s2)"}
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
    width = 0.8 / len(names)
    for i, name in enumerate(names):
        vals = [results["test"][s][name]["eer"] * 100 for s in sets]
        xs = np.arange(len(sets)) + (i - (len(names) - 1) / 2) * width
        bars = ax.bar(xs, vals, width * 0.92, color=COLORS[name], label=name)
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", fontsize=9)
    top = max(results["test"][s][n]["eer"] * 100 for s in sets for n in names)
    ax.set_ylim(0, top * 1.18)                      # marge pour les etiquettes au-dessus des barres
    ax.set_xticks(np.arange(len(sets)), [labels[s] for s in sets])
    ax.set_ylabel("EER (%) - lower is better")
    ax.set_title("Equal Error Rate on unseen sessions")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(conf: dict, title: str, path) -> None:
    m = np.array([[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4), dpi=200)
    ax.imshow(m, cmap="Blues")
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, f"{v}\n({v / max(m[i].sum(), 1) * 100:.1f} %)", ha="center", va="center",
                color="white" if v > m.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1], ["Live", "Attack"])
    ax.set_yticks([0, 1], ["Live", "Attack"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(os.cpu_count() or 4)
    ckpt = torch.load(config.CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    lbp = joblib.load(LBP_MODEL)

    nuaa, webcam = read_manifest(), read_webcam_manifest()
    sets = {"val": [r for r in nuaa + webcam if r["split"] == "val"],
            "nuaa": [r for r in nuaa if r["split"] == "test"],
            "webcam": [r for r in webcam if r["split"] == "test"]}
    sets = {k: v for k, v in sets.items() if v}

    scores = {}
    for name, rows in sets.items():
        print(f"scores {name} ({len(rows)} images)...", flush=True)
        scores[name] = {"y": np.array([r["label"] for r in rows]),
                        "cnn": cnn_scores(model, cfg, rows), "lbp": lbp_scores(lbp, rows)}

    # Calibration du CNN et seuil de fusion : uniquement sur la validation
    v = scores["val"]
    cnn_cal = LogisticRegression().fit(logit(v["cnn"])[:, None], v["y"])

    def fused(s):
        return (cnn_cal.predict_proba(logit(s["cnn"])[:, None])[:, 1] + s["lbp"]) / 2

    thresholds = {"CNN (MobileNetV2)": eer_threshold(v["cnn"], v["y"])[1],
                  "LBP + SVM": float(lbp["threshold"]),
                  "Fusion": eer_threshold(fused(v), v["y"])[1]}
    results = {"thresholds": thresholds, "validation": {}, "test": {}}
    for name, s in scores.items():
        per_model = {"CNN (MobileNetV2)": s["cnn"], "LBP + SVM": s["lbp"], "Fusion": fused(s)}
        out = {m: report(sc, s["y"], thresholds[m]) for m, sc in per_model.items()}
        (results["validation"] if name == "val" else results["test"])[name if name != "val" else "all"] = out
        if name != "val":
            title = "NUAA test (session 3)" if name == "nuaa" else "Webcam test (new person)"
            plot_roc({m: (sc, s["y"]) for m, sc in per_model.items()}, f"ROC - {title}",
                     config.FIGURES_DIR / f"roc_{name}.png")
            plot_confusion(out["Fusion"]["at_validation_threshold"]["confusion"],
                           f"Fusion - {title}", config.FIGURES_DIR / f"confusion_fusion_{name}.png")

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_bars(results, config.FIGURES_DIR / "eer_comparison.png")
    FUSION_META.write_text(json.dumps({
        "method": "mean of Platt-calibrated probabilities",
        "cnn_calibration": {"coef": float(cnn_cal.coef_[0, 0]), "intercept": float(cnn_cal.intercept_[0])},
        "threshold": thresholds["Fusion"]}, indent=2), encoding="utf-8")
    (config.REPORTS_DIR / "final_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    for test_name, per_model in results["test"].items():
        print(f"\n== TEST {test_name} ==")
        for m, r in per_model.items():
            t = r["at_validation_threshold"]
            print(f"  {m:18s} AUC={r['auc']:.4f} EER={r['eer'] * 100:5.2f} %  ACER={t['ACER'] * 100:5.2f} % "
                  f"(APCER={t['APCER'] * 100:.2f} %, BPCER={t['BPCER'] * 100:.2f} %)")


if __name__ == "__main__":
    main()
