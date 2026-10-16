"""Evaluation finale sur le jeu de TEST (session 03, jamais vue a l'entrainement).

Le seuil de decision est celui fixe sur la validation (EER) : on ne le
re-ajuste PAS sur le test, sinon le resultat serait optimiste.

Produit :
- reports/test_metrics.json
- reports/figures/*.png (courbe ROC, matrice de confusion, distribution des
  scores, courbes d'apprentissage) -> utilisees dans le poster.

Usage : python -m src.evaluate
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src import config  # noqa: E402
from src.models.mobilenet import build_model, predict_proba  # noqa: E402
from src.preprocessing.build_manifest import read_manifest  # noqa: E402
from src.preprocessing.dataset import NuaaDataset  # noqa: E402
from src.utils.metrics import auc, eer_threshold, pad_metrics  # noqa: E402

BLUE, ORANGE, GREY = "#1f5fa8", "#d9822b", "#555555"


def plot_roc(scores, labels, eer, path) -> None:
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"MobileNetV2 (AUC = {auc(scores, labels):.4f})")
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="Hasard")
    ax.plot([eer], [1 - eer], "o", color=ORANGE, label=f"EER = {eer * 100:.2f} %")
    ax.set_xlabel("BPCER (vrais visages rejetes)")
    ax.set_ylabel("1 - APCER (attaques detectees)")
    ax.set_title("Courbe ROC - jeu de test NUAA")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(conf: dict, path) -> None:
    m = np.array([[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4), dpi=200)
    ax.imshow(m, cmap="Blues")
    for (i, j), v in np.ndenumerate(m):
        pct = v / m[i].sum() * 100
        ax.text(j, i, f"{v}\n({pct:.1f} %)", ha="center", va="center",
                color="white" if v > m.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1], ["Reel", "Attaque"])
    ax.set_yticks([0, 1], ["Reel", "Attaque"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Verite terrain")
    ax.set_title("Matrice de confusion (test)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_scores(scores, labels, threshold, path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=200)
    bins = np.linspace(0, 1, 41)
    ax.hist(scores[labels == 0], bins=bins, alpha=0.7, color=BLUE, label="Visages reels")
    ax.hist(scores[labels == 1], bins=bins, alpha=0.7, color=ORANGE, label="Attaques (photo)")
    ax.axvline(threshold, color="black", ls="--", lw=1.2, label=f"Seuil = {threshold:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("Score d'attaque predit")
    ax.set_ylabel("Nombre d'images (log)")
    ax.set_title("Distribution des scores - test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_learning_curves(run_id: str, path) -> None:
    client = mlflow.tracking.MlflowClient()
    fig, ax1 = plt.subplots(figsize=(6, 3.6), dpi=200)
    for key, color, label in [("train_loss", BLUE, "Perte entrainement"),
                              ("val_loss", ORANGE, "Perte validation")]:
        hist = sorted(client.get_metric_history(run_id, key), key=lambda m: m.step)
        ax1.plot([m.step for m in hist], [m.value for m in hist], "-o", ms=3, color=color, label=label)
    eer = sorted(client.get_metric_history(run_id, "val_eer"), key=lambda m: m.step)
    ax2 = ax1.twinx()
    ax2.plot([m.step for m in eer], [m.value * 100 for m in eer], "--", color=GREY,
             label="EER validation (%)")
    ax1.axvline(config.HEAD_EPOCHS + 0.5, color=GREY, lw=0.8, ls=":")
    ax1.text(config.HEAD_EPOCHS + 0.6, ax1.get_ylim()[1] * 0.92, "debut fine-tuning", fontsize=8, color=GREY)
    ax1.set_xlabel("Epoque")
    ax1.set_ylabel("Perte (cross-entropy)")
    ax2.set_ylabel("EER validation (%)")
    lines = ax1.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax1.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="upper right", fontsize=8)
    ax1.set_title("Courbes d'apprentissage")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    torch.set_num_threads(os.cpu_count() or 4)
    ckpt = torch.load(config.CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    threshold = float(ckpt["val"]["threshold"])
    cfg = ckpt.get("config", {})

    test_rows = [r for r in read_manifest() if r["split"] == "test"]
    print(f"test={len(test_rows)} images (chargement en memoire...)")
    loader = DataLoader(NuaaDataset(test_rows, train=False, grayscale=cfg.get("grayscale", False),
                                    photo_norm=cfg.get("photo_norm", False)),
                        batch_size=128, num_workers=0)
    probs, labels = predict_proba(model, loader)
    scores, y = probs.numpy(), labels.numpy()

    at_val_thr = pad_metrics(scores, y, threshold)
    test_eer, _ = eer_threshold(scores, y)
    results = {
        "n_test": int(len(y)),
        "n_test_real": int((y == 0).sum()),
        "n_test_spoof": int((y == 1).sum()),
        "auc": auc(scores, y),
        "eer": test_eer,
        "at_validation_threshold": at_val_thr,
        "validation": {k: v for k, v in ckpt["val"].items() if not isinstance(v, dict)},
        "validation_protocol": config.VAL_PROTOCOL,
        "train_config": ckpt.get("config"),
    }

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "test_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_roc(scores, y, test_eer, config.FIGURES_DIR / "roc_curve.png")
    plot_confusion(at_val_thr["confusion"], config.FIGURES_DIR / "confusion_matrix.png")
    plot_scores(scores, y, threshold, config.FIGURES_DIR / "score_distribution.png")

    summary = json.loads((config.MODELS_DIR / "train_summary.json").read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    plot_learning_curves(summary["mlflow_run_id"], config.FIGURES_DIR / "learning_curves.png")
    with mlflow.start_run(run_id=summary["mlflow_run_id"], nested=False):
        mlflow.log_metrics({"test_auc": results["auc"], "test_eer": test_eer,
                            "test_accuracy": at_val_thr["accuracy"], "test_APCER": at_val_thr["APCER"],
                            "test_BPCER": at_val_thr["BPCER"], "test_ACER": at_val_thr["ACER"]})
        mlflow.log_artifacts(str(config.FIGURES_DIR), artifact_path="figures")
        mlflow.log_artifact(str(config.REPORTS_DIR / "test_metrics.json"))

    m = at_val_thr
    print(f"\nTEST (seuil de validation = {threshold:.4f})")
    print(f"  Accuracy = {m['accuracy'] * 100:.2f} %")
    print(f"  APCER    = {m['APCER'] * 100:.2f} %   (attaques acceptees)")
    print(f"  BPCER    = {m['BPCER'] * 100:.2f} %   (vrais visages rejetes)")
    print(f"  ACER     = {m['ACER'] * 100:.2f} %")
    print(f"  AUC      = {results['auc']:.4f}   EER = {test_eer * 100:.2f} %")
    print(f"Figures : {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
