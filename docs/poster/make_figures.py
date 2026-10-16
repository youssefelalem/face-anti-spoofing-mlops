"""Figures du poster qui ne sont pas des resultats : architecture, pipeline MLOps, echantillons.

Usage : python docs/poster/make_figures.py
Sorties : reports/figures/architecture.png, reports/figures/mlops_pipeline.png,
          reports/samples_nuaa.png (contient des visages -> hors Git, voir .gitignore)
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402

NAVY, BLUE, ICE, GREY = "#0B1E3F", "#1F5FA8", "#E8EEF8", "#5C6576"
ORANGE, GREEN, RED = "#D9822B", "#1D8A4E", "#C4322B"


def box(ax, x, y, w, h, title, sub="", fc=ICE, ec=BLUE, tc=NAVY, fs=13):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), title, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=tc)
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=fs - 3.5,
                color="white" if tc == "white" else GREY)


def arrow(ax, x0, y0, x1, y1, color=NAVY):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18,
                                 lw=1.8, color=color))


def architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=220)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    box(ax, 0.1, 1.5, 1.6, 1.3, "Face crop", "RGB 128x128")
    arrow(ax, 1.75, 2.15, 2.15, 2.15)
    # Backbone gele / fine-tune
    ax.add_patch(FancyBboxPatch((2.2, 0.75), 5.0, 2.8, boxstyle="round,pad=0.02,rounding_size=0.1",
                                fc="white", ec=BLUE, lw=1.6, ls="--"))
    ax.text(4.7, 3.78, "MobileNetV2 backbone - pretrained on ImageNet (1.2 M images)",
            ha="center", fontsize=11.5, fontweight="bold", color=NAVY)
    blocks = [("Conv 3x3", "32"), ("Inverted\nresidual x17", "16-320"), ("Conv 1x1", "1280")]
    xs = [2.45, 3.95, 5.65]
    ws = [1.2, 1.45, 1.3]
    for (t, s), x, w in zip(blocks, xs, ws, strict=True):
        box(ax, x, 1.5, w, 1.3, t, s, fs=12)
    arrow(ax, 3.67, 2.15, 3.93, 2.15)
    arrow(ax, 5.42, 2.15, 5.63, 2.15)
    ax.text(4.7, 1.0, "Phase 1: frozen   |   Phase 2: fine-tuned (lr = 1e-4)",
            ha="center", fontsize=10.5, color=GREY, style="italic")
    arrow(ax, 7.25, 2.15, 7.6, 2.15)
    box(ax, 7.65, 1.5, 1.45, 1.3, "Global\navg pool", "1280", fs=12)
    arrow(ax, 9.12, 2.15, 9.35, 2.15)
    box(ax, 9.4, 1.5, 1.5, 1.3, "New head", "Dropout + FC 2", fc=ORANGE, ec=ORANGE, tc="white", fs=12)
    ax.text(10.15, 1.15, "softmax", ha="center", fontsize=10, color=GREY)
    ax.text(10.15, 0.75, "P(real)  P(spoof)", ha="center", fontsize=11, fontweight="bold", color=NAVY)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, transparent=False, facecolor="white")
    plt.close(fig)


def fusion_architecture(path: Path) -> None:
    """Deux branches (CNN profond + texture LBP) calibrees puis fusionnees."""
    # Taille physique ~ celle du cadre du poster (22 x 8 cm) : 12 pt ici = 12 pt imprimes
    fig, ax = plt.subplots(figsize=(8.8, 3.3), dpi=260)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    box(ax, 0.05, 1.45, 1.6, 1.5, "Face", "crop", fs=14)
    # Branche Deep Learning
    arrow(ax, 1.7, 2.45, 2.1, 3.2)
    box(ax, 2.15, 2.6, 3.3, 1.4, "MobileNetV2", "CNN, ImageNet", fs=14)
    arrow(ax, 5.5, 3.3, 5.9, 3.3)
    box(ax, 5.95, 2.6, 1.75, 1.4, "Platt", "calibr.", fs=13)
    # Branche texture
    arrow(ax, 1.7, 1.95, 2.1, 1.2)
    box(ax, 2.15, 0.45, 3.3, 1.4, "LBP + SVM", "micro-texture", fs=14)
    arrow(ax, 5.5, 1.15, 5.9, 1.15)
    box(ax, 5.95, 0.45, 1.75, 1.4, "Platt", "calibr.", fs=13)
    # Fusion
    arrow(ax, 7.75, 3.3, 8.15, 2.5)
    arrow(ax, 7.75, 1.15, 8.15, 1.9)
    box(ax, 8.2, 1.45, 1.4, 1.5, "Mean", "fusion", fc=ORANGE, ec=ORANGE, tc="white", fs=14)
    arrow(ax, 9.65, 2.2, 9.9, 2.2)
    ax.text(10.5, 2.55, "Live", ha="center", fontsize=14, fontweight="bold", color=GREEN)
    ax.text(10.5, 2.1, "or", ha="center", fontsize=11, color=GREY)
    ax.text(10.5, 1.6, "Attack", ha="center", fontsize=14, fontweight="bold", color=RED)
    ax.text(3.8, 4.12, "Deep Learning branch", ha="center", fontsize=12, color=BLUE, fontweight="bold")
    ax.text(3.8, 0.0, "Classical ML branch", ha="center", fontsize=12, color=GREY, fontweight="bold")
    fig.tight_layout(pad=0.2)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def validation_gap(path: Path) -> None:
    """AUC en validation interne vs sur une session jamais vue : le constat principal."""
    import json

    rep = config.REPORTS_DIR

    def load(name):
        p = rep / name
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    v1, v2, lbp = (load("test_metrics_v1_subject_protocol.json"), load("test_metrics_v2_gray_photonorm.json"),
                   load("baseline_lbp_svm.json"))
    items = []
    if v1:
        items.append(("CNN (colour)", v1["validation"]["auc"], v1["auc"]))
    if v2:
        items.append(("CNN (grey + photo-norm)", v2["validation"]["auc"], v2["auc"]))
    if lbp:
        items.append(("LBP + SVM", lbp["validation"]["auc"], lbp["test"]["nuaa"]["auc"]))
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    x = np.arange(len(items))
    ax.bar(x - 0.19, [i[1] for i in items], 0.36, color=ICE, edgecolor=BLUE,
           label="Validation (sessions 1-2)")
    ax.bar(x + 0.19, [i[2] for i in items], 0.36, color=ORANGE, label="Test (unseen session 3)")
    for xi, (_, a, b) in zip(x, items, strict=True):
        ax.text(xi - 0.19, a + 0.008, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(xi + 0.19, b + 0.008, f"{b:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x, [i[0] for i in items])
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("AUC")
    ax.set_title("Internal validation hides the session shift (NUAA)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=9.5, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def mlops(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=260)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.text(0.1, 5.3, "DATA PIPELINE", fontsize=12, fontweight="bold", color=BLUE)
    row1 = [("Ingestion", "SHA-256"), ("Manifest", "splits"),
            ("Contract", "quality, bias,\nleakage"), ("Training", "MLflow"),
            ("Evaluation", "ISO 30107-3")]
    for i, (t, s) in enumerate(row1):
        x = 0.1 + i * 2.2
        box(ax, x, 3.55, 1.9, 1.45, t, s, fs=12.5)
        if i:
            arrow(ax, x - 0.28, 4.27, x - 0.02, 4.27)
    ax.text(1.25, 2.7, "DELIVERY PIPELINE", fontsize=12, fontweight="bold", color=BLUE)
    row2 = [("ONNX", "parity check"), ("CI", "GitHub Actions\npytest"),
            ("Docker", "slim, non-root"), ("FastAPI", "/predict"),
            ("Web demo", "live webcam")]
    for i, (t, s) in enumerate(row2):
        x = 0.1 + i * 2.2
        box(ax, x, 1.0, 1.9, 1.45, t, s, fc="white", fs=12.5)
        if i:
            arrow(ax, x - 0.28, 1.72, x - 0.02, 1.72)
    # Liaison en coude : Evaluation -> ONNX export
    ax.plot([10.05, 10.05, 1.05], [3.53, 3.12, 3.12], color=GREY, lw=1.8, solid_capstyle="round")
    arrow(ax, 1.05, 3.12, 1.05, 2.47, color=GREY)
    ax.text(5.5, 0.45, "Fail fast: a broken contract or test blocks the pipeline",
            ha="center", fontsize=12, style="italic", color=GREY)
    fig.tight_layout(pad=0.2)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def samples(path: Path, n: int = 5) -> None:
    random.seed(7)
    rows = list(csv.DictReader(config.MANIFEST_CSV.open(encoding="utf-8")))
    real = random.sample([r for r in rows if r["label"] == "0" and r["split"] == "test"], n)
    spoof = random.sample([r for r in rows if r["label"] == "1" and r["split"] == "test"], n)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.9), dpi=200)
    for row_axes, group, name, color in [(axes[0], real, "Bona fide (live)", GREEN),
                                         (axes[1], spoof, "Print attack", RED)]:
        for ax, r in zip(row_axes, group, strict=True):
            img = cv2.cvtColor(cv2.imread(str(config.NUAA_DIR / r["path"])), cv2.COLOR_BGR2RGB)
            ax.imshow(cv2.resize(img, (160, 160)))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(color)
                s.set_linewidth(3)
        row_axes[0].set_ylabel(name, fontsize=13, fontweight="bold", color=color)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def webcam_samples(path: Path, subject: str = "youssef", n: int = 5) -> None:
    """Echantillons des donnees webcam de l'auteur uniquement (aucun visage d'un tiers)."""
    random.seed(3)
    rows = list(csv.DictReader(config.WEBCAM_MANIFEST_CSV.open(encoding="utf-8")))
    mine = [r for r in rows if r["subject"] == subject]
    real = random.sample([r for r in mine if r["attack"] == "real"], n)
    spoof = random.sample([r for r in mine if r["attack"] != "real"], n)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.9), dpi=200)
    for row_axes, group, name, color in [(axes[0], real, "Live face", GREEN),
                                         (axes[1], spoof, "Screen attack", RED)]:
        for ax, r in zip(row_axes, group, strict=True):
            img = cv2.cvtColor(cv2.imread(str(config.WEBCAM_DIR / r["path"])), cv2.COLOR_BGR2RGB)
            ax.imshow(cv2.resize(img, (160, 160)))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(color)
                s.set_linewidth(3)
        row_axes[0].set_ylabel(name, fontsize=13, fontweight="bold", color=color)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    architecture(config.FIGURES_DIR / "architecture.png")
    fusion_architecture(config.FIGURES_DIR / "fusion_architecture.png")
    validation_gap(config.FIGURES_DIR / "validation_gap.png")
    mlops(config.FIGURES_DIR / "mlops_pipeline.png")
    samples(config.REPORTS_DIR / "samples_nuaa.png")            # usage interne uniquement
    webcam_samples(config.REPORTS_DIR / "samples_webcam.png")   # poster / video : visage de l'auteur
    print("figures ecrites dans", config.FIGURES_DIR, "et", config.REPORTS_DIR)
