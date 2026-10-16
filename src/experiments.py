"""Comparaison de strategies d'entrainement, SANS utiliser le jeu de test.

Protocole : entrainement sur la session 01, validation sur la session 02.

Chaque configuration est aussi evaluee sur une validation a ECLAIRAGE INVERSE :
on eclaircit artificiellement les visages reels et on assombrit les attaques.
Dans les sessions d'entrainement de NUAA, les visages reels sont plus sombres que
les photos imprimees ; un modele qui a appris le raccourci "clair = attaque"
s'effondre sur cette validation inversee. C'est un test de robustesse qui ne
consomme pas le jeu de test.

Usage : python -m src.experiments
Sortie : reports/experiments.md (+ un run MLflow par configuration)
"""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import config
from src.models.mobilenet import build_model, predict_proba
from src.preprocessing.dataset import NuaaDataset
from src.train import TrainConfig, fit, setup_mlflow
from src.utils.metrics import auc, eer_threshold

FINETUNE_EPOCHS = 6
CANDIDATES = [
    TrainConfig(name="E1-baseline", strong_aug=False, grayscale=False, photo_norm=False,
                finetune_epochs=FINETUNE_EPOCHS),
    TrainConfig(name="E2-photo-norm", strong_aug=False, grayscale=False, photo_norm=True,
                finetune_epochs=FINETUNE_EPOCHS),
    TrainConfig(name="E3-photo-norm-strong-aug", strong_aug=True, grayscale=False, photo_norm=True,
                finetune_epochs=FINETUNE_EPOCHS),
]


def reverse_illumination(ds: NuaaDataset) -> NuaaDataset:
    """Eclaircit les visages reels et assombrit les attaques (inverse le biais des donnees)."""
    imgs = ds.images.astype(np.float32)
    real = (ds.labels.numpy() == 0)[:, None, None, None]
    imgs = np.where(real, imgs * 1.25 + 30, imgs * 0.8 - 10)
    ds.images = np.clip(imgs, 0, 255).astype(np.uint8)
    return ds


def robustness_auc(cfg: TrainConfig, ckpt_path, val_rows) -> tuple[float, float]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    ds = reverse_illumination(NuaaDataset(val_rows, train=False, grayscale=cfg.grayscale,
                                          photo_norm=cfg.photo_norm))
    p, y = predict_proba(model, DataLoader(ds, batch_size=128))
    return auc(p.numpy(), y.numpy()), eer_threshold(p.numpy(), y.numpy())[0]


def main() -> None:
    torch.set_num_threads(os.cpu_count() or 4)
    from src.preprocessing.build_manifest import build_rows

    rows = build_rows(config.NUAA_DIR, protocol="session")     # independant du protocole final
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    out_dir = config.MODELS_DIR / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_mlflow(f"{config.MLFLOW_EXPERIMENT}-cross-session")

    results = []
    for cfg in CANDIDATES:
        ckpt = out_dir / f"{cfg.name}.pt"
        best = fit(cfg, train_rows, val_rows, ckpt)
        rob_auc, rob_eer = robustness_auc(cfg, ckpt, val_rows)
        print(f"[{cfg.name}] eclairage inverse : AUC={rob_auc:.4f} EER={rob_eer:.4f}", flush=True)
        results.append((cfg, best, rob_auc, rob_eer))

    lines = [
        "# Experiences - choix de la configuration (sans le jeu de test)",
        "",
        f"Entrainement : session 01 ({len(train_rows)} images) - validation : session 02 "
        f"({len(val_rows)} images).",
        "",
        "| Configuration | Normalisation photo | Aug. forte | Meilleure epoque | AUC val | EER val "
        "| AUC val eclairage inverse | EER val eclairage inverse |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cfg, b, ra, re in results:
        lines.append(f"| {cfg.name} | {'oui' if cfg.photo_norm else 'non'} | "
                     f"{'oui' if cfg.strong_aug else 'non'} | {b['epoch']} | {b['auc']:.4f} | "
                     f"{b['eer'] * 100:.2f} % | {ra:.4f} | {re * 100:.2f} % |")
    lines += ["", "Regle de choix : la meilleure EER sur la validation a eclairage inverse "
              "(robustesse), a performance normale comparable."]
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "experiments.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
