"""Entrainement de MobileNetV2 (transfer learning) sur NUAA, suivi avec MLflow.

Deux phases :
1. Tete seule : l'extracteur ImageNet est gele, on n'apprend que la derniere couche.
2. Fine-tuning : le reseau est degele (entierement ou a partir d'un bloc donne),
   avec un learning rate 10x plus petit.

Le meilleur modele est choisi sur la VALIDATION (EER le plus bas), avec arret
anticipe (early stopping) pour eviter le sur-apprentissage.

Usage :
    python -m src.train                     # configuration retenue (src/config.py)
    python -m src.train --name essai --strong-aug --grayscale --freeze-blocks 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src import config
from src.models.mobilenet import build_model, count_parameters, predict_proba, set_backbone_trainable
from src.preprocessing.build_manifest import read_manifest
from src.preprocessing.dataset import NuaaDataset
from src.preprocessing.webcam_manifest import read_webcam_manifest
from src.utils.metrics import auc, eer_threshold, pad_metrics


@dataclass
class TrainConfig:
    name: str = "final"
    strong_aug: bool = config.STRONG_AUG
    grayscale: bool = config.GRAYSCALE
    photo_norm: bool = config.PHOTO_NORM            # neutralise luminosite / contraste par visage
    freeze_blocks: int = config.FREEZE_BLOCKS       # blocs de `features` restant geles en phase 2
    head_epochs: int = config.HEAD_EPOCHS
    finetune_epochs: int = config.FINETUNE_EPOCHS
    head_lr: float = config.HEAD_LR
    finetune_lr: float = config.FINETUNE_LR
    patience: int = config.EARLY_STOP_PATIENCE


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_trainable_from(model: nn.Module, freeze_blocks: int) -> None:
    """Gele les `freeze_blocks` premiers blocs de l'extracteur, degele les suivants."""
    for i, block in enumerate(model.features):
        for p in block.parameters():
            p.requires_grad = i >= freeze_blocks


def run_epoch(model, loader, criterion, optimizer, frozen_bn_blocks: int) -> float:
    model.train()
    for block in list(model.features)[:frozen_bn_blocks]:
        block.eval()                   # garde les statistiques BatchNorm d'ImageNet des blocs geles
    total, n = 0.0, 0
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.no_grad()
def evaluate(model, loader) -> dict:
    probs, labels = predict_proba(model, loader)
    p = probs.numpy()
    y = labels.numpy()
    eps = 1e-7
    loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    eer, thr = eer_threshold(p, y)
    return {"loss": loss, "auc": auc(p, y), "eer": eer, "threshold": thr,
            **{k: v for k, v in pad_metrics(p, y, thr).items() if k != "confusion"}}


def fit(cfg: TrainConfig, train_rows: list[dict], val_rows: list[dict], checkpoint: Path) -> dict:
    """Entraine un modele, sauvegarde le meilleur (selon la validation) et renvoie ses metriques."""
    seed_everything(config.SEED)
    train_loader = DataLoader(
        NuaaDataset(train_rows, train=True, strong_aug=cfg.strong_aug, grayscale=cfg.grayscale,
                    photo_norm=cfg.photo_norm),
        batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        NuaaDataset(val_rows, train=False, grayscale=cfg.grayscale, photo_norm=cfg.photo_norm),
        batch_size=128, num_workers=0)
    model = build_model(pretrained=True)
    criterion = nn.CrossEntropyLoss()
    n_blocks = len(model.features)

    with mlflow.start_run(run_name=f"{cfg.name}-{time.strftime('%Y%m%d-%H%M%S')}") as run:
        mlflow.set_tag("validation_protocol", config.VAL_PROTOCOL)
        mlflow.log_params({**asdict(cfg), "architecture": "MobileNetV2 (ImageNet1K_V2)",
                           "img_size": config.IMG_SIZE, "batch_size": config.BATCH_SIZE,
                           "weight_decay": config.WEIGHT_DECAY, "seed": config.SEED,
                           "n_train": len(train_rows), "n_val": len(val_rows)})
        best = {"eer": float("inf"), "loss": float("inf")}
        epoch = 0
        phases = [("head", cfg.head_epochs, cfg.head_lr, n_blocks),
                  ("finetune", cfg.finetune_epochs, cfg.finetune_lr, cfg.freeze_blocks)]
        for phase, n_epochs, lr, frozen in phases:
            if phase == "head":
                set_backbone_trainable(model, False)
            else:
                set_trainable_from(model, frozen)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=config.WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_epochs, 1))
            print(f"\n== [{cfg.name}] phase {phase} : "
                  f"{count_parameters(model)['trainable']:,} parametres entrainables ==")
            patience = 0
            for _ in range(n_epochs):
                epoch += 1
                t0 = time.time()
                train_loss = run_epoch(model, train_loader, criterion, optimizer, frozen)
                scheduler.step()
                val = evaluate(model, val_loader)
                mlflow.log_metrics({"train_loss": train_loss, "val_loss": val["loss"],
                                    "val_auc": val["auc"], "val_eer": val["eer"],
                                    "val_acer": val["ACER"]}, step=epoch)
                print(f"epoque {epoch:2d} [{phase:8s}] train_loss={train_loss:.4f} "
                      f"val_loss={val['loss']:.4f} val_auc={val['auc']:.4f} "
                      f"val_eer={val['eer']:.4f} ({time.time() - t0:.0f}s)", flush=True)

                improved = (val["eer"] < best["eer"] - 1e-4 or
                            (abs(val["eer"] - best["eer"]) <= 1e-4 and val["loss"] < best["loss"]))
                if improved:
                    best = {**val, "epoch": epoch, "phase": phase}
                    torch.save({"state_dict": model.state_dict(), "val": best, "config": asdict(cfg),
                                "img_size": config.IMG_SIZE}, checkpoint)
                    patience = 0
                else:
                    patience += 1
                    if phase == "finetune" and patience >= cfg.patience:
                        print(f"Arret anticipe : pas d'amelioration depuis {patience} epoques.")
                        break

        mlflow.log_metrics({f"best_val_{k}": v for k, v in best.items() if isinstance(v, float)})
        mlflow.log_param("best_epoch", best["epoch"])
        best["mlflow_run_id"] = run.info.run_id
    print(f"[{cfg.name}] meilleur : epoque {best['epoch']} ({best['phase']}), "
          f"val_auc={best['auc']:.4f}, val_eer={best['eer']:.4f}, seuil={best['threshold']:.4f}")
    return best


def setup_mlflow(experiment: str) -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=config.MLFLOW_ARTIFACTS)
    mlflow.set_experiment(experiment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="final")
    parser.add_argument("--strong-aug", action=argparse.BooleanOptionalAction, default=config.STRONG_AUG)
    parser.add_argument("--grayscale", action=argparse.BooleanOptionalAction, default=config.GRAYSCALE)
    parser.add_argument("--photo-norm", action=argparse.BooleanOptionalAction, default=config.PHOTO_NORM)
    parser.add_argument("--freeze-blocks", type=int, default=config.FREEZE_BLOCKS)
    parser.add_argument("--finetune-epochs", type=int, default=config.FINETUNE_EPOCHS)
    args = parser.parse_args()
    cfg = TrainConfig(name=args.name, strong_aug=args.strong_aug, grayscale=args.grayscale,
                      photo_norm=args.photo_norm, freeze_blocks=args.freeze_blocks,
                      finetune_epochs=args.finetune_epochs)

    torch.set_num_threads(os.cpu_count() or 4)
    rows = read_manifest()
    webcam = read_webcam_manifest() if config.USE_WEBCAM else []
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows + webcam if r["split"] == "val"]
    webcam_train = [r for r in webcam if r["split"] == "train"]
    train_rows += webcam_train * config.WEBCAM_OVERSAMPLE
    print(f"train={len(train_rows)} (dont webcam {len(webcam_train)} x{config.WEBCAM_OVERSAMPLE})  "
          f"val={len(val_rows)}  protocole={config.VAL_PROTOCOL}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    setup_mlflow(config.MLFLOW_EXPERIMENT)
    best = fit(cfg, train_rows, val_rows, config.CHECKPOINT)
    summary = config.MODELS_DIR / "train_summary.json"
    summary.write_text(json.dumps({"mlflow_run_id": best["mlflow_run_id"], "config": asdict(cfg),
                                   "best_val": best}, indent=2), encoding="utf-8")
    print(f"Checkpoint : {config.CHECKPOINT}")


if __name__ == "__main__":
    main()
