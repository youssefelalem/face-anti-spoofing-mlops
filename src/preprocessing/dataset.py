"""Dataset PyTorch pour NUAA, avec cache memoire et augmentation de donnees.

Les images sont decodees et redimensionnees UNE seule fois puis gardees en
memoire (~0.8 Go pour tout NUAA) : chaque epoque est alors tres rapide sur CPU.

L'augmentation evite flou et bruit, car ce sont justement des indices d'une
photo imprimee (on ne veut pas apprendre au modele a les ignorer). En revanche
l'augmentation "photometrique" (luminosite, contraste, couleur, gamma) simule
les changements d'eclairage entre sessions d'enregistrement.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src import config
from src.utils.preprocess import IMG_SIZE, normalize, prepare

# Taille de cache un peu plus grande que l'entree du reseau, pour le recadrage aleatoire.
CACHE_SIZE = 144


def image_path(row: dict, nuaa_dir: Path = config.NUAA_DIR) -> Path:
    """Chemin d'une image du manifeste, selon sa source (NUAA ou webcam)."""
    base = config.WEBCAM_DIR if row.get("source") == "webcam" else nuaa_dir
    return base / row["path"]


def load_images(rows: list[dict], nuaa_dir: Path, size: int) -> np.ndarray:
    images = np.empty((len(rows), size, size, 3), dtype=np.uint8)
    for i, r in enumerate(rows):
        img = cv2.imread(str(image_path(r, nuaa_dir)), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(image_path(r, nuaa_dir))
        images[i] = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return images


def photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simule un autre eclairage / une autre camera : gamma, saturation, dominante de couleur."""
    x = img.astype(np.float32) / 255.0
    x = x ** rng.uniform(0.6, 1.6)                                   # gamma
    gray = x.mean(axis=2, keepdims=True)
    x = gray + (x - gray) * rng.uniform(0.5, 1.5)                    # saturation
    x = x * rng.uniform(0.85, 1.15, size=(1, 1, 3))                  # dominante de couleur
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def augment(img: np.ndarray, rng: np.random.Generator, strong: bool = False) -> np.ndarray:
    """Recadrage aleatoire, miroir, petite rotation, luminosite/contraste (+ photometrie forte)."""
    h, w = img.shape[:2]
    scale = rng.uniform(0.85, 1.0)
    side = int(h * scale)
    y0 = rng.integers(0, h - side + 1)
    x0 = rng.integers(0, w - side + 1)
    img = img[y0 : y0 + side, x0 : x0 + side]

    if rng.random() < 0.5:
        img = img[:, ::-1]

    angle = rng.uniform(-10, 10)
    m = cv2.getRotationMatrix2D((side / 2, side / 2), angle, 1.0)
    img = cv2.warpAffine(np.ascontiguousarray(img), m, (side, side), borderMode=cv2.BORDER_REFLECT)

    spread = 0.35 if strong else 0.2
    alpha = rng.uniform(1 - spread, 1 + spread)          # contraste
    beta = rng.uniform(-40, 40) if strong else rng.uniform(-20, 20)   # luminosite
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if strong:
        img = photometric(img, rng)

    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


class NuaaDataset(Dataset):
    def __init__(self, rows: list[dict], train: bool, nuaa_dir: Path = config.NUAA_DIR,
                 seed: int = config.SEED, strong_aug: bool = False, grayscale: bool = False,
                 photo_norm: bool = False) -> None:
        self.labels = torch.tensor([r["label"] for r in rows], dtype=torch.long)
        self.train = train
        self.strong_aug = strong_aug
        self.grayscale = grayscale
        self.photo_norm = photo_norm
        # Evaluation : redimensionnement direct a IMG_SIZE, exactement comme l'application.
        self.images = load_images(rows, nuaa_dir, CACHE_SIZE if train else IMG_SIZE)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        img = augment(self.images[i], self.rng, self.strong_aug) if self.train else self.images[i]
        img = prepare(img, photo_norm=self.photo_norm, grayscale=self.grayscale)
        return torch.from_numpy(normalize(img)), self.labels[i]
