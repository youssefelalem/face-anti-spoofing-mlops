"""Pretraitement d'un visage, partage par l'entrainement ET l'application.

Volontairement ecrit en NumPy/OpenCV (sans PyTorch) : l'application en
production n'embarque pas PyTorch, mais doit appliquer exactement la meme
transformation que celle vue pendant l'entrainement.
"""

from __future__ import annotations

import cv2
import numpy as np

# Valeurs dupliquees de src/config.py pour que ce module reste autonome
# (il est copie tel quel dans l'image Docker de l'application).
# tests/test_preprocess.py verifie qu'elles restent identiques.
IMG_SIZE = 128
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_face(face_bgr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Redimensionne un visage BGR uint8 en carre size x size."""
    if face_bgr.ndim != 3 or face_bgr.shape[2] != 3:
        raise ValueError(f"image BGR attendue (H, W, 3), recu {face_bgr.shape}")
    return cv2.resize(face_bgr, (size, size), interpolation=cv2.INTER_AREA)


def normalize(face_bgr_resized: np.ndarray) -> np.ndarray:
    """BGR uint8 (H, W, 3) -> RGB float32 normalise ImageNet, format (3, H, W)."""
    rgb = cv2.cvtColor(face_bgr_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


PHOTO_MEAN = 128.0
PHOTO_STD = 48.0


def photo_normalize(face_bgr: np.ndarray) -> np.ndarray:
    """Neutralise la luminosite et le contraste globaux du visage.

    Dans NUAA, les visages reels des sessions d'entrainement sont sombres et les
    photos imprimees claires, alors que c'est l'inverse dans la session de test :
    sans cette normalisation, le modele apprend le raccourci "clair = attaque".
    On ramene la luminance de chaque visage a une moyenne et un ecart-type fixes,
    ce qui force le modele a s'appuyer sur la texture et la nettete.
    """
    x = face_bgr.astype(np.float32)
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    x = (x - gray.mean()) / (gray.std() + 1e-6) * PHOTO_STD + PHOTO_MEAN
    return np.clip(x, 0, 255).astype(np.uint8)


def to_gray3(face_bgr: np.ndarray) -> np.ndarray:
    """Niveaux de gris replique sur 3 canaux (le reseau attend toujours 3 canaux)."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def prepare(face_resized: np.ndarray, photo_norm: bool = False, grayscale: bool = False) -> np.ndarray:
    """Etapes optionnelles communes entrainement / inference, appliquees apres redimensionnement."""
    if photo_norm:
        face_resized = photo_normalize(face_resized)
    if grayscale:
        face_resized = to_gray3(face_resized)
    return face_resized


def preprocess(face_bgr: np.ndarray, grayscale: bool = False, photo_norm: bool = False) -> np.ndarray:
    """Visage BGR uint8 de taille quelconque -> tenseur (1, 3, IMG_SIZE, IMG_SIZE)."""
    face = prepare(resize_face(face_bgr), photo_norm=photo_norm, grayscale=grayscale)
    return normalize(face)[None, ...]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
