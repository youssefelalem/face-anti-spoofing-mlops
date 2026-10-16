"""Descripteurs de texture LBP (Local Binary Patterns), sans dependance lourde.

Partage par l'entrainement (src/baseline_lbp_svm.py) et l'application.
Une photo imprimee ou un ecran perd les micro-textures de la peau : les LBP
decrivent ces micro-textures (Maatta et al., 2011).
"""

from __future__ import annotations

import cv2
import numpy as np

SIZE = 64
N_BINS = 59


def _uniform_table(p: int = 8) -> np.ndarray:
    """Code LBP -> indice d'histogramme (58 motifs uniformes + 1 bac 'non uniforme')."""
    table = np.full(2**p, p * (p - 1) + 2, dtype=np.int32)
    k = 0
    for code in range(2**p):
        bits = [(code >> i) & 1 for i in range(p)]
        if sum(bits[i] != bits[(i + 1) % p] for i in range(p)) <= 2:
            table[code] = k
            k += 1
    return table


UNIFORM = _uniform_table()


def lbp_codes(gray: np.ndarray, radius: int) -> np.ndarray:
    """LBP 8 voisins de rayon donne (voisins diagonaux arrondis au pixel le plus proche)."""
    g = gray.astype(np.int16)
    r = radius
    c = g[r:-r, r:-r]
    d = int(round(r / np.sqrt(2)))
    offsets = [(-r, 0), (-d, d), (0, r), (d, d), (r, 0), (d, -d), (0, -r), (-d, -d)]
    code = np.zeros_like(c, dtype=np.int32)
    h, w = g.shape
    for bit, (dy, dx) in enumerate(offsets):
        nb = g[r + dy : h - r + dy, r + dx : w - r + dx]
        code |= (nb >= c).astype(np.int32) << bit
    return UNIFORM[code]


def _hist(codes: np.ndarray) -> np.ndarray:
    h = np.bincount(codes.ravel(), minlength=N_BINS).astype(np.float32)
    return h / max(h.sum(), 1)


def lbp_features(face_bgr: np.ndarray) -> np.ndarray:
    """Visage BGR -> vecteur de 649 valeurs (LBP 8,1 global + 3x3 regions, LBP 8,2 global)."""
    small = cv2.resize(face_bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    c1 = lbp_codes(gray, 1)
    feats = [_hist(c1), _hist(lbp_codes(gray, 2))]
    h, w = c1.shape
    sy, sx = h // 3, w // 3
    for i in range(3):
        for j in range(3):
            feats.append(_hist(c1[i * sy : (i + 1) * sy, j * sx : (j + 1) * sx]))
    return np.concatenate(feats)
