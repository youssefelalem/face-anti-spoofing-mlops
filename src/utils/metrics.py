"""Metriques d'evaluation anti-spoofing (norme ISO/IEC 30107-3).

Convention : score = probabilite d'attaque (spoof). Une image est rejetee
comme attaque si score >= seuil.

- APCER : proportion d'ATTAQUES acceptees a tort comme visages reels
          (faille de securite).
- BPCER : proportion de VRAIS visages rejetes a tort comme attaques
          (gene pour l'utilisateur).
- ACER  : moyenne des deux = (APCER + BPCER) / 2.
- EER   : point ou APCER = BPCER.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def pad_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred_spoof = scores >= threshold
    attacks, bona = labels == 1, labels == 0
    apcer = float(np.mean(~pred_spoof[attacks])) if attacks.any() else 0.0
    bpcer = float(np.mean(pred_spoof[bona])) if bona.any() else 0.0
    tp = int(np.sum(pred_spoof & attacks))
    tn = int(np.sum(~pred_spoof & bona))
    fp = int(np.sum(pred_spoof & bona))
    fn = int(np.sum(~pred_spoof & attacks))
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(len(labels), 1),
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": (apcer + bpcer) / 2,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def eer_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Renvoie (EER, seuil a l'EER). Le seuil se choisit sur la VALIDATION, jamais sur le test."""
    fpr, tpr, thr = roc_curve(labels, scores)        # positif = spoof
    fnr = 1 - tpr                                    # fpr = BPCER, fnr = APCER
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[i] + fnr[i]) / 2)
    threshold = float(min(thr[i], 1.0))              # roc_curve peut renvoyer +inf
    return eer, threshold


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))
