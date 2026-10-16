import numpy as np
import pytest

from src.utils.metrics import auc, eer_threshold, pad_metrics


def test_pad_metrics_known_values():
    # 4 vrais visages (0), 4 attaques (1)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.6, 0.3, 0.9, 0.8, 0.4, 0.7])
    m = pad_metrics(scores, labels, threshold=0.5)
    assert m["BPCER"] == pytest.approx(1 / 4)      # 0.6 : vrai visage rejete
    assert m["APCER"] == pytest.approx(1 / 4)      # 0.4 : attaque acceptee
    assert m["ACER"] == pytest.approx(0.25)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["confusion"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 3}


def test_perfect_separation():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.05, 0.1, 0.9, 0.95])
    eer, thr = eer_threshold(scores, labels)
    assert eer == pytest.approx(0.0)
    assert pad_metrics(scores, labels, thr)["ACER"] == pytest.approx(0.0)
    assert auc(scores, labels) == pytest.approx(1.0)


def test_eer_threshold_is_finite():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 500)
    scores = np.clip(labels * 0.3 + rng.normal(0.35, 0.2, 500), 0, 1)
    eer, thr = eer_threshold(scores, labels)
    assert 0.0 <= eer <= 0.5
    assert np.isfinite(thr) and 0.0 <= thr <= 1.0
