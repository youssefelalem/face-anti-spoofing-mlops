"""Tests de l'API FastAPI avec un modele ONNX factice (aucune donnee ni vrai modele requis)."""

import importlib

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.utils.face_detect import square_with_margin


@pytest.fixture()
def client(dummy_model, monkeypatch):
    model_path, meta_path = dummy_model
    monkeypatch.setenv("FAS_MODEL_PATH", str(model_path))
    monkeypatch.setenv("FAS_META_PATH", str(meta_path))
    # Isole le test des vrais modeles eventuellement presents dans models/saved/
    monkeypatch.setenv("FAS_LBP_PATH", str(model_path.parent / "absent.joblib"))
    monkeypatch.setenv("FAS_FUSION_PATH", str(model_path.parent / "absent.json"))
    import app.main
    importlib.reload(app.main)                     # relit les variables d'environnement
    with TestClient(app.main.app) as c:
        yield c


def jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": "dummy", "version": "test"}


def test_model_info(client):
    body = client.get("/model").json()
    assert body["threshold"] == 0.5
    assert body["test"]["ACER"] == 0.025


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Face Anti-Spoofing" in r.text


def test_predict_without_face(client):
    blank = np.full((240, 320, 3), 127, np.uint8)
    r = client.post("/predict", files={"file": ("x.jpg", jpeg(blank), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["face_found"] is False
    assert body["label"] is None


def test_predict_rejects_invalid_image(client):
    r = client.post("/predict", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


def test_engine_score_is_probability(client):
    import app.main
    engine = app.main.app.state.engine
    face = np.random.default_rng(1).integers(0, 256, (150, 150, 3), dtype=np.uint8)
    assert 0.0 <= engine.spoof_score(face) <= 1.0


def test_fusion_engine_combines_cnn_and_lbp(dummy_model, tmp_path):
    """Avec un LBP-SVM et un fichier de fusion, le score est la moyenne des deux probabilites."""
    import json

    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    from app.inference import AntiSpoofingEngine
    from src.utils.lbp import lbp_features

    rng = np.random.default_rng(0)
    faces = [rng.integers(0, 256, (80, 80, 3), dtype=np.uint8) for _ in range(12)]
    x = np.stack([lbp_features(f) for f in faces])
    y = np.array([0, 1] * 6)
    svm = SVC().fit(x, y)
    cal = LogisticRegression().fit(svm.decision_function(x)[:, None], y)
    joblib.dump({"svm": svm, "calibrator": cal, "threshold": 0.5}, tmp_path / "lbp.joblib")
    (tmp_path / "fusion.json").write_text(json.dumps(
        {"cnn_calibration": {"coef": 1.0, "intercept": 0.0}, "threshold": 0.4}), encoding="utf-8")

    model_path, meta_path = dummy_model
    eng = AntiSpoofingEngine(model_path, meta_path, tmp_path / "lbp.joblib", tmp_path / "fusion.json")
    assert eng.mode.startswith("fusion") and eng.threshold == 0.4
    score, details = eng.score(faces[0])
    assert score == pytest.approx((details["cnn"] + details["lbp"]) / 2, abs=1e-3)
    assert 0.0 <= score <= 1.0


def test_lbp_features_shape_and_normalisation():
    from src.utils.lbp import lbp_features

    f = lbp_features(np.random.default_rng(2).integers(0, 256, (100, 100, 3), dtype=np.uint8))
    assert f.shape == (59 * 11,)
    assert np.allclose(f.reshape(11, 59).sum(axis=1), 1.0)


@pytest.mark.parametrize("box", [(0, 0, 50, 50), (600, 400, 100, 100), (300, 200, 80, 120)])
def test_square_with_margin_stays_inside_image(box):
    b = square_with_margin(*box, shape=(480, 640, 3))
    assert b.w == b.h
    assert 0 <= b.x and b.x + b.w <= 640
    assert 0 <= b.y and b.y + b.h <= 480
