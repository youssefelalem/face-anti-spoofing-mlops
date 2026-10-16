"""Tests de la page de collecte (sans modele : le service doit demarrer quand meme)."""

import importlib

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FAS_ENABLE_COLLECT", "1")
    monkeypatch.setenv("FAS_MODEL_PATH", str(tmp_path / "absent.onnx"))
    import app.collect
    import app.main
    importlib.reload(app.collect)
    importlib.reload(app.main)
    monkeypatch.setattr(app.collect, "WEBCAM_DIR", tmp_path / "webcam")
    with TestClient(app.main.app) as c:
        yield c


def jpeg(img):
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_service_starts_without_model(client):
    assert client.get("/health").json()["status"] == "no-model"
    assert client.get("/model").status_code == 503


def test_collect_page_and_stats(client):
    assert client.get("/collect").status_code == 200
    assert client.get("/collect/stats").json() == {"counts": {}}


def test_collect_frame_without_face_saves_nothing(client):
    blank = np.full((240, 320, 3), 120, np.uint8)
    r = client.post("/collect/frame", files={"file": ("f.jpg", jpeg(blank), "image/jpeg")},
                    data={"session": "s1", "attack": "real", "subject": "alice"})
    assert r.status_code == 200 and r.json() == {"face_found": False}
    assert client.get("/collect/stats").json() == {"counts": {}}


@pytest.mark.parametrize("fields", [
    {"session": "s1", "attack": "mask", "subject": "alice"},
    {"session": "../etc", "attack": "real", "subject": "alice"},
])
def test_collect_rejects_bad_fields(client, fields):
    blank = np.zeros((10, 10, 3), np.uint8)
    r = client.post("/collect/frame", files={"file": ("f.jpg", jpeg(blank), "image/jpeg")}, data=fields)
    assert r.status_code == 400
