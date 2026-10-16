"""API FastAPI + interface web de demonstration.

Routes :
- GET  /            interface webcam (app/static/index.html)
- GET  /health      etat du service et version du modele
- GET  /model       metadonnees du modele (seuil, metriques)
- POST /predict     image (JPEG/PNG) -> {face_found, box, spoof_score, label, latency_ms}

Lancement : uvicorn app.main:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.inference import AntiSpoofingEngine

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
MODEL_PATH = Path(os.environ.get("FAS_MODEL_PATH", ROOT / "models/saved/mobilenetv2_fas.onnx"))
META_PATH = Path(os.environ.get("FAS_META_PATH", ROOT / "models/saved/model_meta.json"))
LBP_PATH = Path(os.environ.get("FAS_LBP_PATH", ROOT / "models/saved/lbp_svm.joblib"))
FUSION_PATH = Path(os.environ.get("FAS_FUSION_PATH", ROOT / "models/saved/fusion.json"))
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


COLLECT_ENABLED = os.environ.get("FAS_ENABLE_COLLECT") == "1"


MODEL_FILES = ("mobilenetv2_fas.onnx", "model_meta.json", "lbp_svm.joblib", "fusion.json")


def fetch_models_from_hub() -> None:
    """Deploiement : telecharge les modeles depuis un depot Hugging Face PRIVE.

    Active seulement si FAS_HF_MODEL_REPO est defini (ex. sur un Space) ; le jeton de
    lecture est fourni par la variable d'environnement HF_TOKEN (secret du Space). Les
    poids ne sont ainsi jamais telechargeables depuis le depot public de l'application.
    """
    repo = os.environ.get("FAS_HF_MODEL_REPO")
    if not repo or MODEL_PATH.is_file():
        return
    from huggingface_hub import hf_hub_download

    for name in MODEL_FILES:
        hf_hub_download(repo, name, local_dir=MODEL_PATH.parent, token=os.environ.get("HF_TOKEN"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    fetch_models_from_hub()
    # Sans modele, le service demarre quand meme (utile pour la collecte de donnees) :
    # les routes de prediction repondent alors 503.
    app.state.engine = (AntiSpoofingEngine(MODEL_PATH, META_PATH, LBP_PATH, FUSION_PATH)
                        if MODEL_PATH.is_file() else None)
    yield


app = FastAPI(title="Face Anti-Spoofing API", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
if COLLECT_ENABLED:
    from app.collect import router as collect_router

    app.include_router(collect_router)


def engine() -> AntiSpoofingEngine:
    if app.state.engine is None:
        raise HTTPException(status_code=503, detail="modele non disponible")
    return app.state.engine


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    if app.state.engine is None:
        return {"status": "no-model", "model": None, "version": None, "collect": COLLECT_ENABLED}
    meta = app.state.engine.meta
    return {"status": "ok", "model": meta["model_name"], "version": meta["version"]}


@app.get("/model")
def model_info() -> dict:
    eng = engine()
    meta = eng.meta
    final = meta.get("final_test") or {}
    # Metriques affichees : celles du modele reellement utilise, sur le test le plus proche de
    # l'usage reel (webcam du projet) s'il existe, sinon NUAA.
    test_set = "webcam" if "webcam" in final else "nuaa"
    if final and eng.fusion:
        test = final[test_set]["Fusion"]
    elif final:
        test = final[test_set]["CNN (MobileNetV2)"]
    else:
        test, test_set = meta.get("test") or {}, "nuaa"
    at_thr = test.get("at_validation_threshold") or {}
    return {
        "model": meta["model_name"],
        "version": meta["version"],
        "mode": eng.mode,
        "threshold": eng.threshold,
        "input_size": meta["input"]["size"],
        "test_set": test_set,
        "test": {k: at_thr.get(k) for k in ("accuracy", "APCER", "BPCER", "ACER")}
        | {"auc": test.get("auc"), "eer": test.get("eer")},
    }


@app.post("/predict")
async def predict(file: Annotated[UploadFile, File()]) -> dict:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image trop volumineuse (max 5 Mo)")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="fichier image illisible")
    return engine().predict(image).as_dict()
