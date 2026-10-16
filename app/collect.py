"""Collecte de donnees webcam propres au projet (active seulement si FAS_ENABLE_COLLECT=1).

Les visages sont recadres EXACTEMENT comme a l'inference (meme detecteur, meme
cadrage) puis enregistres localement dans data/raw/webcam/ (jamais dans Git).

Arborescence : data/raw/webcam/<session>/<attaque>/<sujet>_<numero>.jpg
avec <attaque> dans {real, print, screen}.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from src.utils.face_detect import FaceDetector, crop

ROOT = Path(__file__).resolve().parents[1]
WEBCAM_DIR = ROOT / "data" / "raw" / "webcam"
STATIC = Path(__file__).resolve().parent / "static"
ATTACKS = ("real", "print", "screen")
SAVE_SIZE = 256
SAFE = re.compile(r"^[A-Za-z0-9_-]{1,24}$")

router = APIRouter(prefix="/collect", tags=["collecte"])
_detector: FaceDetector | None = None


def detector() -> FaceDetector:
    global _detector
    if _detector is None:
        _detector = FaceDetector()
    return _detector


def counts() -> dict:
    c = Counter()
    for p in WEBCAM_DIR.glob("*/*/*.jpg"):
        session, attack = p.parent.parent.name, p.parent.name
        subject = p.stem.rsplit("_", 1)[0]
        c[f"{session}/{attack}/{subject}"] += 1
    return dict(sorted(c.items()))


@router.get("", include_in_schema=False)
def page() -> FileResponse:
    return FileResponse(STATIC / "collect.html")


@router.get("/stats")
def stats() -> dict:
    return {"counts": counts()}


@router.post("/frame")
async def frame(
    file: Annotated[UploadFile, File()],
    session: Annotated[str, Form()],
    attack: Annotated[str, Form()],
    subject: Annotated[str, Form()],
) -> dict:
    if attack not in ATTACKS:
        raise HTTPException(400, f"attaque inconnue (attendu : {', '.join(ATTACKS)})")
    if not (SAFE.match(session) and SAFE.match(subject)):
        raise HTTPException(400, "session/sujet : lettres, chiffres, - et _ uniquement")
    image = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "image illisible")
    box = detector().largest_face(image)
    if box is None:
        return {"face_found": False}
    out_dir = WEBCAM_DIR / session / attack
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(list(out_dir.glob(f"{subject}_*.jpg")))
    face = cv2.resize(crop(image, box), (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_dir / f"{subject}_{n:05d}.jpg"), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {"face_found": True, "box": box.as_dict(), "saved": n + 1}
