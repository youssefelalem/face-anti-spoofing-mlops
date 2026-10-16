"""Moteur d'inference : detection du visage + fusion CNN (ONNX) et LBP-SVM.

- CNN MobileNetV2 (ONNX Runtime) : probabilite d'attaque, calibree (Platt).
- LBP + SVM (scikit-learn) : probabilite d'attaque calibree.
- Score final = moyenne des deux ; decision au seuil fixe sur la validation.
Si les fichiers du modele LBP / de fusion sont absents, seul le CNN est utilise.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.utils.face_detect import FaceBox, FaceDetector, crop
from src.utils.preprocess import preprocess, softmax

EPS = 1e-6


@dataclass
class Prediction:
    face_found: bool
    box: FaceBox | None = None
    spoof_score: float | None = None
    label: str | None = None
    latency_ms: float = 0.0
    details: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "face_found": self.face_found,
            "box": self.box.as_dict() if self.box else None,
            "spoof_score": self.spoof_score,
            "label": self.label,
            "latency_ms": round(self.latency_ms, 1),
            "details": self.details,
        }


class AntiSpoofingEngine:
    def __init__(self, model_path: Path, meta_path: Path, lbp_path: Path | None = None,
                 fusion_path: Path | None = None) -> None:
        self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        self.classes = self.meta["classes"]
        inp = self.meta.get("input", {})
        self.grayscale = bool(inp.get("grayscale", False))
        self.photo_norm = bool(inp.get("photo_norm", False))
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.detector = FaceDetector()

        self.lbp = self.fusion = None
        if lbp_path and fusion_path and Path(lbp_path).is_file() and Path(fusion_path).is_file():
            import joblib  # dependance chargee seulement si la fusion est disponible

            self.lbp = joblib.load(lbp_path)
            self.fusion = json.loads(Path(fusion_path).read_text(encoding="utf-8"))
        self.threshold = float(self.fusion["threshold"] if self.fusion else self.meta["threshold"])
        self.mode = "fusion CNN + LBP-SVM" if self.fusion else "CNN"

    def cnn_score(self, face_bgr: np.ndarray) -> float:
        x = preprocess(face_bgr, grayscale=self.grayscale, photo_norm=self.photo_norm)
        logits = self.session.run(None, {self.input_name: x})[0]
        return float(softmax(logits)[0, 1])

    # Compatibilite : score "brut" du CNN
    spoof_score = cnn_score

    def lbp_score(self, face_bgr: np.ndarray) -> float:
        from src.utils.lbp import lbp_features

        d = self.lbp["svm"].decision_function(lbp_features(face_bgr)[None, :])
        return float(self.lbp["calibrator"].predict_proba(d[:, None])[0, 1])

    def score(self, face_bgr: np.ndarray) -> tuple[float, dict]:
        cnn = self.cnn_score(face_bgr)
        if not self.fusion:
            return cnn, {"cnn": round(cnn, 4)}
        cal = self.fusion["cnn_calibration"]
        p = float(np.clip(cnn, EPS, 1 - EPS))
        cnn_cal = 1 / (1 + np.exp(-(cal["coef"] * np.log(p / (1 - p)) + cal["intercept"])))
        lbp = self.lbp_score(face_bgr)
        return float((cnn_cal + lbp) / 2), {"cnn": round(float(cnn_cal), 4), "lbp": round(lbp, 4)}

    def predict(self, image_bgr: np.ndarray) -> Prediction:
        t0 = time.perf_counter()
        box = self.detector.largest_face(image_bgr)
        if box is None:
            return Prediction(face_found=False, latency_ms=(time.perf_counter() - t0) * 1000)
        score, details = self.score(crop(image_bgr, box))
        label = self.classes[1] if score >= self.threshold else self.classes[0]
        return Prediction(True, box, score, label, (time.perf_counter() - t0) * 1000, details)
