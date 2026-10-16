"""Detection de visage pour l'application (OpenCV, detecteur Haar integre).

Le recadrage reproduit approximativement celui des images NUAA
"Detectedface" : un carre centre sur le visage, legerement plus large que
la boite renvoyee par le detecteur.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Cadrage calibre sur 200 images NUAA (Haar relance sur les recadrages officiels) :
# le recadrage NUAA a la meme taille que la boite Haar (rapport median 0.98), mais
# le visage y est centre a 41 % de la hauteur -> on decale le carre vers le bas.
CROP_MARGIN = 1.0
CROP_SHIFT_Y = 0.09
MIN_FACE_PX = 60


@dataclass(frozen=True)
class FaceBox:
    x: int
    y: int
    w: int
    h: int

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


class FaceDetector:
    def __init__(self) -> None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise RuntimeError(f"impossible de charger le detecteur Haar : {path}")

    def largest_face(self, image_bgr: np.ndarray) -> FaceBox | None:
        """Renvoie la boite (carree, avec marge) du plus grand visage, ou None."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(MIN_FACE_PX, MIN_FACE_PX)
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return square_with_margin(int(x), int(y), int(w), int(h), image_bgr.shape)


def square_with_margin(x: int, y: int, w: int, h: int, shape) -> FaceBox:
    """Rend la boite carree (cadrage NUAA : marge + decalage vertical) et la borne a l'image."""
    img_h, img_w = shape[:2]
    side = int(round(max(w, h) * CROP_MARGIN))
    cx, cy = x + w / 2, y + h / 2 + CROP_SHIFT_Y * side
    side = min(side, img_w, img_h)
    x0 = int(round(min(max(cx - side / 2, 0), img_w - side)))
    y0 = int(round(min(max(cy - side / 2, 0), img_h - side)))
    return FaceBox(x0, y0, side, side)


def crop(image_bgr: np.ndarray, box: FaceBox) -> np.ndarray:
    return image_bgr[box.y : box.y + box.h, box.x : box.x + box.w]
