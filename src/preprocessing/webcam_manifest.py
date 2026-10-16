"""Manifeste des donnees webcam collectees par l'etudiant (app/collect.py).

Entree : data/raw/webcam/<session>/<attaque>/<sujet>_<numero>.jpg
Sortie : data/processed/webcam_manifest.csv

- Nettoyage : un recadrage est garde seulement si le detecteur retrouve un visage
  dedans (elimine les faux positifs du detecteur, ex. motifs du mur).
- Decoupage : les sessions WEBCAM_TEST_SESSIONS servent UNIQUEMENT au test ;
  dans les autres sessions, les 20 % dernieres images de chaque clip vont en
  validation (decoupage temporel : pas d'images quasi identiques des deux cotes).

Usage : python -m src.preprocessing.webcam_manifest
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from src import config
from src.utils.face_detect import FaceDetector

FIELDS = ["path", "label", "subject", "session", "split", "attack", "source"]
ATTACK_LABEL = {"real": 0, "print": 1, "screen": 1}
VAL_FRACTION = 0.2


def has_face(detector: FaceDetector, path: Path) -> bool:
    img = cv2.imread(str(path))
    if img is None:
        return False
    h, w = img.shape[:2]
    padded = cv2.copyMakeBorder(img, h // 2, h // 2, w // 2, w // 2, cv2.BORDER_CONSTANT, value=0)
    return detector.largest_face(padded) is not None


def build_rows(root: Path = config.WEBCAM_DIR, test_sessions=config.WEBCAM_TEST_SESSIONS,
               clean: bool = True) -> tuple[list[dict], int]:
    detector = FaceDetector() if clean else None
    clips = defaultdict(list)
    for p in sorted(root.glob("*/*/*.jpg")):
        session, attack = p.parent.parent.name, p.parent.name
        if attack in ATTACK_LABEL:
            clips[(session, attack, p.stem.rsplit("_", 1)[0])].append(p)
    rows, rejected = [], 0
    for (session, attack, subject), paths in sorted(clips.items()):
        kept = [p for p in paths if not clean or has_face(detector, p)]
        rejected += len(paths) - len(kept)
        n_val = int(round(len(kept) * VAL_FRACTION))
        for i, p in enumerate(kept):
            if session in test_sessions:
                split = "test"
            else:
                split = "val" if i >= len(kept) - n_val else "train"
            rows.append({"path": p.relative_to(root).as_posix(), "label": ATTACK_LABEL[attack],
                         "subject": subject, "session": session, "split": split,
                         "attack": attack, "source": "webcam"})
    return rows, rejected


def read_webcam_manifest(path: Path = config.WEBCAM_MANIFEST_CSV) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return [{**r, "label": int(r["label"])} for r in csv.DictReader(f)]


def main() -> None:
    rows, rejected = build_rows()
    config.WEBCAM_MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    with config.WEBCAM_MANIFEST_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter((r["split"], r["session"], r["attack"]) for r in rows)
    print(f"Manifeste webcam : {config.WEBCAM_MANIFEST_CSV} ({len(rows)} images, "
          f"{rejected} rejetees par le nettoyage)")
    for (split, session, attack), n in sorted(counts.items()):
        print(f"  {split:5s} {session:4s} {attack:6s} {n:5d}")


if __name__ == "__main__":
    main()
