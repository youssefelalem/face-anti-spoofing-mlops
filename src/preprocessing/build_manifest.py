"""Construit le manifeste des donnees a partir des fichiers d'index officiels NUAA.

Sortie : data/processed/manifest.csv (path, label, subject, session, split).

Decoupage :
- test  = index officiels *_test_face.txt  (session 03, jamais vue a l'entrainement)
- val   = index officiels *_train_face.txt :
          protocole "session" -> session VAL_SESSION (02) ; protocole "subject" -> sujets VAL_SUBJECTS
- train = le reste des index officiels *_train_face.txt

Usage : python -m src.preprocessing.build_manifest
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from src import config

INDEX_FILES = {
    ("real", "train"): ("ClientFace", "client_train_face.txt"),
    ("real", "test"): ("ClientFace", "client_test_face.txt"),
    ("spoof", "train"): ("ImposterFace", "imposter_train_face.txt"),
    ("spoof", "test"): ("ImposterFace", "imposter_test_face.txt"),
}
FIELDS = ["path", "label", "subject", "session", "split"]


def parse_index(nuaa_dir: Path, folder: str, index_name: str) -> list[tuple[str, str, str]]:
    """Lit un fichier d'index NUAA -> [(chemin_relatif, sujet, session)]."""
    rows = []
    for line in (nuaa_dir / index_name).read_text(encoding="latin-1").splitlines():
        if not line.strip():
            continue
        rel = line.split()[0].replace("\\", "/")          # ex. 0001/0001_00_00_01_0.jpg
        subject, _glasses, _pos, session, _num = Path(rel).stem.split("_")
        rows.append((f"{folder}/{rel}", subject, session))
    return rows


def build_rows(nuaa_dir: Path, protocol: str = config.VAL_PROTOCOL,
               val_subjects=config.VAL_SUBJECTS, val_session: str = config.VAL_SESSION) -> list[dict]:
    if protocol not in ("subject", "session"):
        raise ValueError(f"protocole de validation inconnu : {protocol}")
    rows = []
    for (cls, official_split), (folder, index_name) in INDEX_FILES.items():
        for rel, subject, session in parse_index(nuaa_dir, folder, index_name):
            if official_split == "test":
                split = "test"
            elif protocol == "session":
                split = "val" if session == val_session else "train"
            else:
                split = "val" if subject in val_subjects else "train"
            rows.append(
                {"path": rel, "label": config.LABELS[cls], "subject": subject,
                 "session": session, "split": split}
            )
    return rows


def write_manifest(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path = config.MANIFEST_CSV) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [{**r, "label": int(r["label"])} for r in csv.DictReader(f)]


def main() -> None:
    rows = build_rows(config.NUAA_DIR)
    write_manifest(rows, config.MANIFEST_CSV)
    counts = Counter((r["split"], config.CLASS_NAMES[r["label"]]) for r in rows)
    print(f"Manifeste ecrit : {config.MANIFEST_CSV} ({len(rows)} images)")
    for split in ("train", "val", "test"):
        print(f"  {split:5s} real={counts[(split, 'real')]:5d}  spoof={counts[(split, 'spoof')]:5d}")


if __name__ == "__main__":
    main()
