"""Controle qualite des donnees contre data_contract.yaml (Data Quality gate).

Chaque regle du contrat devient un test explicite. Le script ecrit un rapport
JSON (reports/data_quality_report.json) et renvoie un code de sortie non nul si
une regle est violee : l'entrainement ne doit pas demarrer sur des donnees
non conformes.

Usage : python -m src.preprocessing.validate_data
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import yaml

from src import config
from src.preprocessing.build_manifest import read_manifest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(results: list, name: str, ok: bool, detail, severity: str = "error") -> None:
    results.append({"check": name, "passed": bool(ok), "severity": severity, "detail": detail})


BIAS_ACTIONS = {
    "luminance_moyenne": "normalisation photometrique par visage (src/utils/preprocess.py)",
    "saturation_moyenne": "entree en niveaux de gris (src/utils/preprocess.py)",
}


def check_bias(rows: list[dict], stats: dict, bias: dict, results: list) -> None:
    """Une caracteristique triviale (luminance, saturation) separe-t-elle seule real / spoof ?"""
    from sklearn.metrics import roc_auc_score

    train = [r for r in rows if r["split"] == "train" and r["path"] in stats]
    y = [r["label"] for r in train]
    for feature in bias["features"]:
        x = [stats[r["path"]][feature] for r in train]
        score = roc_auc_score(y, x)
        separability = max(score, 1 - score)        # sens de la correlation indifferent
        mean = {c: round(sum(v for v, lab in zip(x, y, strict=True) if lab == c) / max(y.count(c), 1), 1)
                for c in (0, 1)}
        check(results, f"biais_{feature}", separability <= bias["max_auc"],
              {"auc_feature_seule": round(separability, 3), "seuil": bias["max_auc"],
               "moyenne_real": mean[0], "moyenne_spoof": mean[1], "action": BIAS_ACTIONS.get(feature)},
              severity=bias.get("severity", "error"))


def check_schema(rows: list[dict], schema: dict, results: list) -> None:
    cols = schema["columns"]
    bad = Counter()
    pattern = re.compile(cols["subject"]["pattern"])
    for r in rows:
        if not r["path"]:
            bad["path"] += 1
        if r["label"] not in cols["label"]["allowed"]:
            bad["label"] += 1
        if not pattern.match(r["subject"]):
            bad["subject"] += 1
        if r["session"] not in cols["session"]["allowed"]:
            bad["session"] += 1
        if r["split"] not in cols["split"]["allowed"]:
            bad["split"] += 1
    check(results, "schema", not bad, dict(bad) or "toutes les colonnes conformes")


def check_images(rows: list[dict], q: dict, nuaa_dir: Path, results: list) -> dict:
    """Controles image par image ; renvoie luminance et saturation moyennes par image."""
    missing, undecodable, wrong_channels, too_small, not_square = [], [], [], [], []
    luminance = {}
    for r in rows:
        p = nuaa_dir / r["path"]
        if not p.is_file():
            missing.append(r["path"])
            continue
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            undecodable.append(r["path"])
            continue
        h, w = img.shape[:2]
        channels = 1 if img.ndim == 2 else img.shape[2]
        if channels == 3:
            luminance[r["path"]] = {
                "luminance_moyenne": float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()),
                "saturation_moyenne": float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean()),
            }
        if channels != q["image"]["channels"]:
            wrong_channels.append(r["path"])
        if min(h, w) < q["image"]["min_side_px"]:
            too_small.append(r["path"])
        if max(h, w) / min(h, w) > q["image"]["max_aspect_ratio"]:
            not_square.append(r["path"])
    c = q["completeness"]
    check(results, "fichiers_presents", len(missing) <= c["max_missing_files"],
          {"manquants": len(missing), "exemples": missing[:5]})
    check(results, "images_decodables", not undecodable,
          {"illisibles": len(undecodable), "exemples": undecodable[:5]})
    check(results, "canaux_couleur", not wrong_channels,
          {"non_conformes": len(wrong_channels), "exemples": wrong_channels[:5]})
    check(results, "taille_minimale", not too_small,
          {"trop_petites": len(too_small), "exemples": too_small[:5]})
    check(results, "format_carre", not not_square,
          {"non_carrees": len(not_square), "exemples": not_square[:5]})
    return luminance


def check_volume_and_balance(rows: list[dict], q: dict, results: list) -> None:
    n = Counter((r["split"], r["label"]) for r in rows)
    observed = {
        "total_images": len(rows),
        "train_val_real": n[("train", 0)] + n[("val", 0)],
        "train_val_spoof": n[("train", 1)] + n[("val", 1)],
        "test_real": n[("test", 0)],
        "test_spoof": n[("test", 1)],
    }
    expected = q["volume"]
    check(results, "volumetrie", observed == expected,
          {"attendu": expected, "observe": observed})

    dup = len(rows) - len({r["path"] for r in rows})
    check(results, "doublons", dup <= q["completeness"]["max_duplicate_paths"], {"doublons": dup})

    train_total = n[("train", 0)] + n[("train", 1)]
    ratio = min(n[("train", 0)], n[("train", 1)]) / max(train_total, 1)
    check(results, "equilibre_classes_train", ratio >= q["balance"]["min_class_ratio_train"],
          {"ratio_classe_minoritaire": round(ratio, 3)})


def check_leakage(rows: list[dict], leak: dict, results: list) -> None:
    by_split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    if leak.get("train_test_disjoint_sessions"):
        tr = {r["session"] for r in by_split["train"] + by_split["val"]}
        te = {r["session"] for r in by_split["test"]}
        check(results, "sessions_train_test_disjointes", not (tr & te),
              {"train_val": sorted(tr), "test": sorted(te)})
    if leak.get("train_val_disjoint_sessions"):
        tr = {r["session"] for r in by_split["train"]}
        va = {r["session"] for r in by_split["val"]}
        check(results, "sessions_train_val_disjointes", not (tr & va),
              {"train": sorted(tr), "val": sorted(va)})
    if leak.get("train_val_disjoint_subjects"):
        tr = {r["subject"] for r in by_split["train"]}
        va = {r["subject"] for r in by_split["val"]}
        check(results, "sujets_train_val_disjoints", not (tr & va),
              {"train": sorted(tr), "val": sorted(va)})
    if leak.get("no_path_in_two_splits"):
        seen = {}
        clash = [r["path"] for r in rows if seen.setdefault(r["path"], r["split"]) != r["split"]]
        check(results, "aucun_fichier_dans_deux_splits", not clash, {"conflits": len(clash)})


def check_archive(source: dict, results: list) -> None:
    expected = source.get("archive_sha256")
    if not config.NUAA_ZIP.is_file():
        check(results, "empreinte_archive", expected is None,
              "archive absente : controle d'empreinte ignore")
        return
    actual = sha256(config.NUAA_ZIP)
    check(results, "empreinte_archive", expected is None or actual == expected,
          {"attendu": expected, "observe": actual})


def validate(contract_path: Path = config.DATA_CONTRACT) -> dict:
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    rows = read_manifest()
    results: list = []
    check_archive(contract["source"], results)
    check_schema(rows, contract["schema"], results)
    luminance = check_images(rows, contract["quality"], config.NUAA_DIR, results)
    check_volume_and_balance(rows, contract["quality"], results)
    check_leakage(rows, contract["leakage"], results)
    if "bias" in contract["quality"]:
        check_bias(rows, luminance, contract["quality"]["bias"], results)
    return {
        "contract": f'{contract["contract"]["name"]} v{contract["contract"]["version"]}',
        "passed": all(r["passed"] for r in results if r["severity"] == "error"),
        "warnings": [r["check"] for r in results if not r["passed"] and r["severity"] == "warning"],
        "checks": results,
    }


def main() -> int:
    report = validate()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORTS_DIR / "data_quality_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for r in report["checks"]:
        tag = "OK  " if r["passed"] else ("WARN" if r["severity"] == "warning" else "KO  ")
        extra = f'  {r["detail"]}' if not r["passed"] else ""
        print(f'[{tag}] {r["check"]}{extra}')
    print(f'Contrat {report["contract"]} : {"RESPECTE" if report["passed"] else "VIOLE"}')
    print(f"Rapport : {out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
