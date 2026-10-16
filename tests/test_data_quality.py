"""Tests du manifeste et des garde-fous anti-fuite, sur une mini-arborescence NUAA factice."""

from pathlib import Path

from src.preprocessing.build_manifest import build_rows, read_manifest, write_manifest
from src.preprocessing.validate_data import check_leakage

LEAK_RULES = {"train_test_disjoint_sessions": True, "train_val_disjoint_subjects": True,
              "no_path_in_two_splits": True}
SESSION_RULES = {"train_test_disjoint_sessions": True, "train_val_disjoint_sessions": True,
                 "no_path_in_two_splits": True}


def make_fake_nuaa(root: Path) -> Path:
    index = {
        "client_train_face.txt": ["0001\\0001_00_00_01_0.jpg 1 2 3 4 5 6",
                                  "0008\\0008_01_02_02_5.jpg 1 2 3 4 5 6"],
        "client_test_face.txt": ["0011\\0011_00_01_03_7.jpg 1 2 3 4 5 6"],
        "imposter_train_face.txt": ["0002\\0002_00_00_02_1.jpg 1 2 3 4 5 6",
                                    "0009\\0009_00_03_01_9.jpg 1 2 3 4 5 6"],
        "imposter_test_face.txt": ["0012\\0012_01_01_03_3.jpg 1 2 3 4 5 6", ""],
    }
    for name, lines in index.items():
        (root / name).write_text("\n".join(lines), encoding="latin-1")
    return root


def test_session_protocol_uses_session_02_for_validation(tmp_path):
    rows = build_rows(make_fake_nuaa(tmp_path), protocol="session", val_session="02")
    assert {r["session"] for r in rows if r["split"] == "val"} == {"02"}
    assert {r["session"] for r in rows if r["split"] == "train"} == {"01"}
    results = []
    check_leakage(rows, SESSION_RULES, results)
    assert all(r["passed"] for r in results), results


def test_unknown_protocol_is_rejected(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        build_rows(make_fake_nuaa(tmp_path), protocol="random")


def test_build_rows_splits_and_labels(tmp_path):
    rows = build_rows(make_fake_nuaa(tmp_path), protocol="subject", val_subjects=("0008", "0009"))
    by_path = {r["path"]: r for r in rows}
    assert len(rows) == 6                                    # ligne vide ignoree
    assert by_path["ClientFace/0001/0001_00_00_01_0.jpg"] == {
        "path": "ClientFace/0001/0001_00_00_01_0.jpg", "label": 0,
        "subject": "0001", "session": "01", "split": "train"}
    assert by_path["ClientFace/0008/0008_01_02_02_5.jpg"]["split"] == "val"
    assert by_path["ImposterFace/0009/0009_00_03_01_9.jpg"]["label"] == 1
    assert {r["split"] for r in rows if r["session"] == "03"} == {"test"}


def test_manifest_roundtrip(tmp_path):
    rows = build_rows(make_fake_nuaa(tmp_path))
    out = tmp_path / "manifest.csv"
    write_manifest(rows, out)
    assert read_manifest(out) == rows


def test_leakage_checks_pass_on_clean_split(tmp_path):
    results = []
    rows = build_rows(make_fake_nuaa(tmp_path), protocol="subject", val_subjects=("0008", "0009"))
    check_leakage(rows, LEAK_RULES, results)
    assert all(r["passed"] for r in results), results


def test_leakage_detects_subject_in_train_and_val():
    rows = [
        {"path": "a.jpg", "label": 0, "subject": "0001", "session": "01", "split": "train"},
        {"path": "b.jpg", "label": 0, "subject": "0001", "session": "02", "split": "val"},
        {"path": "c.jpg", "label": 1, "subject": "0005", "session": "01", "split": "test"},
    ]
    raw = []
    check_leakage(rows, LEAK_RULES, raw)
    results = {r["check"]: r["passed"] for r in raw}
    assert results["sujets_train_val_disjoints"] is False
    assert results["sessions_train_test_disjointes"] is False
