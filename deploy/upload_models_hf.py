"""Publie les modeles exportes dans un depot Hugging Face PRIVE (registre de modeles distant).

Les poids ne sont jamais publics (licence NUAA : recherche non commerciale, pas de
redistribution) ; l'application deployee les telecharge au demarrage avec un jeton de
lecture (variables FAS_HF_MODEL_REPO et HF_TOKEN, voir app/main.py et render.yaml).

Prerequis : `hf auth login`. Usage : python deploy/upload_models_hf.py
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = ["mobilenetv2_fas.onnx", "model_meta.json", "lbp_svm.joblib", "fusion.json"]


def main() -> None:
    api = HfApi()
    repo = f"{api.whoami()['name']}/face-anti-spoofing-models"
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    for name in MODEL_FILES:
        api.upload_file(path_or_fileobj=str(ROOT / "models" / "saved" / name), path_in_repo=name,
                        repo_id=repo, repo_type="model", commit_message=f"Upload {name}")
    print(f"Modeles publies (depot prive) : https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
