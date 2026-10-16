"""Promeut le checkpoint d'une experience comme modele final (avant evaluation et export).

Usage : python -m src.promote E1-baseline
"""

from __future__ import annotations

import json
import shutil
import sys

import mlflow
import torch

from src import config


def main(name: str) -> None:
    src_ckpt = config.MODELS_DIR / "experiments" / f"{name}.pt"
    ckpt = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(f"{config.MLFLOW_EXPERIMENT}-cross-session")
    runs = mlflow.search_runs([exp.experiment_id], filter_string=f"attributes.run_name LIKE '{name}-%'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    run_id = runs.iloc[0]["run_id"]
    shutil.copyfile(src_ckpt, config.CHECKPOINT)
    summary = {"mlflow_run_id": run_id, "promoted_from": name, "config": ckpt.get("config"),
               "best_val": ckpt["val"]}
    (config.MODELS_DIR / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("stage", "final")
    print(f"{name} promu : {config.CHECKPOINT} (run MLflow {run_id})")


if __name__ == "__main__":
    main(sys.argv[1])
