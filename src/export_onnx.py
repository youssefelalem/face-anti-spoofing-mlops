"""Export du modele entraine au format ONNX, pour l'application.

Pourquoi ONNX ? L'application n'a plus besoin de PyTorch (~1 Go) : seulement
onnxruntime (~15 Mo). L'image Docker est plus legere et demarre plus vite.

Le script verifie que PyTorch et ONNX Runtime donnent les MEMES sorties, puis
ecrit model_meta.json (seuil, pretraitement, metriques) lu par l'application.

Usage : python -m src.export_onnx
"""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src import config
from src.models.mobilenet import build_model


def main() -> None:
    ckpt = torch.load(config.CHECKPOINT, map_location="cpu", weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
    torch.onnx.export(
        model, (dummy,), str(config.ONNX_MODEL),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(config.ONNX_MODEL)))

    # Verification de parite PyTorch <-> ONNX Runtime
    x = torch.randn(8, 3, config.IMG_SIZE, config.IMG_SIZE)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(config.ONNX_MODEL), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input": x.numpy()})[0]
    max_diff = float(np.abs(ref - out).max())
    if max_diff > 1e-3:
        raise RuntimeError(f"ecart PyTorch/ONNX trop grand : {max_diff}")

    test_metrics_path = config.REPORTS_DIR / "test_metrics.json"
    test_metrics = (json.loads(test_metrics_path.read_text(encoding="utf-8"))
                    if test_metrics_path.is_file() else None)
    final_path = config.REPORTS_DIR / "final_metrics.json"
    final = json.loads(final_path.read_text(encoding="utf-8")) if final_path.is_file() else None
    sha = hashlib.sha256(config.ONNX_MODEL.read_bytes()).hexdigest()
    meta = {
        "model_name": "mobilenetv2-fas-nuaa",
        "version": time.strftime("%Y.%m.%d"),
        "onnx_sha256": sha,
        "input": {"size": config.IMG_SIZE,
                  "grayscale": bool(ckpt.get("config", {}).get("grayscale", False)),
                  "photo_norm": bool(ckpt.get("config", {}).get("photo_norm", False)),
                  "color": "RGB",
                  "mean": list(config.IMAGENET_MEAN), "std": list(config.IMAGENET_STD)},
        "classes": config.CLASS_NAMES,
        "threshold": float(ckpt["val"]["threshold"]),
        "validation": {k: v for k, v in ckpt["val"].items() if not isinstance(v, dict)},
        "test": test_metrics,
        "final_test": final["test"] if final else None,
        "parity_max_abs_diff": max_diff,
    }
    config.MODEL_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    size_mb = config.ONNX_MODEL.stat().st_size / 1e6
    print(f"ONNX : {config.ONNX_MODEL} ({size_mb:.1f} Mo), ecart max PyTorch/ONNX = {max_diff:.2e}")
    print(f"Meta : {config.MODEL_META}")


if __name__ == "__main__":
    main()
