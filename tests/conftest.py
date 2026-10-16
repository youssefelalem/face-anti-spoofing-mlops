"""Fixtures partagees : un mini-modele ONNX factice pour tester l'API sans le vrai modele."""

from __future__ import annotations

import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from src.utils.preprocess import IMG_SIZE


def make_dummy_onnx(path) -> None:
    """(N, 3, H, W) -> moyenne globale -> couche lineaire -> 2 logits."""
    w = numpy_helper.from_array(np.array([[0.5, -0.5], [0.2, 0.1], [-0.3, 0.4]], np.float32), "W")
    b = numpy_helper.from_array(np.array([0.0, 0.0], np.float32), "B")
    graph = helper.make_graph(
        [helper.make_node("GlobalAveragePool", ["input"], ["pooled"]),
         helper.make_node("Flatten", ["pooled"], ["flat"]),
         helper.make_node("Gemm", ["flat", "W", "B"], ["logits"])],
        "dummy",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, IMG_SIZE, IMG_SIZE])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2])],
        initializer=[w, b],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


@pytest.fixture(scope="session")
def dummy_model(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    model_path, meta_path = d / "model.onnx", d / "model_meta.json"
    make_dummy_onnx(model_path)
    meta_path.write_text(json.dumps({
        "model_name": "dummy", "version": "test", "threshold": 0.5, "classes": ["real", "spoof"],
        "input": {"size": IMG_SIZE},
        "test": {"auc": 0.99, "eer": 0.02,
                 "at_validation_threshold": {"accuracy": 0.97, "APCER": 0.03, "BPCER": 0.02, "ACER": 0.025}},
    }), encoding="utf-8")
    return model_path, meta_path
