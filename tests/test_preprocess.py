import cv2
import numpy as np
import pytest

from src import config
from src.utils import preprocess as pp


def test_constants_match_config():
    """Le module d'inference duplique les constantes : elles doivent rester identiques."""
    assert pp.IMG_SIZE == config.IMG_SIZE
    assert np.allclose(pp.MEAN, config.IMAGENET_MEAN)
    assert np.allclose(pp.STD, config.IMAGENET_STD)


@pytest.mark.parametrize("shape", [(64, 64, 3), (203, 203, 3), (480, 640, 3)])
def test_preprocess_output_shape_and_dtype(shape):
    img = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    x = pp.preprocess(img)
    assert x.shape == (1, 3, pp.IMG_SIZE, pp.IMG_SIZE)
    assert x.dtype == np.float32
    assert x.flags["C_CONTIGUOUS"]


def test_normalize_converts_bgr_to_rgb_and_standardizes():
    img = np.zeros((pp.IMG_SIZE, pp.IMG_SIZE, 3), np.uint8)
    img[..., 2] = 255                              # rouge pur en BGR
    x = pp.normalize(img)
    expected_red = (1.0 - pp.MEAN[0]) / pp.STD[0]
    assert np.allclose(x[0], expected_red)         # canal R en premier apres conversion
    assert np.allclose(x[1], -pp.MEAN[1] / pp.STD[1])


def test_photo_normalize_removes_global_brightness_and_contrast():
    """Le meme visage, plus sombre ou plus clair, doit donner (presque) la meme entree."""
    rng = np.random.default_rng(3)
    face = rng.integers(60, 190, (pp.IMG_SIZE, pp.IMG_SIZE, 3)).astype(np.uint8)
    darker = np.clip(face * 0.6, 0, 255).astype(np.uint8)
    brighter = np.clip(face * 1.2 + 20, 0, 255).astype(np.uint8)
    ref = pp.photo_normalize(face).astype(float)
    for variant in (darker, brighter):
        assert np.abs(pp.photo_normalize(variant).astype(float) - ref).mean() < 2.0
    gray = cv2.cvtColor(pp.photo_normalize(darker), cv2.COLOR_BGR2GRAY)
    assert abs(gray.mean() - pp.PHOTO_MEAN) < 2 and abs(gray.std() - pp.PHOTO_STD) < 2


def test_preprocess_options_keep_shape():
    face = np.random.default_rng(4).integers(0, 256, (90, 90, 3), dtype=np.uint8)
    for kwargs in ({"photo_norm": True}, {"grayscale": True}, {"photo_norm": True, "grayscale": True}):
        assert pp.preprocess(face, **kwargs).shape == (1, 3, pp.IMG_SIZE, pp.IMG_SIZE)


def test_preprocess_rejects_grayscale():
    with pytest.raises(ValueError):
        pp.preprocess(np.zeros((100, 100), np.uint8))


def test_softmax_rows_sum_to_one():
    p = pp.softmax(np.array([[2.0, -1.0], [0.0, 0.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.allclose(p[1], [0.5, 0.5])
