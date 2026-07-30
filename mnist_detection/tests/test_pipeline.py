"""Correctness tests for the parts most likely to be silently wrong.

Run with:  python -m pytest mnist_detection/tests/test_pipeline.py -v

Philosophy: do not test that TensorFlow works. Test *your* math — coordinate
conversion, IoU, and shape contracts. Those are where detection bugs hide,
because they fail silently rather than raising.
"""

from __future__ import annotations

import numpy as np

from mnist_detection.dataset.preprocessing import (
    CANVAS_SIZE,
    build_localization_arrays,
    place_digit_on_canvas,
)
from mnist_detection.evaluation.metrics import compute_iou
from mnist_detection.utils.visualization import denormalize_box, validate_boxes


# --------------------------------------------------------------------------- #
# IoU — the classic bug farm
# --------------------------------------------------------------------------- #
def test_iou_identical_boxes_is_one():
    assert np.isclose(compute_iou([[0.0, 0.0, 1.0, 1.0]], [[0.0, 0.0, 1.0, 1.0]])[0], 1.0)


def test_iou_half_overlap():
    # B is the top half of A -> intersection 0.5, union 1.0
    assert np.isclose(compute_iou([[0, 0, 1, 1]], [[0, 0, 0.5, 1]])[0], 0.5)


def test_iou_disjoint_is_zero():
    """Regression test for the missing max(0, ...) clamp."""
    assert np.isclose(compute_iou([[0, 0, 0.4, 0.4]], [[0.6, 0.6, 1.0, 1.0]])[0], 0.0)


def test_iou_quarter_overlap():
    # A = [0,0,1,1] area 1; B = [0.5,0.5,1.5,1.5] area 1; inter = 0.25
    # union = 1 + 1 - 0.25 = 1.75 -> IoU = 1/7
    assert np.isclose(compute_iou([[0, 0, 1, 1]], [[0.5, 0.5, 1.5, 1.5]])[0], 0.25 / 1.75)


def test_iou_is_batched():
    a = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32)
    b = np.array([[0, 0, 1, 1], [0.5, 0, 1, 1]], dtype=np.float32)
    assert compute_iou(a, b).shape == (2,)


# --------------------------------------------------------------------------- #
# Canvas placement
# --------------------------------------------------------------------------- #
def test_placement_preserves_ink():
    """Pasting must neither create nor destroy pixel mass."""
    rng = np.random.default_rng(0)
    digit = rng.random((28, 28)).astype(np.float32)
    canvas, _ = place_digit_on_canvas(digit, rng)
    assert np.isclose(canvas.sum(), digit.sum(), rtol=1e-5)


def test_box_matches_pasted_region():
    """The normalized box, converted back to pixels, must bound all the ink."""
    rng = np.random.default_rng(7)
    digit = np.ones((28, 28), dtype=np.float32)
    canvas, box = place_digit_on_canvas(digit, rng)

    left, top, right, bottom = denormalize_box(box, CANVAS_SIZE, CANVAS_SIZE)
    assert canvas[top:bottom, left:right].sum() == canvas.sum()
    assert (bottom - top, right - left) == (28, 28)


def test_box_is_normalized_and_ordered():
    rng = np.random.default_rng(3)
    for _ in range(200):
        _, box = place_digit_on_canvas(np.zeros((28, 28), np.float32), rng)
        ymin, xmin, ymax, xmax = box
        assert 0.0 <= ymin < ymax <= 1.0
        assert 0.0 <= xmin < xmax <= 1.0


def test_placement_covers_extremes():
    """Guards the off-by-one: the bottom-right placement must be reachable."""
    rng = np.random.default_rng(1)
    maxes = [place_digit_on_canvas(np.zeros((28, 28), np.float32), rng)[1][2]
             for _ in range(3000)]
    assert np.isclose(max(maxes), 1.0)


# --------------------------------------------------------------------------- #
# Array/shape contracts
# --------------------------------------------------------------------------- #
def test_build_arrays_shapes_and_ranges():
    images = (np.random.rand(16, 28, 28) * 255).astype(np.uint8)
    labels = np.arange(16) % 10
    canvases, one_hot, boxes = build_localization_arrays(images, labels)

    assert canvases.shape == (16, 75, 75, 1)
    assert one_hot.shape == (16, 10)
    assert boxes.shape == (16, 4)
    assert canvases.dtype == np.float32
    assert 0.0 <= canvases.min() and canvases.max() <= 1.0
    assert np.allclose(one_hot.sum(axis=1), 1.0)
    assert np.array_equal(np.argmax(one_hot, axis=1), labels)


def test_deterministic_given_seed():
    images = (np.random.rand(8, 28, 28) * 255).astype(np.uint8)
    labels = np.zeros(8, dtype=np.uint8)
    a = build_localization_arrays(images, labels, seed=5)[2]
    b = build_localization_arrays(images, labels, seed=5)[2]
    assert np.array_equal(a, b)


def test_validate_boxes_promotes_single_box():
    assert validate_boxes([0.1, 0.2, 0.3, 0.4]).shape == (1, 4)


def test_denormalize_swaps_to_xy_order():
    """ymin=0.0, xmin=0.5 -> left=37, top=0. Catches y/x transposition."""
    left, top, right, bottom = denormalize_box([0.0, 0.5, 0.5, 1.0], 100, 100)
    assert (left, top, right, bottom) == (50, 0, 100, 50)
