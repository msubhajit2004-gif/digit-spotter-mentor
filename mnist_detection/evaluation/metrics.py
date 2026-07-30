"""Evaluation metrics: classification accuracy, box loss, and IoU.

WHY MSE IS NOT AN EVALUATION METRIC
-----------------------------------
MSE is a fine *training* loss (smooth, convex, differentiable everywhere) but a
poor *reporting* metric: "MSE = 0.0031" tells a reviewer nothing about whether
the box is usable. IoU is the standard because it is:

  * scale-invariant  — a 5-pixel error on a 20px object matters more than on a
    200px object, and IoU captures that automatically;
  * bounded in [0, 1] — directly interpretable and comparable across datasets;
  * the basis of the detection community's headline number (mAP@0.5, which
    counts a detection as correct iff IoU >= 0.5 AND the class is right).

INTERSECTION OVER UNION — THE FORMULA
-------------------------------------
For boxes A and B given as [ymin, xmin, ymax, xmax]:

    yi_min = max(A.ymin, B.ymin)        xi_min = max(A.xmin, B.xmin)
    yi_max = min(A.ymax, B.ymax)        xi_max = min(A.xmax, B.xmax)

    inter_h = max(0, yi_max - yi_min)
    inter_w = max(0, xi_max - xi_min)
    intersection = inter_h * inter_w

    area(A) = (A.ymax - A.ymin) * (A.xmax - A.xmin)
    area(B) = (B.ymax - B.ymin) * (B.xmax - B.xmin)

    union = area(A) + area(B) - intersection      <- subtract, or you
                                                     double-count the overlap

    IoU = intersection / (union + eps)

ASCII intuition::

        +-------------+
        |   A         |
        |      +------+------+
        |      | I    |      |
        +------+------+   B  |
               |             |
               +-------------+

    IoU = area(I) / (area(A) + area(B) - area(I))

Boundary behaviour
    * Identical boxes            -> IoU = 1
    * Disjoint boxes             -> intersection = 0 -> IoU = 0
    * B contained in A, half size-> IoU = 0.5

WHY IoU IS NOT USED AS THE LOSS HERE
    When boxes do not overlap, IoU = 0 for *every* non-overlapping
    configuration — the gradient is zero and carries no signal about which
    direction to move. That is exactly the situation at initialization. Modern
    detectors fix this with GIoU/DIoU, which add a penalty term based on the
    smallest enclosing box, restoring a gradient in the disjoint case. For this
    project: train with MSE, *evaluate* with IoU.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

EPSILON = 1e-9  # guards against 0/0 when both boxes are degenerate


# --------------------------------------------------------------------------- #
# NumPy IoU (for offline evaluation / reporting)
# --------------------------------------------------------------------------- #
def compute_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Element-wise IoU between two aligned sets of boxes.

    Parameters
        boxes_a : (N, 4) or (4,) — normalized [ymin, xmin, ymax, xmax]
        boxes_b : same shape as ``boxes_a``

    Returns
        (N,) float32 array of IoU values in [0, 1]. Scalar input -> shape (1,).

    Note this is *pairwise-aligned*, not the (N, M) cross-product matrix used
    for NMS/matching in multi-object detection. Single-object detection has
    exactly one prediction per image, so aligned is what we want.

    Common mistakes
        * Forgetting ``max(0, ...)`` on the intersection sides. Two disjoint
          boxes produce negative widths whose product is *positive*, yielding a
          bogus positive IoU. This is the single most common IoU bug.
        * Forgetting to subtract the intersection from the union -> IoU capped
          around 0.5 even for perfect boxes.
        * Mixing pixel and normalized coordinates between the two arguments.
          IoU is scale-invariant only if *both* boxes use the same scale.

    Debugging tips
        Unit-test with known cases::

            compute_iou([[0,0,1,1]], [[0,0,1,1]])      -> 1.0
            compute_iou([[0,0,1,1]], [[0,0,0.5,1]])    -> 0.5
            compute_iou([[0,0,.4,.4]], [[.6,.6,1,1]])  -> 0.0
    """
    a = np.atleast_2d(np.asarray(boxes_a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(boxes_b, dtype=np.float64))
    if a.shape != b.shape or a.shape[-1] != 4:
        raise ValueError(f"Expected matching (N, 4) shapes; got {a.shape} and {b.shape}")

    inter_ymin = np.maximum(a[:, 0], b[:, 0])
    inter_xmin = np.maximum(a[:, 1], b[:, 1])
    inter_ymax = np.minimum(a[:, 2], b[:, 2])
    inter_xmax = np.minimum(a[:, 3], b[:, 3])

    # Clamp at 0: negative side length means "no overlap on this axis".
    inter_h = np.maximum(0.0, inter_ymax - inter_ymin)
    inter_w = np.maximum(0.0, inter_xmax - inter_xmin)
    intersection = inter_h * inter_w

    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a + area_b - intersection

    return (intersection / (union + EPSILON)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Keras streaming metric (for live monitoring during fit)
# --------------------------------------------------------------------------- #
class MeanIoUMetric(tf.keras.metrics.Metric):
    """Streaming mean IoU usable inside ``model.compile(metrics=...)``.

    Why a Metric subclass instead of a plain function?
        A stateless function is recomputed per batch and Keras averages the
        per-batch means. That is only correct when all batches are the same
        size; the last partial batch skews the epoch average. A ``Metric``
        accumulates ``total`` and ``count`` in ``tf.Variable``s and divides once,
        giving the exact sample-weighted mean. It also survives
        ``@tf.function`` tracing and distribution strategies.

    Lifecycle
        ``__init__``      — create state variables (once).
        ``update_state``  — called per batch with (y_true, y_pred).
        ``result``        — called when Keras needs the current value.
        ``reset_state``   — called at the start of every epoch / evaluate.

    Shapes
        y_true : (B, 4)   y_pred : (B, 4)   ->  result: scalar

    Performance
        Pure elementwise TF ops, negligible cost. All computation stays on the
        accelerator — never call ``.numpy()`` inside ``update_state``, it forces
        a device sync and breaks graph mode.
    """

    def __init__(self, name: str = "mean_iou", **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        self.total_iou = self.add_weight(name="total_iou", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):  # noqa: D102
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        inter_ymin = tf.maximum(y_true[:, 0], y_pred[:, 0])
        inter_xmin = tf.maximum(y_true[:, 1], y_pred[:, 1])
        inter_ymax = tf.minimum(y_true[:, 2], y_pred[:, 2])
        inter_xmax = tf.minimum(y_true[:, 3], y_pred[:, 3])

        inter_h = tf.maximum(0.0, inter_ymax - inter_ymin)
        inter_w = tf.maximum(0.0, inter_xmax - inter_xmin)
        intersection = inter_h * inter_w

        area_true = tf.maximum(0.0, y_true[:, 2] - y_true[:, 0]) * tf.maximum(
            0.0, y_true[:, 3] - y_true[:, 1]
        )
        area_pred = tf.maximum(0.0, y_pred[:, 2] - y_pred[:, 0]) * tf.maximum(
            0.0, y_pred[:, 3] - y_pred[:, 1]
        )
        union = area_true + area_pred - intersection
        iou = intersection / (union + EPSILON)

        if sample_weight is not None:
            iou = iou * tf.cast(sample_weight, tf.float32)

        self.total_iou.assign_add(tf.reduce_sum(iou))
        self.count.assign_add(tf.cast(tf.size(iou), tf.float32))

    def result(self):  # noqa: D102
        return tf.math.divide_no_nan(self.total_iou, self.count)

    def reset_state(self):  # noqa: D102
        self.total_iou.assign(0.0)
        self.count.assign(0.0)


# --------------------------------------------------------------------------- #
# Aggregate report
# --------------------------------------------------------------------------- #
def classification_accuracy(y_true_one_hot: np.ndarray, y_pred_probs: np.ndarray) -> float:
    """Top-1 accuracy from one-hot targets and softmax probabilities.

    Math
        ``accuracy = (1/N) * sum_i 1[argmax(y_true_i) == argmax(y_pred_i)]``

    Why argmax on both sides?
        One-hot targets make argmax the class id; softmax outputs make argmax
        the most likely class. Accuracy ignores confidence entirely — a 0.35
        correct prediction scores the same as a 0.99 one. That is why you also
        watch crossentropy, which *is* confidence-sensitive.

    Limitation
        Accuracy is misleading on imbalanced data. MNIST is near-balanced
        (~10% per class), so it is honest here. For imbalanced problems report
        per-class precision/recall or a confusion matrix instead.
    """
    return float(np.mean(np.argmax(y_true_one_hot, axis=1) == np.argmax(y_pred_probs, axis=1)))


def bounding_box_mse(y_true_boxes: np.ndarray, y_pred_boxes: np.ndarray) -> float:
    """Mean squared error over all four normalized coordinates.

    Math
        ``MSE = (1 / (N*4)) * sum_{i,k} (y_true[i,k] - y_pred[i,k])^2``

    Interpretation trick
        ``sqrt(MSE) * 75`` converts the number back into *pixels of average
        per-coordinate error* on a 75px canvas — far more meaningful in a report
        than the raw MSE. e.g. MSE 0.0009 -> 0.03 normalized -> ~2.25 px.

    Why squared and not absolute?
        Squaring is differentiable at 0 and penalizes large errors
        super-linearly, which speeds up early convergence. The downside is
        sensitivity to outliers — one wildly wrong box dominates the batch
        gradient. Production detectors use Smooth L1 (Huber): quadratic near 0
        for a clean gradient, linear far away for outlier robustness.
    """
    return float(np.mean((np.asarray(y_true_boxes) - np.asarray(y_pred_boxes)) ** 2))


def evaluate_predictions(
    y_true_one_hot: np.ndarray,
    y_true_boxes: np.ndarray,
    y_pred_probs: np.ndarray,
    y_pred_boxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> dict:
    """Compute the full evaluation report for a set of predictions.

    Returns a dict with
        accuracy          : top-1 classification accuracy
        box_mse           : mean squared coordinate error
        box_rmse_pixels   : RMSE expressed in 75px canvas pixels
        mean_iou          : average IoU over all samples
        median_iou        : median IoU (robust to the tail of bad boxes)
        iou_above_thresh  : fraction of samples with IoU >= threshold
        detection_accuracy: fraction that are BOTH correctly classified AND
                            IoU >= threshold. This is the closest single number
                            to what "correct detection" means in the literature
                            (it is essentially mAP@0.5 for the single-object,
                            single-prediction case).
        per_sample_iou    : (N,) array, kept for plotting histograms.

    Why report median and a threshold rate, not just the mean?
        Mean IoU is dragged down by a small number of catastrophic failures.
        Seeing mean 0.71 / median 0.83 immediately tells you the distribution is
        left-skewed and that you should go look at the worst 5% of samples.
    """
    ious = compute_iou(y_true_boxes, y_pred_boxes)
    correct_class = np.argmax(y_true_one_hot, axis=1) == np.argmax(y_pred_probs, axis=1)
    box_mse = bounding_box_mse(y_true_boxes, y_pred_boxes)

    return {
        "accuracy": classification_accuracy(y_true_one_hot, y_pred_probs),
        "box_mse": box_mse,
        "box_rmse_pixels": float(np.sqrt(box_mse) * 75.0),
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "iou_above_thresh": float(np.mean(ious >= iou_threshold)),
        "detection_accuracy": float(np.mean(correct_class & (ious >= iou_threshold))),
        "per_sample_iou": ious,
    }


def print_report(report: dict, iou_threshold: float = 0.5) -> None:
    """Pretty-print the evaluation dict returned by :func:`evaluate_predictions`."""
    print("\n" + "=" * 52)
    print("EVALUATION REPORT")
    print("=" * 52)
    print(f"  Classification accuracy : {report['accuracy']:.4f}")
    print(f"  Bounding box MSE        : {report['box_mse']:.6f}")
    print(f"  Box RMSE (pixels @75)   : {report['box_rmse_pixels']:.2f} px")
    print(f"  Mean IoU                : {report['mean_iou']:.4f}")
    print(f"  Median IoU              : {report['median_iou']:.4f}")
    print(f"  IoU >= {iou_threshold:.2f}            : {report['iou_above_thresh']:.4f}")
    print(f"  Detection accuracy      : {report['detection_accuracy']:.4f}")
    print("=" * 52 + "\n")
