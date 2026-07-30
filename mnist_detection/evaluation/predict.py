"""Inference and qualitative evaluation.

TRAINING VS INFERENCE — THE MENTAL MODEL
----------------------------------------
Training answers "how do I change the weights?"; inference answers "given fixed
weights, what is the output?". Practical consequences:

  * No gradients, no optimizer state, no shuffling.
  * Layers with train/test-divergent behaviour (Dropout, BatchNorm) switch mode.
    Keras handles this automatically via the ``training`` flag — ``predict`` and
    ``__call__(x, training=False)`` set it to False, ``fit`` sets it to True.
    Our model has neither layer, but knowing this is a standard viva question.
  * Batch size only affects throughput, never the result.

``model.predict(x)`` vs ``model(x)``
    ``predict`` loops over batches, converts to NumPy, and is the right call
    for large arrays. ``model(x, training=False)`` runs eagerly on a single
    batch and returns tensors — lower latency for one sample, but it will OOM
    on a 10k-sample array. Use ``predict`` for evaluation, ``__call__`` for
    single-image serving.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from mnist_detection.evaluation.metrics import (
    compute_iou,
    evaluate_predictions,
    print_report,
)
from mnist_detection.utils.visualization import plot_predictions, render_sample


def load_trained_model(checkpoint_path: str) -> tf.keras.Model:
    """Load a saved ``.keras`` model.

    Purpose
        Restore architecture + weights + optimizer state from one file.

    Common mistakes
        * Loading a model that used a custom layer/metric without passing
          ``custom_objects={...}`` -> ``ValueError: Unknown layer``. Our model
          uses only built-ins, so no custom_objects is needed; the custom
          ``MeanIoUMetric`` is only attached when explicitly compiled with it.
        * Loading legacy ``.h5`` with a subclassed model — architecture is not
          serialized there. Prefer ``.keras``.
    """
    return tf.keras.models.load_model(checkpoint_path)


def predict(model: tf.keras.Model, images: np.ndarray, batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over an array of images.

    Parameters
        images : (N, 75, 75, 1) float32 in [0, 1]. A single (75, 75, 1) image is
            auto-expanded to (1, 75, 75, 1) — forgetting the batch axis is the
            most common inference error and produces
            ``ValueError: Input 0 is incompatible with layer: expected ndim=4``.

    Returns
        probs : (N, 10) softmax probabilities
        boxes : (N, 4) normalized [ymin, xmin, ymax, xmax], unclipped

    Performance
        Larger ``batch_size`` amortizes kernel-launch overhead; 128-512 is
        typical for a model this small. The bottleneck at inference is usually
        host->device transfer, not compute.
    """
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 3:
        images = images[np.newaxis, ...]
    if images.ndim != 4 or images.shape[1:] != (75, 75, 1):
        raise ValueError(
            f"Expected images of shape (N, 75, 75, 1); got {images.shape}."
        )

    outputs = model.predict(images, batch_size=batch_size, verbose=0)
    return outputs["classification"], outputs["bounding_box"]


def predict_single(model: tf.keras.Model, image: np.ndarray) -> dict:
    """Low-latency single-image inference returning a human-readable dict.

    Returns
        ``{"label": int, "confidence": float, "box": (4,) np.ndarray}``

    Why report confidence?
        A detector that must abstain needs a threshold. ``max(softmax)`` is a
        crude but usable confidence. Caveat for the viva: neural networks are
        systematically *over*confident; calibration (temperature scaling) is the
        production fix.
    """
    probs, boxes = predict(model, image)
    idx = int(np.argmax(probs[0]))
    return {
        "label": idx,
        "confidence": float(probs[0][idx]),
        "box": boxes[0],
    }


def evaluate_and_visualize(
    model: tf.keras.Model,
    val_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    num_samples: int = 10,
    iou_threshold: float = 0.5,
    save_path: str | None = None,
    show_worst: bool = True,
) -> dict:
    """Full quantitative + qualitative evaluation on the validation split.

    Parameters
        val_arrays : ``(images, one_hot, boxes)`` as returned by
            ``get_datasets``. Using the raw arrays (not the batched dataset)
            keeps ordering stable so sample *i* in the plot is sample *i* in the
            metrics — essential when you go hunting for a specific failure.
        show_worst : also plot the lowest-IoU samples. **Always look at the
            worst cases.** Random samples confirm what already works; the tail
            tells you what to fix next.

    Returns
        The report dict from :func:`evaluate_predictions`.
    """
    images, one_hot, true_boxes = val_arrays
    probs, pred_boxes = predict(model, images)

    report = evaluate_predictions(one_hot, true_boxes, probs, pred_boxes, iou_threshold)
    print_report(report, iou_threshold)

    ious = report["per_sample_iou"]
    true_labels = np.argmax(one_hot, axis=1)
    pred_labels = np.argmax(probs, axis=1)

    plot_predictions(
        images,
        true_boxes,
        pred_boxes,
        true_labels,
        pred_labels,
        ious=ious,
        num_samples=num_samples,
        save_path=save_path,
    )

    if show_worst:
        worst = np.argsort(ious)[:num_samples]
        print(f"[predict] worst {len(worst)} samples by IoU: {ious[worst].round(3).tolist()}")
        plot_predictions(
            images[worst],
            true_boxes[worst],
            pred_boxes[worst],
            true_labels[worst],
            pred_labels[worst],
            ious=ious[worst],
            num_samples=num_samples,
            save_path=save_path.replace(".png", "_worst.png") if save_path else None,
        )

    return report


def demo_single_image(model: tf.keras.Model, val_arrays: tuple, index: int = 0) -> None:
    """Print and render one prediction — the smallest end-to-end sanity check."""
    images, one_hot, true_boxes = val_arrays
    result = predict_single(model, images[index])
    iou = float(compute_iou(true_boxes[index], result["box"])[0])

    print(
        f"[predict] sample {index}: "
        f"true={int(np.argmax(one_hot[index]))} pred={result['label']} "
        f"conf={result['confidence']:.3f} IoU={iou:.3f}"
    )
    render_sample(
        images[index],
        true_box=true_boxes[index],
        pred_box=result["box"],
        true_label=str(int(np.argmax(one_hot[index]))),
        pred_label=str(result["label"]),
    ).save("artifacts/single_prediction.png")
    print("[predict] saved artifacts/single_prediction.png")
