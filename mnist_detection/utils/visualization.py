"""Reusable visualization utilities for single-object detection.

WHY VISUALIZATION IS NOT OPTIONAL
---------------------------------
Detection metrics are aggregate scalars; they hide *systematic* failures.
A model can reach IoU 0.75 on average while consistently under-sizing boxes,
or while being perfect on centered digits and useless on corner digits. The
only reliable way to see that is to draw the boxes.

Rule of thumb from production CV: **look at 50 random predictions before you
trust any number.**

COLOR CONVENTION (fixed for this project)
    green = ground truth
    red   = prediction

COORDINATE CONTRACT
    Every function in this module takes boxes as
    ``[ymin, xmin, ymax, xmax]`` normalized to [0, 1], matching
    ``dataset/preprocessing.py``. Conversion to pixels happens in exactly one
    place (:func:`denormalize_box`) so there is a single point of failure.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

GROUND_TRUTH_COLOR = (0, 200, 0)      # green
PREDICTION_COLOR = (220, 0, 0)        # red


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def validate_image(image: np.ndarray) -> np.ndarray:
    """Coerce an image tensor to (H, W) float32 and validate its shape.

    Accepted inputs
        (H, W), (H, W, 1), (1, H, W, 1)  — the last form is what you get when
        you call ``model.predict`` on a single expanded sample and forget to
        squeeze.

    Returns
        (H, W) float32 array with values clipped to [0, 1].

    Why validate at all?
        Matplotlib silently misrenders a (H, W, 1) array in some versions and
        raises a cryptic ``TypeError: Invalid shape`` in others. Failing early
        with a message that names the actual shape saves real debugging time.
    """
    array = np.asarray(image, dtype=np.float32)
    array = np.squeeze(array)  # drops all size-1 axes: (1,H,W,1) -> (H,W)
    if array.ndim != 2:
        raise ValueError(
            f"Expected a grayscale image reducible to (H, W); got shape "
            f"{np.asarray(image).shape} which squeezes to {array.shape}."
        )
    return np.clip(array, 0.0, 1.0)


def validate_boxes(boxes: np.ndarray) -> np.ndarray:
    """Coerce box input to (N, 4) float32 and sanity-check the ordering.

    Accepted inputs
        (4,) single box, or (N, 4) multiple boxes.

    Raises
        ValueError if the trailing axis is not 4.

    Warns (via print) if ymin >= ymax or xmin >= xmax, which almost always
    means the coordinate order was swapped (e.g. [xmin, ymin, xmax, ymax]).
    A freshly initialized model legitimately produces degenerate boxes, so this
    is a warning rather than an exception.
    """
    array = np.asarray(boxes, dtype=np.float32)
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2 or array.shape[-1] != 4:
        raise ValueError(
            f"Expected boxes of shape (4,) or (N, 4); got {array.shape}."
        )
    degenerate = (array[:, 0] >= array[:, 2]) | (array[:, 1] >= array[:, 3])
    if degenerate.any():
        print(
            f"[visualization] warning: {int(degenerate.sum())} degenerate box(es) "
            "(ymin>=ymax or xmin>=xmax). Check coordinate order or model warm-up."
        )
    return array


def denormalize_box(box: Sequence[float], height: int, width: int) -> tuple[int, int, int, int]:
    """Convert a normalized [ymin, xmin, ymax, xmax] box to pixel coordinates.

    Returns
        ``(left, top, right, bottom)`` — note the reordering! PIL's
        ``ImageDraw.rectangle`` wants x-first, y-first, i.e.
        ``[(x0, y0), (x1, y1)]``. Doing the swap in one dedicated function is
        the whole point: every other call site stays in (y, x) space.

    Math
        pixel = normalized * dimension, then rounded and clamped to the image
        bounds so a slightly out-of-range prediction (the linear head is
        unbounded!) cannot raise or draw off-canvas.
    """
    ymin, xmin, ymax, xmax = (float(v) for v in box)
    left = int(round(np.clip(xmin, 0.0, 1.0) * width))
    top = int(round(np.clip(ymin, 0.0, 1.0) * height))
    right = int(round(np.clip(xmax, 0.0, 1.0) * width))
    bottom = int(round(np.clip(ymax, 0.0, 1.0) * height))
    return left, top, right, bottom


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def draw_boxes_pil(
    image: np.ndarray,
    boxes: np.ndarray,
    color: tuple[int, int, int],
    labels: Iterable[str] | None = None,
    width: int = 2,
    base: Image.Image | None = None,
) -> Image.Image:
    """Draw one or more bounding boxes on a grayscale image using PIL.

    Purpose
        PIL gives crisp 1-pixel-accurate rectangles and text. Matplotlib
        patches are fine too, but PIL lets us *bake* the boxes into an RGB
        array, which is handy for logging to TensorBoard or saving to disk.

    Parameters
        image  : (H, W) or (H, W, 1) array in [0, 1].
        boxes  : (4,) or (N, 4) normalized [ymin, xmin, ymax, xmax].
        color  : RGB tuple.
        labels : optional iterable of strings, one per box.
        width  : rectangle line thickness in pixels.
        base   : optional existing RGB ``Image`` to draw on top of. This is how
                 we overlay predicted boxes onto an image that already has the
                 ground-truth boxes — avoiding a duplicated "draw two sets"
                 function (DRY).

    Returns
        A PIL ``Image`` in RGB mode, size (W, H).

    Internal working
        Grayscale floats are scaled to uint8 0..255, converted to RGB (so the
        colored boxes are representable), then rectangles are stroked.

    Common mistakes
        * Passing float images to ``Image.fromarray`` without scaling — PIL
          interprets mode 'F' and you get a black or clipped image.
        * Drawing on an 'L' (grayscale) image and expecting color: green
          becomes gray. Always ``.convert("RGB")`` first.
    """
    boxes = validate_boxes(boxes)

    if base is None:
        gray = validate_image(image)
        height, width_px = gray.shape
        base = Image.fromarray((gray * 255).astype(np.uint8), mode="L").convert("RGB")
    else:
        base = base.copy()
        width_px, height = base.size

    draw = ImageDraw.Draw(base)
    label_list = list(labels) if labels is not None else [None] * len(boxes)

    for box, label in zip(boxes, label_list):
        left, top, right, bottom = denormalize_box(box, height, width_px)
        draw.rectangle([(left, top), (right, bottom)], outline=color, width=width)
        if label:
            # Place the caption above the box, or inside it if there is no room.
            text_y = top - 11 if top >= 11 else top + 1
            draw.text((left + 1, text_y), str(label), fill=color)

    return base


def render_sample(
    image: np.ndarray,
    true_box: np.ndarray | None = None,
    pred_box: np.ndarray | None = None,
    true_label: str | None = None,
    pred_label: str | None = None,
) -> Image.Image:
    """Compose a single annotated RGB image (green truth + red prediction).

    Returns
        PIL ``Image`` ready for ``plt.imshow`` or ``.save``.

    Design note
        This is the only function that knows the green/red convention, so
        changing the color scheme is a one-line edit.
    """
    canvas: Image.Image | None = None
    if true_box is not None:
        canvas = draw_boxes_pil(
            image, true_box, GROUND_TRUTH_COLOR, labels=[true_label] if true_label else None
        )
    if pred_box is not None:
        canvas = draw_boxes_pil(
            image,
            pred_box,
            PREDICTION_COLOR,
            labels=[pred_label] if pred_label else None,
            base=canvas,
        )
    if canvas is None:
        gray = validate_image(image)
        canvas = Image.fromarray((gray * 255).astype(np.uint8), mode="L").convert("RGB")
    return canvas


# --------------------------------------------------------------------------- #
# Matplotlib grid
# --------------------------------------------------------------------------- #
def plot_predictions(
    images: np.ndarray,
    true_boxes: np.ndarray,
    pred_boxes: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    ious: np.ndarray | None = None,
    num_samples: int = 10,
    cols: int = 5,
    save_path: str | None = None,
) -> None:
    """Plot a grid of samples with ground-truth (green) and predicted (red) boxes.

    Parameters
        images      : (N, H, W, 1) float32 in [0, 1]
        true_boxes  : (N, 4) normalized
        pred_boxes  : (N, 4) normalized
        true_labels : (N,) int  — class ids, NOT one-hot. Convert with argmax
                      before calling; keeping this function one-hot-agnostic
                      makes it reusable for any label source.
        pred_labels : (N,) int
        ious        : optional (N,) float — shown in the title, and used to
                      color the title green/red against a 0.5 threshold.
        num_samples : how many cells to draw.
        cols        : grid width; rows are derived.
        save_path   : if given, write a PNG instead of only showing.

    Reading the output
        * Red box larger than green  -> the box head over-predicts extent; often
          means the regression loss is under-weighted relative to classification.
        * Red box centered but tiny  -> classic MSE behavior early in training:
          predicting the dataset mean box minimizes MSE before real features
          are learned. It should shrink away after a few epochs.
        * Correct box, wrong digit   -> localization and classification are
          decoupled; look at the confusion pairs (4/9, 3/5, 7/1).
    """
    num_samples = int(min(num_samples, len(images)))
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.6 * rows))
    axes = np.atleast_1d(axes).ravel()

    for idx in range(num_samples):
        ax = axes[idx]
        composed = render_sample(
            images[idx],
            true_box=true_boxes[idx],
            pred_box=pred_boxes[idx],
        )
        ax.imshow(np.asarray(composed))
        ax.axis("off")

        title = f"gt {int(true_labels[idx])} / pred {int(pred_labels[idx])}"
        if ious is not None:
            title += f"\nIoU {float(ious[idx]):.2f}"
            ok = float(ious[idx]) >= 0.5 and int(true_labels[idx]) == int(pred_labels[idx])
            ax.set_title(title, fontsize=8, color="green" if ok else "red")
        else:
            ax.set_title(title, fontsize=8)

    for ax in axes[num_samples:]:
        ax.axis("off")

    fig.suptitle("green = ground truth   |   red = prediction", fontsize=10)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"[visualization] saved {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_training_history(history_dict: dict, save_path: str | None = None) -> None:
    """Plot the multi-task learning curves produced by ``model.fit``.

    ``history.history`` for a two-headed model contains keys such as::

        loss                                  # weighted sum of both losses
        classification_loss                   # categorical crossentropy
        bounding_box_loss                     # MSE
        classification_accuracy
        bounding_box_mse
        val_*                                 # same set on validation data

    Diagnostic reading
        * ``loss`` falling while ``bounding_box_loss`` is flat means the
          classification term dominates the gradient — rebalance ``loss_weights``.
        * ``val_*`` rising while ``*`` falls is textbook overfitting.
    """
    keys = [
        ("classification_loss", "val_classification_loss", "Classification loss (CE)"),
        ("bounding_box_loss", "val_bounding_box_loss", "Bounding box loss (MSE)"),
        ("classification_accuracy", "val_classification_accuracy", "Classification accuracy"),
    ]
    available = [k for k in keys if k[0] in history_dict]

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    axes = np.atleast_1d(axes)

    for ax, (train_key, val_key, title) in zip(axes, available):
        ax.plot(history_dict[train_key], label="train")
        if val_key in history_dict:
            ax.plot(history_dict[val_key], label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"[visualization] saved {save_path}")
    else:
        plt.show()
    plt.close(fig)
