"""Data pipeline for MNIST single-object detection.

CONCEPT FIRST — WHY THIS FILE EXISTS
------------------------------------
Plain MNIST is a *classification* dataset: a 28x28 digit that already fills the
frame. There is nothing to localize, because the object is the image.

To turn it into an *object localization* dataset we synthesize a larger empty
canvas (75x75) and paste the 28x28 digit at a random (row, col) offset. Because
*we* choose the offset, we know the exact ground-truth bounding box for free —
no human annotation needed. This is the cheapest possible way to study
localization, and it isolates the learning problem: the network must learn
"where is the ink" and "which digit is it" at the same time.

TENSOR SHAPE ROADMAP
--------------------
    raw MNIST image        (28, 28)        uint8,  values 0..255
    normalized             (28, 28)        float32, values 0..1
    padded canvas          (75, 75)        float32
    channel dim added      (75, 75, 1)     float32   <- model input
    label                  ()              int64  ->  (10,) one-hot float32
    bbox                   (4,)            float32, normalized [ymin,xmin,ymax,xmax]

    after .batch(B):
        images  (B, 75, 75, 1)
        labels  (B, 10)
        boxes   (B, 4)

BOX FORMAT
----------
We use [ymin, xmin, ymax, xmax] *normalized to [0, 1]* because that is the
convention of `tf.image.draw_bounding_boxes` and `tf.image.crop_and_resize`.
Sticking to a library convention saves you from silent coordinate bugs.

    ymin = top    / 75      xmin = left  / 75
    ymax = bottom / 75      xmax = right / 75

COMMON MISTAKE: mixing (x, y) and (y, x) order. Symptom: the model trains, loss
goes down, but the drawn boxes are mirrored across the diagonal. Fix: pick ONE
order and assert it in every function's docstring (we do).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf

# --------------------------------------------------------------------------- #
# Constants — keep magic numbers in exactly one place.
# --------------------------------------------------------------------------- #
DIGIT_SIZE: int = 28        # MNIST digits are 28x28
CANVAS_SIZE: int = 75       # Target canvas edge length
NUM_CLASSES: int = 10       # Digits 0..9
AUTOTUNE = tf.data.AUTOTUNE  # Lets tf.data pick parallelism at runtime


# --------------------------------------------------------------------------- #
# 1. Raw dataset loading
# --------------------------------------------------------------------------- #
def load_mnist() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load the MNIST dataset from Keras' cached download.

    Purpose
        Fetch the canonical 70k-image handwritten digit dataset.

    Syntax
        ``tf.keras.datasets.mnist.load_data(path='mnist.npz')``

    Parameters
        path : str — filename inside ``~/.keras/datasets``. Default is fine.

    Returns
        ``((x_train, y_train), (x_test, y_test))`` where
        ``x_train`` : (60000, 28, 28) uint8, ``y_train`` : (60000,) uint8,
        ``x_test``  : (10000, 28, 28) uint8, ``y_test``  : (10000,) uint8.

    Internal working
        Downloads a single compressed ``.npz`` (~11 MB) once, then memory-maps
        it on later calls. No preprocessing is applied.

    Common mistakes
        * Assuming the arrays are float — they are ``uint8`` in [0, 255].
          Feeding them straight into a Conv2D makes gradients ~255x too large
          and training diverges (loss -> nan).
        * Assuming a channel axis exists. It does not; shape is (N, 28, 28).

    Debugging tips
        ``print(x_train.dtype, x_train.shape, x_train.max())`` — you should see
        ``uint8 (60000, 28, 28) 255``.

    Performance
        First call hits the network. In CI, pre-bake the cache or vendor the
        ``.npz`` so builds are offline-deterministic.
    """
    return tf.keras.datasets.mnist.load_data()


# --------------------------------------------------------------------------- #
# 2. Random canvas placement + bounding box generation
# --------------------------------------------------------------------------- #
def place_digit_on_canvas(
    digit: np.ndarray,
    rng: np.random.Generator,
    canvas_size: int = CANVAS_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Paste one 28x28 digit at a random location on a blank square canvas.

    Purpose
        Manufacture a localization task with exact, free ground truth.

    Parameters
        digit : np.ndarray, shape (28, 28), float32 in [0, 1].
        rng   : np.random.Generator — injected so runs are reproducible.
                Injecting the generator (instead of calling ``np.random.*``
                internally) is a software-engineering choice: it makes the
                function *pure* w.r.t. its inputs and therefore unit-testable.
        canvas_size : int — edge length of the output canvas. Must be >= 28.

    Returns
        canvas : (canvas_size, canvas_size) float32 in [0, 1]
        bbox   : (4,) float32 = [ymin, xmin, ymax, xmax] normalized to [0, 1]

    Mathematical intuition
        Let ``(r, c)`` be the top-left pixel of the pasted digit, drawn
        uniformly from ``{0, ..., S-28}``. Then in pixel space the box is
        ``(r, c, r+28, c+28)`` and normalization is a simple division by ``S``:

            ymin = r / S,  xmin = c / S,  ymax = (r+28) / S,  xmax = (c+28) / S

        Normalization makes the regression target resolution-independent and
        keeps all four targets on the same O(1) scale as the network's
        activations — critical for MSE, which is scale-sensitive.

    Common mistakes
        * Using ``rng.integers(0, S - 28)`` — ``integers`` is half-open, so the
          bottom/right-most placement never occurs and the model learns a subtle
          spatial bias. Correct upper bound is ``S - 28 + 1``.
        * Using the *ink* extent as the box instead of the 28x28 patch extent.
          Both are defensible; the patch extent is the standard choice for this
          exercise. Be consistent — the model can only learn one convention.

    Debugging tips
        Assert ``0 <= ymin < ymax <= 1`` and that ``canvas.sum() ==
        digit.sum()`` (pasting must not create or destroy ink).
    """
    if digit.shape != (DIGIT_SIZE, DIGIT_SIZE):
        raise ValueError(f"Expected digit of shape (28, 28), got {digit.shape}")
    if canvas_size < DIGIT_SIZE:
        raise ValueError(f"canvas_size must be >= {DIGIT_SIZE}, got {canvas_size}")

    max_offset = canvas_size - DIGIT_SIZE  # inclusive upper bound
    row = int(rng.integers(0, max_offset + 1))
    col = int(rng.integers(0, max_offset + 1))

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    canvas[row : row + DIGIT_SIZE, col : col + DIGIT_SIZE] = digit

    scale = float(canvas_size)
    bbox = np.array(
        [
            row / scale,                     # ymin
            col / scale,                     # xmin
            (row + DIGIT_SIZE) / scale,      # ymax
            (col + DIGIT_SIZE) / scale,      # xmax
        ],
        dtype=np.float32,
    )
    return canvas, bbox


def build_localization_arrays(
    images: np.ndarray,
    labels: np.ndarray,
    canvas_size: int = CANVAS_SIZE,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized construction of the full synthetic localization dataset.

    Purpose
        Apply :func:`place_digit_on_canvas` to every MNIST sample and produce
        three aligned NumPy arrays ready for ``tf.data``.

    Parameters
        images : (N, 28, 28) uint8 — raw MNIST pixels.
        labels : (N,) uint8 — integer class ids 0..9.
        canvas_size : int — output canvas edge.
        seed : int — RNG seed; identical seeds give byte-identical datasets.

    Returns
        canvases : (N, S, S, 1) float32 in [0, 1]
        one_hot  : (N, 10)      float32
        boxes    : (N, 4)       float32, normalized [ymin, xmin, ymax, xmax]

    Why normalize pixels to [0, 1]?
        Conv weights are initialized assuming unit-ish scale inputs (Glorot).
        With 0..255 inputs the first layer's pre-activations are ~255x larger,
        ReLU saturates the optimizer's step size, and Adam needs many steps just
        to rescale. Dividing by 255 is the cheapest possible normalization and
        is sufficient for MNIST (per-channel mean/std is used for natural
        images).

    Why one-hot encode?
        We pair the classification head with ``categorical_crossentropy``, which
        expects a probability *vector* target of shape (10,). The alternative is
        integer targets + ``sparse_categorical_crossentropy`` (saves memory:
        4 bytes vs 40 bytes per sample). One-hot is used here because it makes
        the loss formula ``-sum(y_true * log(y_pred))`` literally visible.

    Performance
        Materializing 60k 75x75 float32 canvases costs
        ``60000 * 75 * 75 * 4 B ~= 1.35 GB``. That fits in RAM on a dev box and
        makes ``.cache()`` free, but in production you would generate canvases
        lazily inside ``map()`` (or store TFRecords) instead of holding the whole
        tensor in memory.
    """
    rng = np.random.default_rng(seed)
    num_samples = images.shape[0]

    canvases = np.zeros((num_samples, canvas_size, canvas_size), dtype=np.float32)
    boxes = np.zeros((num_samples, 4), dtype=np.float32)

    # Scale once, outside the loop: one vectorized op instead of N scalar ops.
    normalized_digits = images.astype(np.float32) / 255.0

    for i in range(num_samples):
        canvases[i], boxes[i] = place_digit_on_canvas(
            normalized_digits[i], rng, canvas_size
        )

    # (N, S, S) -> (N, S, S, 1). Conv2D requires an explicit channel axis.
    canvases = np.expand_dims(canvases, axis=-1)

    one_hot = np.eye(NUM_CLASSES, dtype=np.float32)[labels.astype(np.int32)]

    return canvases, one_hot, boxes


# --------------------------------------------------------------------------- #
# 3. tf.data pipeline
# --------------------------------------------------------------------------- #
def make_dataset(
    canvases: np.ndarray,
    one_hot: np.ndarray,
    boxes: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    shuffle_buffer: int = 10_000,
    cache: bool = True,
) -> tf.data.Dataset:
    """Wrap the arrays in a performant, correctly ordered ``tf.data`` pipeline.

    Element structure
        ``(image, {"classification": one_hot, "bounding_box": box})``

        The dict keys MUST match the ``name=`` of the model's output layers.
        Keras uses those names to route each target to the right loss. If they
        don't match you get ``ValueError: Found unexpected keys that do not
        correspond to any Model output``.

    Shapes per element
        image (75, 75, 1) -> batched (B, 75, 75, 1)
        classification (10,) -> (B, 10)
        bounding_box (4,) -> (B, 4)

    THE ORDER OF OPERATIONS AND WHY
        1. ``from_tensor_slices`` — turns the leading axis into the element
           axis. Input (N, ...) -> N elements of shape (...).
        2. ``cache()`` — memoizes everything *upstream* of it after the first
           epoch. Placed BEFORE shuffle so the expensive deterministic work is
           cached but the *ordering* still varies per epoch. Caching after
           shuffle would freeze one permutation forever — a classic bug that
           quietly reduces effective data diversity.
        3. ``shuffle(buffer)`` — maintains a reservoir of ``buffer`` elements
           and samples from it. buffer=1 means no shuffling; buffer >= dataset
           size means a true uniform permutation. Insufficient buffer with
           class-sorted data yields correlated batches, which makes SGD's
           gradient estimate biased within an epoch.
        4. ``batch(B)`` — stacks B elements, adding a leading batch axis.
           Batching after shuffling is required; batching first would shuffle
           whole batches, not samples.
        5. ``prefetch(AUTOTUNE)`` — decouples producer (CPU) from consumer
           (GPU) with a background thread, so data for step *n+1* is prepared
           while step *n* runs on the accelerator. Always last.

    Where does ``map()`` go?
        Between ``from_tensor_slices`` and ``cache`` for deterministic
        preprocessing (resize, cast), and AFTER ``cache`` for random
        augmentation (so the randomness is re-rolled every epoch). Here the
        canvases are pre-built, so the only map is the label-dict repack.

    Parameters
        batch_size : int — larger = smoother gradients + better GPU
            utilization, but fewer updates per epoch and more memory.
        shuffle_buffer : int — trade RAM for randomness.
        cache : bool — set False if the dataset does not fit in RAM.

    Debugging tips
        ``print(ds.element_spec)`` shows dtypes and shapes without consuming
        data. ``imgs, tgts = next(iter(ds))`` pulls one real batch.
    """

    def _to_targets(image, label, box):
        """Repack a flat tuple into (input, {output_name: target}) form."""
        return image, {"classification": label, "bounding_box": box}

    dataset = tf.data.Dataset.from_tensor_slices((canvases, one_hot, boxes))
    dataset = dataset.map(_to_targets, num_parallel_calls=AUTOTUNE)

    if cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def get_datasets(
    batch_size: int = 64,
    canvas_size: int = CANVAS_SIZE,
    seed: int = 42,
    subset: int | None = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """End-to-end entry point: MNIST on disk -> two ready ``tf.data`` pipelines.

    Train/validation split
        We use MNIST's official train/test partition (60k / 10k). This is a
        *held-out* split, not a random one: the test digits were written by a
        disjoint group of writers. That is exactly what you want for an honest
        generalization estimate. A random split of the 70k pool would leak
        writer style across the boundary and inflate accuracy.

        Note the naming: we call the 10k set "validation" because we watch it
        every epoch. Strictly, using it for early stopping / hyperparameter
        choice means it is no longer a clean test set. Production practice:
        carve a third split (e.g. 55k train / 5k val / 10k test).

    Parameters
        batch_size  : int
        canvas_size : int
        seed        : int — reproducibility for placement.
        subset      : optional int — cap the number of training samples. Useful
                      for smoke tests and for demonstrating overfitting.

    Returns
        (train_ds, val_ds, raw_val) where ``raw_val`` is the tuple of NumPy
        arrays for the validation split, handed back so visualization and IoU
        code can index individual samples without re-iterating a pipeline.

    Common mistakes
        * Shuffling the validation set — harmless for metrics but makes
          per-epoch qualitative comparisons non-reproducible. We disable it.
        * Reusing the same RNG seed for train and val placement. Harmless here
          (different images), but be explicit about it.
    """
    (x_train, y_train), (x_val, y_val) = load_mnist()

    if subset is not None:
        x_train, y_train = x_train[:subset], y_train[:subset]

    train_arrays = build_localization_arrays(x_train, y_train, canvas_size, seed)
    val_arrays = build_localization_arrays(x_val, y_val, canvas_size, seed + 1)

    train_ds = make_dataset(*train_arrays, batch_size=batch_size, shuffle=True)
    val_ds = make_dataset(*val_arrays, batch_size=batch_size, shuffle=False)

    return train_ds, val_ds, val_arrays
