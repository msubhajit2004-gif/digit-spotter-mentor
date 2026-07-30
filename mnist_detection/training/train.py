"""Training loop orchestration.

WHAT ``model.fit`` ACTUALLY DOES, PER STEP
------------------------------------------
    1. Pull one batch ``(x, {"classification": y_c, "bounding_box": y_b})``
       from the prefetch queue.
    2. Forward pass under a ``tf.GradientTape``: record every op from ``x`` to
       both outputs.
    3. Compute ``L_total = w_c * CE + w_b * MSE`` (plus any regularization).
    4. ``tape.gradient(L_total, model.trainable_variables)`` — reverse-mode
       automatic differentiation walks the recorded tape backwards, applying the
       chain rule. At the fork (the shared Dense(128)) the two incoming
       gradients are SUMMED, then propagate as one signal through the trunk.
    5. ``optimizer.apply_gradients(...)`` — Adam updates every weight.
    6. Metrics accumulate their state; the progress bar shows running means.

The whole step is compiled into a graph by ``tf.function`` on the first two
calls (hence the slow first steps), then executed as a fused, optimized kernel
sequence.

CALLBACKS = COMPOSABLE TRAINING POLICY
    Rather than writing conditionals inside a hand-rolled loop, Keras exposes
    hooks. Each callback below solves one real problem; that separation is the
    software-architecture point.
"""

from __future__ import annotations

import os
from typing import Any

import tensorflow as tf

from mnist_detection.dataset.preprocessing import get_datasets
from mnist_detection.models.model import build_model, compile_model


def build_callbacks(checkpoint_path: str, patience: int = 5) -> list[tf.keras.callbacks.Callback]:
    """Assemble the standard training callbacks.

    ModelCheckpoint
        Purpose: persist the *best* weights, not the last ones. The final epoch
        is frequently not the best epoch.
        Key parameters:
            ``monitor`` — which logged metric to watch. Must exactly match a key
                in ``history.history``; a typo means the callback silently never
                fires (it treats every epoch as an improvement or none).
            ``save_best_only=True`` — overwrite only on improvement.
            ``mode='min'`` for losses, ``'max'`` for accuracy/IoU.
        Common mistake: monitoring training loss instead of validation loss —
        you then checkpoint the most overfit model.

    EarlyStopping
        Purpose: stop when validation loss stops improving, saving compute and
        preventing overfitting.
            ``patience`` — epochs of no improvement tolerated. Too small and you
                stop during a normal plateau; too large and you waste time.
            ``restore_best_weights=True`` — critical. Without it the model in
                memory after ``fit`` is the *worst* one (the last, degraded
                epoch), which is a classic silent bug.

    ReduceLROnPlateau
        Purpose: anneal the learning rate when progress stalls. Large steps
        explore, small steps refine. Cutting LR by 0.5 on a plateau typically
        buys another visible drop in loss. An explicit schedule (cosine decay)
        is the more common production choice because it is deterministic and
        reproducible; plateau-based is more forgiving when you do not yet know
        how many epochs you need.
    """
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-5,
            verbose=1,
        ),
    ]


def train(
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    bounding_box_weight: float = 10.0,
    checkpoint_path: str = "artifacts/mnist_detector.keras",
    subset: int | None = None,
    seed: int = 42,
) -> tuple[tf.keras.Model, Any, tuple]:
    """Run the full training procedure and return (model, history, val_arrays).

    Parameters
        epochs : int — full passes over the training set.
        batch_size : int — 64 is a good default: enough samples for a stable
            gradient estimate, small enough for frequent updates.
        learning_rate : float — Adam's base step size.
        bounding_box_weight : float — see the loss-balancing discussion in
            ``models/model.py``. Default 10.0 because raw MSE is ~20x smaller
            than crossentropy at initialization, so unweighted training
            under-trains the box head.
        checkpoint_path : str — ``.keras`` extension uses the modern single-file
            format (architecture + weights + optimizer state).
        subset : optional int — train on fewer samples for a fast smoke test.
        seed : int — seeds Python/NumPy/TF RNGs for reproducibility.

    Reproducibility note
        ``tf.keras.utils.set_random_seed`` seeds all three RNGs at once. Full
        bit-determinism additionally needs
        ``tf.config.experimental.enable_op_determinism()``, which costs speed
        because it disables non-deterministic cuDNN kernels. Worth it for
        debugging, not for production training.

    Debugging tips
        * Loss goes to ``nan`` on step 1 -> inputs not normalized, or LR far too
          high. Check ``next(iter(train_ds))[0].numpy().max()`` == 1.0.
        * Accuracy stuck at exactly 0.10 -> the model predicts one class; usually
          a broken label pipeline (all targets identical) or dead ReLUs.
        * Box loss flat while class loss falls -> loss imbalance; raise
          ``bounding_box_weight``.
        * ``val_loss`` far below ``loss`` -> you shuffled validation into
          training, or dropout is active only in train (not the case here).
        * Sanity test before any long run: ``subset=256, epochs=30``. A correct
          model MUST reach ~100% train accuracy and near-zero box loss on 256
          samples. If it cannot overfit a tiny set, the bug is in the model or
          pipeline, not the hyperparameters.
    """
    tf.keras.utils.set_random_seed(seed)

    print("[train] building datasets ...")
    train_ds, val_ds, val_arrays = get_datasets(
        batch_size=batch_size, seed=seed, subset=subset
    )
    print(f"[train] element_spec: {train_ds.element_spec}")

    print("[train] building model ...")
    model = build_model()
    model = compile_model(
        model,
        learning_rate=learning_rate,
        classification_weight=1.0,
        bounding_box_weight=bounding_box_weight,
    )
    model.summary()

    print("[train] starting fit ...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=build_callbacks(checkpoint_path),
        verbose=1,
    )

    print(f"[train] best weights restored; saving to {checkpoint_path}")
    model.save(checkpoint_path)
    return model, history, val_arrays
