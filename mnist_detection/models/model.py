"""Two-headed CNN for simultaneous digit classification and localization.

THE ARCHITECTURE, IN WORDS BEFORE CODE
--------------------------------------

    Input (75, 75, 1)
          |
    +-----v-----------------------------------------+
    |  SHARED FEATURE EXTRACTOR                      |
    |  Conv2D(16, 3x3) + ReLU   -> (73, 73, 16)      |
    |  AveragePooling2D(2x2)    -> (36, 36, 16)      |
    |  Conv2D(32, 3x3) + ReLU   -> (34, 34, 32)      |
    |  AveragePooling2D(2x2)    -> (17, 17, 32)      |
    |  Conv2D(64, 3x3) + ReLU   -> (15, 15, 64)      |
    |  AveragePooling2D(2x2)    -> (7, 7, 64)        |
    |  Flatten                  -> (3136,)           |
    |  Dense(128) + ReLU        -> (128,)            |
    +-----+--------------------------+---------------+
          |                          |
    Dense(10, softmax)         Dense(4, linear)
    "classification" (10,)     "bounding_box" (4,)

WHY SHARE THE CONVOLUTIONAL TRUNK?
    1. The features needed to say "this is a 7" and the features needed to say
       "the ink spans rows 20-48" are largely the *same* features: edges,
       strokes, corners, and their spatial arrangement. Learning them twice
       wastes parameters and data.
    2. Multi-task learning acts as a regularizer. The box head forces the trunk
       to retain spatial information that a pure classifier would happily throw
       away (a classifier is rewarded for translation invariance; a localizer is
       punished for it). The resulting representation generalizes better on
       both tasks than either trained alone.
    3. Inference cost: one forward pass through the expensive part, two cheap
       heads. This is exactly the design of every real detector (SSD, YOLO,
       Faster R-CNN all share a backbone between class and box branches).

    The tension to be aware of: the two tasks pull the trunk in different
    directions. Pooling improves classification (invariance) but destroys
    localization precision. Our 3 pooling layers reduce 75 -> 7, i.e. each final
    feature cell covers ~10 input pixels; that is the floor on our localization
    resolution. Real detectors solve this with feature pyramids (FPN) and by
    predicting *offsets from anchors* rather than absolute coordinates.

WHY AVERAGE POOLING INSTEAD OF MAX POOLING?
    MaxPooling keeps the strongest activation and discards where the others
    were — great for "is this feature present?", lossy for "where exactly?".
    AveragePooling retains a smoother, more linear summary of the region, which
    the box regressor can decode into sub-cell positions. Empirically on this
    task AvgPool gives noticeably tighter boxes. This is a *localization-driven*
    architectural decision and a good viva answer.

WHY LINEAR (NO ACTIVATION) ON THE BOX HEAD?
    The targets are continuous values in [0, 1]. A sigmoid would also bound the
    output to [0, 1] and is a defensible choice, but it saturates: targets near
    0 or 1 (digits at the canvas edge, which happen constantly here) sit in the
    flat tail where gradients vanish. Linear output keeps gradients constant
    w.r.t. the pre-activation, which pairs cleanly with MSE. The cost is that
    the model *can* emit out-of-range values — which is why the visualization
    layer clips.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model

INPUT_SHAPE = (75, 75, 1)
NUM_CLASSES = 10


def build_feature_extractor(inputs: tf.Tensor) -> tf.Tensor:
    """Build the shared convolutional trunk.

    Parameters
        inputs : symbolic KerasTensor of shape (None, 75, 75, 1).

    Returns
        KerasTensor of shape (None, 3136) — the flattened feature map.

    Conv2D explained
        Syntax: ``layers.Conv2D(filters, kernel_size, activation, padding)``
        * filters      — number of independent learned kernels = output depth.
        * kernel_size  — 3x3 is the standard: the smallest kernel that has a
                         notion of center + all 8 neighbours. Two stacked 3x3s
                         have the receptive field of one 5x5 but use fewer
                         parameters (2*9=18 vs 25 per channel pair) and add an
                         extra nonlinearity.
        * padding      — 'valid' (our default) shrinks H,W by (k-1); 'same'
                         preserves them via zero padding.
        Shape rule (valid, stride 1): ``out = in - kernel_size + 1``.
        Parameter count: ``kernel_h * kernel_w * in_channels * filters + filters``.
        e.g. layer 1: 3*3*1*16 + 16 = 160 params.

        Internal working: each kernel slides across the input computing a dot
        product — a learned, translation-equivariant matched filter. Weight
        sharing is what makes CNNs data-efficient versus a Dense layer, which
        would need 75*75*128 = 720k params for a comparable first layer.

    ReLU explained
        ``f(x) = max(0, x)``. Non-saturating for x > 0 so gradients do not
        vanish in deep stacks; cheap (a single comparison). Failure mode: "dying
        ReLU" — a unit whose pre-activation is always negative has zero gradient
        forever. Symptom: a large fraction of activations are exactly 0 across
        all inputs. Mitigation: lower LR, or LeakyReLU.

    AveragePooling2D explained
        ``layers.AveragePooling2D(pool_size=2)`` -> ``out = floor(in / 2)``.
        No learnable parameters. Purpose: reduce spatial resolution (4x fewer
        activations per layer -> quadratic compute savings) and enlarge the
        effective receptive field of subsequent layers.
        Note 73 -> 36 (floor), not 36.5: one row/column is silently discarded.
        That asymmetry is harmless but worth knowing when you hand-check shapes.

    Flatten explained
        (None, 7, 7, 64) -> (None, 3136). No parameters, no computation — just
        a view change. It is also where translation equivariance ENDS: position
        becomes encoded in the index, which is precisely what lets the following
        Dense layer read off coordinates. A GlobalAveragePooling2D here would
        give a better classifier and a much worse localizer.

    Common mistakes
        * Feeding (75, 75) without the channel axis -> Conv2D raises
          ``Input 0 of layer conv2d is incompatible: expected ndim=4, found ndim=3``.
        * Forgetting that the batch axis is ``None`` in symbolic shapes.
    """
    x = layers.Conv2D(16, 3, activation="relu", name="conv_1")(inputs)
    x = layers.AveragePooling2D(2, name="pool_1")(x)

    x = layers.Conv2D(32, 3, activation="relu", name="conv_2")(x)
    x = layers.AveragePooling2D(2, name="pool_2")(x)

    x = layers.Conv2D(64, 3, activation="relu", name="conv_3")(x)
    x = layers.AveragePooling2D(2, name="pool_3")(x)

    x = layers.Flatten(name="flatten")(x)
    return x


def build_dense_bottleneck(features: tf.Tensor, units: int = 128) -> tf.Tensor:
    """Shared fully connected layer that both heads read from.

    Parameters
        features : (None, 3136)
        units    : width of the bottleneck.

    Returns
        (None, 128)

    Purpose
        Compresses 3136 spatial-and-semantic features into a compact joint
        representation. This is the *last shared point*: gradients from both
        heads meet here and are summed before flowing back into the trunk.

    Parameter count
        3136 * 128 + 128 = 401,536 — roughly 95% of the whole model. This is
        typical and is why modern architectures replace big Flatten+Dense stacks
        with global pooling; we keep it because we need the spatial detail.

    Mathematical intuition
        ``y = ReLU(W x + b)``, W of shape (3136, 128). Each of the 128 units is
        a learned linear probe over the entire feature map followed by a
        threshold — an arbitrary nonlinear function approximator, no spatial
        prior.
    """
    return layers.Dense(units, activation="relu", name="dense_shared")(features)


def build_model(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
) -> Model:
    """Assemble the two-output detection model with the Keras Functional API.

    Why the Functional API and not Sequential?
        ``Sequential`` is a strict linear chain: one input, one output, no
        branching. Our graph forks into two heads, which Sequential cannot
        express. The Functional API builds an explicit directed acyclic graph of
        layers by calling layers on tensors — it supports multiple inputs and
        outputs, shared layers, and residual connections, while still being
        fully serializable (unlike arbitrary subclassed models).

    Returns
        A compiled-ready ``tf.keras.Model`` with:
            inputs  : (None, 75, 75, 1)
            outputs : {"classification": (None, 10), "bounding_box": (None, 4)}

    Why name the outputs?
        Returning a *dict* of outputs lets you address each head by name in
        ``compile(loss={...}, metrics={...}, loss_weights={...})`` and in the
        target dict produced by the data pipeline. Positional lists work too,
        but silently pair the wrong loss to the wrong head if you reorder — a
        bug that produces plausible-looking but nonsense training.

    Softmax head
        ``Dense(10, activation="softmax")``.
        ``softmax(z)_i = exp(z_i) / sum_j exp(z_j)`` — maps 10 unbounded logits
        to a probability simplex (non-negative, sums to 1).
        Production note: it is numerically safer to output raw logits and use
        ``CategoricalCrossentropy(from_logits=True)``, which fuses log and
        softmax via the log-sum-exp trick and avoids ``log(0)``. We use explicit
        softmax here because the assignment specifies it and because it makes
        ``model.predict`` directly interpretable as probabilities.

    Bounding box head
        ``Dense(4)`` with no activation -> identity. Output order must match the
        target order: [ymin, xmin, ymax, xmax].

    Debugging tips
        * ``model.summary()`` — verify every shape and the total param count.
        * ``tf.keras.utils.plot_model(model, show_shapes=True)`` — renders the
          branch structure; the fastest way to confirm the heads really share
          the trunk rather than accidentally being two separate stacks.
        * If ``model.output_names`` does not match your target dict keys,
          ``fit`` fails immediately — check it before a long run.
    """
    inputs = layers.Input(shape=input_shape, name="image")

    features = build_feature_extractor(inputs)
    shared = build_dense_bottleneck(features)

    classification = layers.Dense(
        num_classes, activation="softmax", name="classification"
    )(shared)

    bounding_box = layers.Dense(4, activation=None, name="bounding_box")(shared)

    return Model(
        inputs=inputs,
        outputs={"classification": classification, "bounding_box": bounding_box},
        name="mnist_detector",
    )


def compile_model(
    model: Model,
    learning_rate: float = 1e-3,
    classification_weight: float = 1.0,
    bounding_box_weight: float = 1.0,
) -> Model:
    """Attach optimizer, per-head losses, and per-head metrics.

    HOW KERAS COMBINES MULTIPLE LOSSES
        With a dict of losses, Keras computes each head's loss independently and
        forms the scalar it actually differentiates:

            L_total = w_cls * CE(y_cls, p_cls) + w_box * MSE(y_box, p_box)

        Only ``L_total`` is passed to ``tape.gradient``. The per-head values you
        see in the progress bar are reported for monitoring only.

    HOW GRADIENTS FLOW THROUGH THE SHARED TRUNK
        Let ``h`` be the shared Dense(128) output. By the chain rule and the
        linearity of differentiation:

            dL_total/dh = w_cls * dCE/dh + w_box * dMSE/dh

        The two gradient signals are literally *summed* at the fork, then that
        single combined vector back-propagates through Dense -> Flatten ->
        Conv3 -> ... -> Conv1. Every shared weight therefore receives one update
        that is a compromise between the two objectives. Head-specific weights
        (the two final Dense layers) receive gradient from their own loss only.

    WHY LOSS WEIGHTS MATTER
        The two losses live on different scales. Categorical crossentropy starts
        near ``ln(10) = 2.30``; MSE on coordinates in [0, 1] starts around
        0.05-0.15. With equal weights the classification gradient dominates
        early, and the box head learns slowly. Diagnosis: ``bounding_box_loss``
        plateaus while ``classification_loss`` drops. Remedy: raise
        ``bounding_box_weight`` (10-50 is reasonable here) or switch to a
        scale-free box loss (Smooth L1 / GIoU, as real detectors do).

    Adam explained
        ``tf.keras.optimizers.Adam(learning_rate)`` maintains per-parameter
        exponential moving averages of the gradient (m, beta_1=0.9) and its
        square (v, beta_2=0.999), bias-corrects both, and steps

            theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)

        The per-parameter adaptive step is exactly what a multi-task model with
        mismatched loss scales needs — it partially compensates for the
        imbalance automatically. Default lr=1e-3 is a strong baseline; symptoms
        of too-high lr are a loss that spikes to nan, of too-low a loss that
        decreases linearly and slowly.

    Metrics explained
        * ``accuracy`` on the classification head: with one-hot targets and
          softmax outputs Keras resolves this to ``CategoricalAccuracy`` =
          ``mean(argmax(y_true) == argmax(y_pred))``. A pure accuracy of 1.0
          with a bad IoU means the model classifies but does not localize.
        * ``mse`` on the box head duplicates the loss; kept explicit so the
          reported number is unaffected by ``loss_weights``.

    Common mistakes
        * Passing a list of losses in the wrong order relative to
          ``model.outputs``.
        * Using ``sparse_categorical_crossentropy`` with one-hot targets —
          shape error, or worse, silently wrong if shapes happen to broadcast.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "classification": tf.keras.losses.CategoricalCrossentropy(),
            "bounding_box": tf.keras.losses.MeanSquaredError(),
        },
        loss_weights={
            "classification": classification_weight,
            "bounding_box": bounding_box_weight,
        },
        metrics={
            "classification": ["accuracy"],
            "bounding_box": ["mse"],
        },
    )
    return model
