"""CLI entry point: train, evaluate, or inspect the MNIST detector.

Usage
    python -m mnist_detection.main train --epochs 10
    python -m mnist_detection.main train --subset 256 --epochs 30   # overfit test
    python -m mnist_detection.main evaluate --checkpoint artifacts/mnist_detector.keras
    python -m mnist_detection.main inspect                          # look at the data first

WHY A CLI AND NOT A NOTEBOOK?
    Notebooks are excellent for exploration and terrible for reproducibility:
    hidden execution order, un-diffable JSON, no way to run in CI. The
    professional pattern is: modules hold the logic, a thin CLI composes them,
    and a notebook (if any) just *imports* the modules. Every function here is
    importable and unit-testable in isolation.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from mnist_detection.dataset.preprocessing import get_datasets
from mnist_detection.evaluation.predict import (
    demo_single_image,
    evaluate_and_visualize,
    load_trained_model,
)
from mnist_detection.training.train import train
from mnist_detection.utils.visualization import plot_predictions, plot_training_history

ARTIFACTS = "artifacts"


def cmd_inspect(args: argparse.Namespace) -> None:
    """Render ground-truth boxes only — verify the data before touching a model.

    This step is not optional discipline theatre. If the boxes are misaligned
    here, the model will faithfully learn the misalignment and every downstream
    metric will look fine while the system is wrong.
    """
    _, _, val_arrays = get_datasets(batch_size=args.batch_size, subset=args.subset)
    images, one_hot, boxes = val_arrays
    labels = np.argmax(one_hot, axis=1)

    os.makedirs(ARTIFACTS, exist_ok=True)
    plot_predictions(
        images,
        boxes,
        boxes,  # draw truth twice; red should sit exactly on green
        labels,
        labels,
        num_samples=args.num_samples,
        save_path=f"{ARTIFACTS}/ground_truth.png",
    )
    print("[inspect] red must perfectly overlap green. If not, the box math is wrong.")


def cmd_train(args: argparse.Namespace) -> None:
    """Train, plot learning curves, then evaluate."""
    os.makedirs(ARTIFACTS, exist_ok=True)
    model, history, val_arrays = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        bounding_box_weight=args.box_weight,
        checkpoint_path=args.checkpoint,
        subset=args.subset,
    )
    plot_training_history(history.history, save_path=f"{ARTIFACTS}/history.png")
    evaluate_and_visualize(
        model,
        val_arrays,
        num_samples=args.num_samples,
        save_path=f"{ARTIFACTS}/predictions.png",
    )
    demo_single_image(model, val_arrays, index=0)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Load a checkpoint and evaluate it without retraining."""
    os.makedirs(ARTIFACTS, exist_ok=True)
    model = load_trained_model(args.checkpoint)
    _, _, val_arrays = get_datasets(batch_size=args.batch_size)
    evaluate_and_visualize(
        model,
        val_arrays,
        num_samples=args.num_samples,
        save_path=f"{ARTIFACTS}/predictions.png",
    )


def build_parser() -> argparse.ArgumentParser:
    """Define the CLI surface."""
    parser = argparse.ArgumentParser(description="MNIST single-object detection")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--batch-size", type=int, default=64)
        p.add_argument("--num-samples", type=int, default=10)
        p.add_argument("--subset", type=int, default=None)
        p.add_argument(
            "--checkpoint", type=str, default=f"{ARTIFACTS}/mnist_detector.keras"
        )

    p_inspect = sub.add_parser("inspect", help="visualize ground-truth boxes")
    common(p_inspect)
    p_inspect.set_defaults(func=cmd_inspect)

    p_train = sub.add_parser("train", help="train the detector")
    common(p_train)
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--learning-rate", type=float, default=1e-3)
    p_train.add_argument("--box-weight", type=float, default=10.0)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="evaluate a saved checkpoint")
    common(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    """Parse arguments and dispatch."""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
