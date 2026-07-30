# MNIST Single-Object Detection — from scratch in TensorFlow/Keras

A teaching-grade implementation of **simultaneous classification + localization**
without the TensorFlow Object Detection API. Every module is written to be
explained in a viva, not just executed.

```
mnist_detection/
├── dataset/preprocessing.py     # synthesis, normalization, tf.data pipeline
├── utils/visualization.py       # PIL box drawing + Matplotlib grids
├── models/model.py              # shared CNN trunk + 2 heads (Functional API)
├── training/train.py            # compile, callbacks, fit
├── evaluation/metrics.py        # accuracy, box MSE, IoU
├── evaluation/predict.py        # inference + qualitative evaluation
├── tests/test_pipeline.py       # correctness checks for the tricky math
└── main.py                      # CLI
```

## Quick start

```bash
pip install -r mnist_detection/requirements.txt

# 1. ALWAYS look at the data first — red must land exactly on green.
python -m mnist_detection.main inspect

# 2. Sanity test: a correct model MUST overfit 256 samples.
python -m mnist_detection.main train --subset 256 --epochs 30

# 3. Real run.
python -m mnist_detection.main train --epochs 12

# 4. Re-evaluate a checkpoint without retraining.
python -m mnist_detection.main evaluate
```

Artifacts (`ground_truth.png`, `history.png`, `predictions.png`,
`predictions_worst.png`, `single_prediction.png`) land in `artifacts/`.

## The problem in one diagram

```
                                  +--> Dense(10) softmax --> class probs (10,)
 image (75,75,1) --> shared CNN --+
                                  +--> Dense(4)  linear  --> [ymin,xmin,ymax,xmax]
```

One backbone, two objectives, one summed loss:

```
L = w_cls * CategoricalCrossentropy  +  w_box * MSE
```

At the fork the two gradients are **added** before flowing back through the
shared layers — that is the entire mechanism of multi-task learning.

## Concept index (where each idea is explained in depth)

| Concept | File |
| --- | --- |
| Why synthesize a canvas; box normalization math | `dataset/preprocessing.py` |
| `cache` → `shuffle` → `batch` → `prefetch` ordering and why | `dataset/preprocessing.py` |
| One-hot vs sparse targets | `dataset/preprocessing.py` |
| Conv2D / ReLU / AvgPool / Flatten shape+param math | `models/model.py` |
| Why AveragePooling beats MaxPooling for localization | `models/model.py` |
| Why the box head is linear, not sigmoid | `models/model.py` |
| How Keras combines multiple losses; gradient flow | `models/model.py` |
| Loss-scale imbalance and `loss_weights` | `models/model.py` |
| Adam update rule | `models/model.py` |
| IoU formula, boundary cases, why not an IoU loss | `evaluation/metrics.py` |
| Streaming `Metric` vs stateless function | `evaluation/metrics.py` |
| Callbacks: checkpoint / early stopping / LR plateau | `training/train.py` |
| `predict` vs `__call__`, train/inference mode | `evaluation/predict.py` |

## Expected results (12 epochs, `box_weight=10`)

| Metric | Ballpark |
| --- | --- |
| Classification accuracy | 0.98 – 0.99 |
| Box MSE | ~0.0008 – 0.0015 |
| Box RMSE in pixels | ~2 – 3 px on a 75px canvas |
| Mean IoU | 0.80 – 0.88 |
| IoU ≥ 0.5 rate | > 0.97 |

## Debugging playbook

| Symptom | Root cause | Fix |
| --- | --- | --- |
| `loss: nan` on step 1 | pixels still 0–255, or LR too high | divide by 255; drop LR to 1e-4 |
| Accuracy pinned at 0.10 | targets broken / all identical | print one batch of labels |
| `expected ndim=4, found ndim=3` | missing channel or batch axis | `np.expand_dims` |
| Box loss flat, class loss drops | loss scales imbalanced | raise `bounding_box_weight` |
| Boxes mirrored across diagonal | `(x,y)` vs `(y,x)` order mixed | one conversion point: `denormalize_box` |
| IoU > 0 for disjoint boxes | missing `max(0, ...)` clamp | see `compute_iou` |
| Perfect IoU capped near 0.5 | forgot `- intersection` in union | see `compute_iou` |
| Model after `fit` is worse than mid-training | no `restore_best_weights` | it is enabled in `build_callbacks` |
| Cannot overfit 256 samples | bug in model or pipeline, not hyperparams | stop tuning, debug |

## Learning project vs production system

| Here (learning) | Production detector |
| --- | --- |
| Exactly one object per image | Variable object count → anchors / DETR queries |
| Direct absolute coordinate regression | Offsets from anchors or centers |
| MSE box loss | Smooth L1 / GIoU / DIoU (outlier-robust, scale-free) |
| Flatten + big Dense | FCN head + FPN, no resolution-destroying flatten |
| Mean IoU as headline | mAP@[.5:.95] with per-class AP |
| No NMS | NMS / soft-NMS to dedupe overlapping boxes |
| Synthetic pastes | Real annotated data + heavy augmentation |
| 3-layer CNN from scratch | Pretrained backbone (ResNet/EfficientNet/ViT) |
| Keras `.keras` checkpoint | SavedModel/ONNX/TFLite, quantization, versioned serving |
| Eyeballing the grid | Automated regression suite on a frozen eval set |

## Suggested extensions

1. Swap MSE for `tf.keras.losses.Huber()` and compare mean vs median IoU.
2. Replace AveragePooling with MaxPooling and measure the IoU drop — this is
   the cleanest empirical demonstration of the invariance/localization tradeoff.
3. Attach `MeanIoUMetric` to the box head in `compile_model` and watch IoU live.
4. Paste **two** digits and predict two boxes — you will immediately hit the
   assignment problem that anchors and Hungarian matching exist to solve.
5. Ablate multi-task learning: train classification-only and box-only models and
   compare each against the joint model.
