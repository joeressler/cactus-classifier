# cactus-classifier

Image classifier to tell true *Trichocereus* cacti apart from common lookalikes (hard negatives).

## Install

Install [uv](https://docs.astral.sh/uv/) if needed, then sync the project environment:

```bash
uv sync
```

The project targets Python 3.11 or newer. PyTorch will use CUDA when an available
CUDA build is installed; otherwise it uses Apple MPS or CPU automatically. For
GPU-specific PyTorch installation needs, follow the official PyTorch installation
selector before running `uv sync`.

## Build the dataset

```bash
uv run python build_dataset.py  # up to 250 research-grade images per class -> ./dataset
```

Useful flags: `--per-class N`, `--size medium|large|original`, `--photos-per-obs N`,
`--licensed-only` (Creative Commons photos only), `--seed N`, `--clean`.

Output:

```
dataset/
  train/<class>/   70%
  val/<class>/     20%
  test/<class>/    10%
  manifest.csv     source URL, license, and attribution for every image
```

The split is grouped by iNaturalist observation, so photos of the same plant never
end up in both train and test.

## Classes

| class | iNaturalist taxon |
|---|---|
| trichocereus_pachanoi | 829212 *Trichocereus macrogonus* var. *pachanoi* |
| trichocereus_peruvianus | 870382 *Trichocereus macrogonus* subsp. *peruvianus* |
| trichocereus_bridgesii | 870017 *Trichocereus bridgesii* |
| miss_cereus_repandus | 320319 *Cereus repandus* |
| miss_myrtillocactus_geometrizans | 274227 *Myrtillocactus geometrizans* |
| miss_pilosocereus_pachycladus | 442108 *Pilosocereus pachycladus* |

Photos belong to their iNaturalist observers. Check `manifest.csv` for each image's license
before redistributing.

Add a seventh `unknown` class with licensed examples from other cacti, succulents, and
unrelated plants:

```bash
uv run python build_unknown_dataset.py --out dataset --per-taxon 20 --seed 42
```

This samples up to 20 photos from each of 12 iNaturalist taxa, keeps one image per
observation, and splits each taxon independently into train/val/test. It only adds
the `unknown` folders and appends their image source, license, and attribution to the
existing manifest. Use `--replace` to replace a previously generated unknown class.
The resulting unknown class represents those sampled taxa; it cannot cover every
possible out-of-scope image.

## Train a classifier

The dataset builder creates the `ImageFolder` layout expected by the trainer. Train
the default pretrained ResNet-18 model with:

```bash
uv run cactus-train train --data dataset --output runs/resnet18
```

Training uses the validation split to select the checkpoint with the best macro F1 by default.
Use `--selection-metric accuracy` when the goal is to maximize validation accuracy.
The test split is not used during training or checkpoint selection. After training,
evaluate the selected checkpoint once on the held-out test split:

```bash
uv run cactus-train evaluate --checkpoint runs/resnet18/best.pt --data dataset --split test --output runs/resnet18/test-metrics.json
```

The report includes accuracy, macro F1, per-class precision/recall/F1, the confusion
matrix, a Trichocereus-versus-known-lookalikes summary, and a majority-class baseline
derived from the training split. If the dataset contains `unknown`, it also reports
unknown-versus-known precision, recall, F1, and confusion matrix. Training writes
`history.json` with per-epoch train and validation metrics.

Evaluation also reports confidence diagnostics: mean top-class softmax score, expected
calibration error (ECE, 15 equal-width bins), multiclass Brier score, and per-bin
accuracy/confidence. Softmax scores are model confidence estimates, not guaranteed
probabilities. To select a post-training rule that maps low-confidence predictions to
`unknown`, tune on validation data and then measure the selected rule on test data:

```bash
uv run cactus-train calibrate --checkpoint runs/convnext-small-unknown-resumed-workers0/best.pt --data dataset --output runs/convnext-small-unknown-resumed-workers0/calibration.json
```

This does not update model weights. It sweeps thresholds from 0.00 to 1.00 in 0.01
increments, selects the validation threshold with the highest unknown-vs-known F1, then
reports test metrics. Ties prefer fewer known-class rejections and then the lower
threshold. A known-class rejection is a lookalike or Trichocereus image labeled
`unknown`; an unknown miss is an `unknown` image labeled as a specific class.

### Selected unknown-aware ConvNeXt Small evaluation

The retained checkpoint is `runs/convnext-small-unknown-resumed-workers0/best.pt`.
On the held-out test split (169 images), accuracy was **87.6%** and macro F1 was
**0.874**. The matrix below uses actual classes as rows and predicted classes as columns;
the class order is the same on both axes.

| Actual \\ Predicted | Cereus repandus | Myrtillocactus | Pilosocereus | T. bridgesii | T. pachanoi | T. peruvianus | Unknown |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cereus repandus | 24 | 0 | 0 | 0 | 0 | 1 | 0 |
| Myrtillocactus | 0 | 23 | 0 | 0 | 0 | 2 | 0 |
| Pilosocereus | 6 | 2 | 15 | 1 | 0 | 1 | 0 |
| T. bridgesii | 0 | 1 | 0 | 23 | 0 | 1 | 0 |
| T. pachanoi | 0 | 0 | 0 | 1 | 21 | 3 | 0 |
| T. peruvianus | 0 | 0 | 0 | 0 | 1 | 20 | 0 |
| Unknown | 1 | 0 | 0 | 0 | 0 | 0 | 22 |

| Class | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| *Cereus repandus* | 25 | 0.774 | 0.960 | 0.857 |
| *Myrtillocactus geometrizans* | 25 | 0.885 | 0.920 | 0.902 |
| *Pilosocereus pachycladus* | 25 | 1.000 | 0.600 | 0.750 |
| *T. bridgesii* | 25 | 0.920 | 0.920 | 0.920 |
| *T. pachanoi* | 25 | 0.955 | 0.840 | 0.894 |
| *T. peruvianus* | 21 | 0.714 | 0.952 | 0.816 |
| Unknown | 23 | 1.000 | 0.957 | 0.978 |

The largest error source was *Pilosocereus pachycladus*: 10 of its 25 images were
misclassified, most often as *Cereus repandus* (6 images). *T. pachanoi* was mistaken
for *T. peruvianus* in 3 of 25 images. The single unknown miss was predicted as
*Cereus repandus*.

Test confidence diagnostics were **91.5% mean top-class confidence**, **87.6% accuracy**,
**0.064 ECE**, and **0.181 multiclass Brier score**. Confidence was not uniformly
reliable: in the 86.7–93.3% confidence bin, 6 of 11 predictions were correct (54.5%).
The very-high-confidence bin (93.3–100%) contained 126 images and was 97.6% accurate.

Validation selected an added rejection threshold of **0.00**, meaning no extra
low-confidence predictions should be converted to `unknown` for this checkpoint under
the stated F1 objective. The trained `unknown` output itself already rejected 22/23
unknown test images (95.7%); it labeled 0/146 known-class test images as unknown. Thus,
the measured unknown miss rate was 1/23 (4.3%) and known-class rejection rate was 0/146.
These unknown results describe the sampled unknown taxa, not every possible plant or
out-of-scope image.

To continue fine-tuning a saved project checkpoint, pass `--init-checkpoint` and
choose a separate output directory. The model name, class list, and image size must
match the checkpoint; the optimizer is restarted with the requested learning rate.
`--epochs` is the number of additional epochs for this continuation.

Available configuration:

```text
--model resnet18|resnet50|resnet101|efficientnet_b0|mobilenet_v3_small|convnext_small|vit_b_16
--epochs 20 --batch-size 32 --image-size 224
--learning-rate 0.001 --weight-decay 0.0001
--selection-metric macro_f1|accuracy
--workers 4 --seed 42 --device auto|cpu|cuda|mps
--pretrained / --no-pretrained
```

For example, a smaller CPU-friendly run without downloading pretrained weights:

```bash
uv run cactus-train train --model mobilenet_v3_small --epochs 5 --batch-size 16 --device cpu --no-pretrained
```
