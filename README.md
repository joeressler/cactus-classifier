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

## Train a classifier

The dataset builder creates the `ImageFolder` layout expected by the trainer. Train
the default pretrained ResNet-18 model with:

```bash
uv run cactus-train train --data dataset --output runs/resnet18
```

Training uses the validation split to select the checkpoint with the best macro F1.
The test split is not used during training or checkpoint selection. After training,
evaluate the selected checkpoint once on the held-out test split:

```bash
uv run cactus-train evaluate --checkpoint runs/resnet18/best.pt --data dataset --split test --output runs/resnet18/test-metrics.json
```

The report includes six-class accuracy, macro F1, per-class precision/recall/F1,
the confusion matrix, a Trichocereus-versus-lookalikes summary, and a majority-class
baseline derived from the training split. Training also writes `history.json` with
the per-epoch train and validation metrics.

To continue fine-tuning a saved project checkpoint, pass `--init-checkpoint` and
choose a separate output directory. The model name, class list, and image size must
match the checkpoint; the optimizer is restarted with the requested learning rate.
`--epochs` is the number of additional epochs for this continuation.

Available configuration:

```text
--model resnet18|resnet50|resnet101|efficientnet_b0|mobilenet_v3_small|convnext_small|vit_b_16
--epochs 20 --batch-size 32 --image-size 224
--learning-rate 0.001 --weight-decay 0.0001
--workers 4 --seed 42 --device auto|cpu|cuda|mps
--pretrained / --no-pretrained
```

For example, a smaller CPU-friendly run without downloading pretrained weights:

```bash
uv run cactus-train train --model mobilenet_v3_small --epochs 5 --batch-size 16 --device cpu --no-pretrained
```
