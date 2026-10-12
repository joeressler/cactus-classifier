---
pipeline_tag: image-classification
library_name: pytorch
tags:
  - image-classification
  - computer-vision
  - convnext
  - cactus
  - trichocereus
---

# Trichocereus Cactus Classifier

ConvNeXt Small image classifier for three *Trichocereus* species, three cactus
lookalikes, and a limited `unknown` category trained from other plants.

> **License review required:** No model license is declared yet. Before making this
> repository public, review the licenses and attribution requirements of the source
> images and decide which license, if any, can be applied to the trained weights.
> Training images are not included in this repository.

## Model details

- Architecture: Torchvision ConvNeXt Small, initialized from pretrained ImageNet
  weights and fine-tuned for seven classes.
- Input: RGB image, resized to 256 pixels on the shorter side, center-cropped to
  224 × 224, converted to a tensor, and normalized with ImageNet mean and standard
  deviation.
- Output: scores for each class, sorted from highest to lowest.
- Classes: `miss_cereus_repandus`, `miss_myrtillocactus_geometrizans`,
  `miss_pilosocereus_pachycladus`, `trichocereus_bridgesii`,
  `trichocereus_pachanoi`, `trichocereus_peruvianus`, and `unknown`.

This is a custom PyTorch/Torchvision model. It is not a Transformers `AutoModel` and
requires the included `inference.py` loader. The model files can be hosted on the Hub;
the Hub does not require a model to use Transformers.

## Evaluation

On the held-out test split of 169 images, the model achieved **87.6% accuracy** and
**0.874 macro F1**. For the sampled unknown-plant examples, unknown recall was **95.7%**
(22 of 23), with **0% known-class rejection** (0 of 146). The validation-selected
additional confidence threshold was 0.00, so no extra low-confidence rejection is
applied.

The `unknown` class represents only the sampled other cacti, succulents, and plants.
It does not represent every possible out-of-scope image. In particular, non-plant
objects can receive a confident cactus prediction. Scores are softmax outputs and are
not calibrated probabilities.

Detailed held-out metrics and validation threshold analysis are included in
`test-metrics.json` and `calibration.json`.

## Dataset and training

The training dataset was assembled from research-grade iNaturalist observations.
Splits were grouped by observation to reduce leakage between training and evaluation.
Source image license and attribution records are kept in the project dataset manifest;
those records and the images themselves are not included here.

Training and export code: [cactus-classifier](https://github.com/joeressler/cactus-classifier).

## Inference

Install dependencies from `requirements.txt`, then download this repository with the
Hugging Face Hub client and pass an image to the custom loader:

```python
from huggingface_hub import snapshot_download
from PIL import Image

from inference import load_model, predict

model_dir = snapshot_download("YOUR_USERNAME/cactus-classifier")
model, preprocess, class_names, device = load_model(model_dir)

with Image.open("cactus.jpg") as image:
    scores = predict(image, model, preprocess, class_names, device)

print(scores[0])
```

Replace `YOUR_USERNAME/cactus-classifier` with the final Hub repository ID.
