"""Load and run the exported ConvNeXt Small model with PyTorch and Torchvision."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


def load_model(model_dir: str | Path, device: str = "auto"):
    """Load model weights and the preprocessing parameters from an export folder."""
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config["model_name"] != "convnext_small":
        raise ValueError(f"Unsupported model architecture: {config['model_name']}")

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    target = torch.device(device)

    model = models.convnext_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(
        model.classifier[-1].in_features, config["num_labels"],
    )
    state_dict = torch.load(
        model_dir / config["weights_file"], map_location="cpu", weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.to(target).eval()

    preprocess = transforms.Compose([
        transforms.Resize(config["resize_size"]),
        transforms.CenterCrop(config["image_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=config["image_mean"], std=config["image_std"]),
    ])
    return model, preprocess, config["class_names"], target


@torch.inference_mode()
def predict(image: Image.Image, model, preprocess, class_names: list[str], device) -> list[dict]:
    """Return all class scores, ordered from highest to lowest."""
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().tolist()
    return sorted(
        ({"class_name": name, "probability": float(probabilities[index])}
         for index, name in enumerate(class_names)),
        key=lambda score: score["probability"], reverse=True,
    )
