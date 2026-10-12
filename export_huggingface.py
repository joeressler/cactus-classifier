"""Export the selected cactus checkpoint as a Hugging Face model repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "runs/convnext-small-unknown-resumed-workers0/best.pt"
DEFAULT_OUTPUT = ROOT / "runs/huggingface-export"
ASSETS = ROOT / "huggingface"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != "convnext_small":
        raise SystemExit("This export currently supports the selected ConvNeXt Small checkpoint only.")

    class_names = checkpoint["class_names"]
    image_size = int(checkpoint["image_size"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    # Export only model tensors; the original training checkpoint is not needed
    # for inference and contains additional training metadata.
    torch.save(checkpoint["state_dict"], output / "model.pt")
    config = {
        "model_name": "convnext_small",
        "architecture": "torchvision.models.convnext_small",
        "num_labels": len(class_names),
        "class_names": class_names,
        "id2label": {str(index): name for index, name in enumerate(class_names)},
        "label2id": {name: index for index, name in enumerate(class_names)},
        "image_size": image_size,
        "resize_size": image_size + 32,
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "unknown_class": "unknown",
        "unknown_rejection_threshold": 0.0,
        "checkpoint_selection_metric": checkpoint.get("selection_metric", "macro_f1"),
        "checkpoint_selection_score": checkpoint.get("selection_score"),
        "weights_file": "model.pt",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    for name in ("README.md", "inference.py", "requirements.txt"):
        shutil.copy2(ASSETS / name, output / name)

    for filename in ("test-metrics.json", "calibration.json"):
        metrics_path = args.checkpoint.parent / filename
        if metrics_path.is_file():
            shutil.copy2(metrics_path, output / filename)

    print(f"Prepared Hugging Face model repository at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
