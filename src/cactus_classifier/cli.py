"""Train and evaluate image classifiers for the cactus dataset."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


MODEL_NAMES = ("resnet18", "resnet50", "resnet101", "efficientnet_b0", "mobilenet_v3_small", "convnext_small", "vit_b_16")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    constructor = getattr(models, name)
    weights = None
    if pretrained:
        weights = models.get_model_weights(constructor).DEFAULT
    model = constructor(weights=weights)
    if name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name in ("mobilenet_v3_small", "convnext_small"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name == "vit_b_16":
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model


def make_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
        transforms.ToTensor(),
        normalize,
    ])
    evaluation = transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train, evaluation


def make_loader(path: Path, transform: transforms.Compose, batch_size: int,
                workers: int, shuffle: bool) -> DataLoader:
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset split directory does not exist: {path}")
    dataset = datasets.ImageFolder(path, transform=transform)
    if not dataset.classes:
        raise ValueError(f"No class folders found under {path}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=torch.cuda.is_available())


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    seen = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += labels.size(0)
    return loss_sum / max(seen, 1), correct / max(seen, 1)


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int], float]:
    model.eval()
    actual: list[int] = []
    predicted: list[int] = []
    loss_sum = 0.0
    seen = 0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        actual.extend(labels.cpu().tolist())
        predicted.extend(logits.argmax(dim=1).cpu().tolist())
        seen += labels.size(0)
    return actual, predicted, loss_sum / max(seen, 1)


def report(actual: list[int], predicted: list[int], class_names: list[str], loss: float,
           majority_class: int | None = None) -> dict:
    details = classification_report(actual, predicted, labels=list(range(len(class_names))),
                                    target_names=class_names, output_dict=True, zero_division=0)
    result = {
        "loss": loss,
        "accuracy": details["accuracy"],
        "macro_f1": details["macro avg"]["f1-score"],
        "per_class": {name: details[name] for name in class_names},
        "confusion_matrix": confusion_matrix(actual, predicted,
                                                labels=list(range(len(class_names)))).tolist(),
    }
    trichocereus_ids = [i for i, name in enumerate(class_names) if name.startswith("trichocereus_")]
    if trichocereus_ids:
        trichocereus = set(trichocereus_ids)
        binary_actual = [int(value in trichocereus) for value in actual]
        binary_predicted = [int(value in trichocereus) for value in predicted]
        binary = classification_report(binary_actual, binary_predicted, labels=[1, 0],
                                       target_names=["trichocereus", "lookalike"],
                                       output_dict=True, zero_division=0)
        result["trichocereus_vs_lookalikes"] = {
            "accuracy": binary["accuracy"],
            "trichocereus_precision": binary["trichocereus"]["precision"],
            "trichocereus_recall": binary["trichocereus"]["recall"],
            "trichocereus_f1": binary["trichocereus"]["f1-score"],
            "confusion_matrix": confusion_matrix(binary_actual, binary_predicted,
                                                    labels=[1, 0]).tolist(),
        }
    if majority_class is not None:
        baseline = [majority_class] * len(actual)
        result["majority_class_baseline"] = {
            "class": class_names[majority_class],
            "accuracy": sum(a == p for a, p in zip(actual, baseline)) / max(len(actual), 1),
            "macro_f1": classification_report(actual, baseline,
                                                labels=list(range(len(class_names))),
                                                target_names=class_names, output_dict=True,
                                                zero_division=0)["macro avg"]["f1-score"],
        }
    return result


def train(args: argparse.Namespace) -> int:
    seed_everything(args.seed)
    device = get_device(args.device)
    train_transform, eval_transform = make_transforms(args.image_size)
    train_loader = make_loader(args.data / "train", train_transform, args.batch_size,
                               args.workers, True)
    val_loader = make_loader(args.data / "val", eval_transform, args.batch_size,
                             args.workers, False)
    class_names = train_loader.dataset.classes
    if val_loader.dataset.classes != class_names:
        raise ValueError("Train and validation folders must contain the same classes in the same order")

    initial_checkpoint = None
    if args.init_checkpoint:
        initial_checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        if initial_checkpoint.get("model_name") != args.model:
            raise ValueError("Initial checkpoint model does not match --model")
        if initial_checkpoint.get("class_names") != class_names:
            raise ValueError("Initial checkpoint classes do not match the training dataset")
        if initial_checkpoint.get("image_size") != args.image_size:
            raise ValueError("Initial checkpoint image size does not match --image-size")

    model = make_model(args.model, len(class_names), args.pretrained and initial_checkpoint is None).to(device)
    if initial_checkpoint is not None:
        model.load_state_dict(initial_checkpoint["state_dict"])
    class_counts = [0] * len(class_names)
    for _, class_index in train_loader.dataset.samples:
        class_counts[class_index] += 1
    majority_class = max(range(len(class_counts)), key=class_counts.__getitem__)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    best_path = output / "best.pt"
    history: list[dict] = []

    print(f"Device: {device}; model: {args.model}; classes: {', '.join(class_names)}")
    if initial_checkpoint is not None:
        initial_actual, initial_predicted, initial_val_loss = predict(model, val_loader, device)
        initial_metrics = report(initial_actual, initial_predicted, class_names, initial_val_loss)
        best_macro_f1 = initial_metrics["macro_f1"]
        torch.save({"model_name": args.model, "class_names": class_names,
                    "image_size": args.image_size, "state_dict": model.state_dict(),
                    "val_accuracy": initial_metrics["accuracy"],
                    "val_macro_f1": best_macro_f1, "train_class_counts": class_counts,
                    "majority_class": majority_class,
                    "initialized_from": str(args.init_checkpoint)}, best_path)
        print(f"Starting checkpoint validation macro F1: {best_macro_f1:.3f}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion,
                                                      optimizer, device)
        actual, predicted, val_loss = predict(model, val_loader, device)
        val_metrics = report(actual, predicted, class_names, val_loss)
        val_accuracy = val_metrics["accuracy"]
        val_macro_f1 = val_metrics["macro_f1"]
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
               "val_loss": val_loss, "val_accuracy": val_accuracy,
               "val_macro_f1": val_macro_f1}
        history.append(row)
        print(f"Epoch {epoch:03d}/{args.epochs}: train loss {train_loss:.4f}, "
              f"accuracy {train_accuracy:.3f}; val loss {val_loss:.4f}, "
              f"accuracy {val_accuracy:.3f}, macro F1 {val_macro_f1:.3f}")
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            torch.save({"model_name": args.model, "class_names": class_names,
                        "image_size": args.image_size, "state_dict": model.state_dict(),
                        "val_accuracy": val_accuracy, "val_macro_f1": val_macro_f1,
                        "train_class_counts": class_counts,
                        "majority_class": majority_class,
                        "initialized_from": str(args.init_checkpoint) if args.init_checkpoint else None},
                       best_path)

    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {best_path} (validation macro F1 {best_macro_f1:.3f})")
    return 0


def evaluate(args: argparse.Namespace) -> int:
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    class_names = checkpoint["class_names"]
    _, eval_transform = make_transforms(checkpoint["image_size"])
    loader = make_loader(args.data / args.split, eval_transform, args.batch_size,
                         args.workers, False)
    if loader.dataset.classes != class_names:
        raise ValueError("Evaluation class folders do not match the checkpoint classes")
    model = make_model(checkpoint["model_name"], len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    actual, predicted, loss = predict(model, loader, device)
    metrics = report(actual, predicted, class_names, loss,
                     majority_class=checkpoint.get("majority_class"))
    print(json.dumps(metrics, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cactus-train",
                                     description="Train and evaluate cactus image classifiers")
    commands = parser.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train", help="train a model and save the best validation checkpoint")
    train_parser.add_argument("--data", type=Path, default=Path("dataset"), help="ImageFolder dataset root")
    train_parser.add_argument("--output", type=Path, default=Path("runs/experiment"), help="output directory")
    train_parser.add_argument("--init-checkpoint", type=Path,
                              help="initialize from a saved checkpoint for fine-tuning")
    train_parser.add_argument("--model", choices=MODEL_NAMES, default="resnet18")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--workers", type=int, default=4)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    train_parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                              help="initialize from ImageNet weights (use --no-pretrained offline)")
    train_parser.set_defaults(func=train)

    eval_parser = commands.add_parser("evaluate", help="evaluate a saved checkpoint on a dataset split")
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--data", type=Path, default=Path("dataset"))
    eval_parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    eval_parser.add_argument("--batch-size", type=int, default=32)
    eval_parser.add_argument("--workers", type=int, default=4)
    eval_parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    eval_parser.add_argument("--output", type=Path, help="optional path to write metrics JSON")
    eval_parser.set_defaults(func=evaluate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 2
