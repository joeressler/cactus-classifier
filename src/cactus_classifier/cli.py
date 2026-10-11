"""Train and evaluate image classifiers for the cactus dataset."""

from __future__ import annotations

import argparse
import json
import random
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


@torch.inference_mode()
def predict_with_probabilities(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> tuple[list[int], np.ndarray, float]:
    """Return targets, softmax probabilities, and mean cross-entropy loss."""
    model.eval()
    actual: list[int] = []
    probabilities: list[np.ndarray] = []
    loss_sum = 0.0
    seen = 0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy())
        actual.extend(labels.cpu().tolist())
        seen += labels.size(0)
    return actual, np.asarray(probabilities), loss_sum / max(seen, 1)


def confidence_metrics(actual: list[int], probabilities: np.ndarray,
                       bins: int = 15) -> dict:
    """Summarize top-label confidence calibration and multiclass Brier score."""
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == np.asarray(actual)
    edges = np.linspace(0.0, 1.0, bins + 1)
    calibration_bins = []
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        in_bin = (confidence >= lower) & (confidence < upper if index < bins - 1
                                          else confidence <= upper)
        count = int(in_bin.sum())
        if not count:
            continue
        accuracy = float(correct[in_bin].mean())
        mean_confidence = float(confidence[in_bin].mean())
        ece += count / len(actual) * abs(accuracy - mean_confidence)
        calibration_bins.append({
            "lower_confidence": float(lower),
            "upper_confidence": float(upper),
            "count": count,
            "accuracy": accuracy,
            "mean_confidence": mean_confidence,
        })
    targets = np.eye(probabilities.shape[1], dtype=np.float64)[np.asarray(actual)]
    return {
        "mean_top_confidence": float(confidence.mean()),
        "accuracy_at_top_confidence": float(correct.mean()),
        "expected_calibration_error_15_bins": float(ece),
        "multiclass_brier_score": float(np.mean(np.sum((probabilities - targets) ** 2, axis=1))),
        "calibration_bins": calibration_bins,
    }


def threshold_metrics(actual: list[int], probabilities: np.ndarray,
                      class_names: list[str], threshold: float) -> dict:
    """Map low-confidence predictions to unknown and summarize the tradeoff."""
    if "unknown" not in class_names:
        raise ValueError("Threshold-based rejection requires an 'unknown' class")
    unknown_id = class_names.index("unknown")
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    predicted[confidence < threshold] = unknown_id
    actual_array = np.asarray(actual)
    is_unknown = actual_array == unknown_id
    predicted_unknown = predicted == unknown_id
    unknown_count = int(is_unknown.sum())
    known_count = len(actual) - unknown_count
    tp = int(np.sum(is_unknown & predicted_unknown))
    fp = int(np.sum(~is_unknown & predicted_unknown))
    fn = unknown_count - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(unknown_count, 1)
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(predicted == actual_array)),
        "macro_f1": float(classification_report(
            actual, predicted, labels=list(range(len(class_names))),
            target_names=class_names, output_dict=True, zero_division=0,
        )["macro avg"]["f1-score"]),
        "unknown_precision": precision,
        "unknown_recall": recall,
        "unknown_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "known_rejection_rate": fp / max(known_count, 1),
        "unknown_miss_rate": fn / max(unknown_count, 1),
        "unknown_rejection_rate": recall,
        "known_count": known_count,
        "unknown_count": unknown_count,
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=list(range(len(class_names))),
        ).tolist(),
    }


def choose_unknown_threshold(actual: list[int], probabilities: np.ndarray,
                             class_names: list[str]) -> dict:
    """Select the validation threshold maximizing unknown-vs-known F1.

    Ties prefer fewer known examples rejected, then the lower threshold.
    """
    candidates = np.linspace(0.0, 1.0, 101)
    results = [threshold_metrics(actual, probabilities, class_names, float(value))
               for value in candidates]
    return max(results, key=lambda row: (row["unknown_f1"],
                                         -row["known_rejection_rate"],
                                         -row["threshold"]))


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
    lookalike_ids = [i for i, name in enumerate(class_names) if name.startswith("miss_")]
    if trichocereus_ids and lookalike_ids:
        trichocereus = set(trichocereus_ids)
        known_ids = trichocereus | set(lookalike_ids)
        known_positions = [i for i, value in enumerate(actual) if value in known_ids]
        known_actual = [actual[i] for i in known_positions]
        known_predicted = [predicted[i] for i in known_positions]
        binary_actual = [int(value in trichocereus) for value in known_actual]
        binary_predicted = [int(value in trichocereus) for value in known_predicted]
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
    unknown_ids = {i for i, name in enumerate(class_names) if name == "unknown"}
    if unknown_ids:
        unknown_actual = [int(value in unknown_ids) for value in actual]
        unknown_predicted = [int(value in unknown_ids) for value in predicted]
        unknown_report = classification_report(
            unknown_actual, unknown_predicted, labels=[1, 0],
            target_names=["unknown", "known"], output_dict=True, zero_division=0,
        )
        result["unknown_vs_known"] = {
            "accuracy": unknown_report["accuracy"],
            "unknown_precision": unknown_report["unknown"]["precision"],
            "unknown_recall": unknown_report["unknown"]["recall"],
            "unknown_f1": unknown_report["unknown"]["f1-score"],
            "confusion_matrix": confusion_matrix(unknown_actual, unknown_predicted,
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
    best_selection_score = -1.0
    best_path = output / "best.pt"
    history: list[dict] = []

    print(f"Device: {device}; model: {args.model}; classes: {', '.join(class_names)}")
    if initial_checkpoint is not None:
        initial_actual, initial_predicted, initial_val_loss = predict(model, val_loader, device)
        initial_metrics = report(initial_actual, initial_predicted, class_names, initial_val_loss)
        best_selection_score = initial_metrics[args.selection_metric]
        torch.save({"model_name": args.model, "class_names": class_names,
                    "image_size": args.image_size, "state_dict": model.state_dict(),
                    "val_accuracy": initial_metrics["accuracy"],
                    "val_macro_f1": initial_metrics["macro_f1"],
                    "selection_metric": args.selection_metric,
                    "selection_score": best_selection_score,
                    "train_class_counts": class_counts,
                    "majority_class": majority_class,
                    "initialized_from": str(args.init_checkpoint)}, best_path)
        print(f"Starting checkpoint validation {args.selection_metric}: {best_selection_score:.3f}")

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
        selection_score = val_metrics[args.selection_metric]
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            torch.save({"model_name": args.model, "class_names": class_names,
                        "image_size": args.image_size, "state_dict": model.state_dict(),
                        "val_accuracy": val_accuracy, "val_macro_f1": val_macro_f1,
                        "selection_metric": args.selection_metric,
                        "selection_score": best_selection_score,
                        "train_class_counts": class_counts,
                        "majority_class": majority_class,
                        "initialized_from": str(args.init_checkpoint) if args.init_checkpoint else None},
                       best_path)

    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {best_path} (validation {args.selection_metric} {best_selection_score:.3f})")
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
    actual, probabilities, loss = predict_with_probabilities(model, loader, device)
    predicted = probabilities.argmax(axis=1).tolist()
    metrics = report(actual, predicted, class_names, loss,
                     majority_class=checkpoint.get("majority_class"))
    metrics["confidence"] = confidence_metrics(actual, probabilities)
    print(json.dumps(metrics, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return 0


def calibrate(args: argparse.Namespace) -> int:
    """Tune an unknown-rejection threshold on validation, then report test results."""
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    class_names = checkpoint["class_names"]
    if "unknown" not in class_names:
        raise ValueError("Calibration requires a checkpoint trained with an 'unknown' class")
    _, eval_transform = make_transforms(checkpoint["image_size"])
    model = make_model(checkpoint["model_name"], len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["state_dict"])

    split_outputs: dict[str, tuple[list[int], np.ndarray, float]] = {}
    for split in ("val", "test"):
        loader = make_loader(args.data / split, eval_transform, args.batch_size,
                             args.workers, False)
        if loader.dataset.classes != class_names:
            raise ValueError(f"{split.title()} class folders do not match the checkpoint classes")
        split_outputs[split] = predict_with_probabilities(model, loader, device)

    val_actual, val_probabilities, _ = split_outputs["val"]
    selected = choose_unknown_threshold(val_actual, val_probabilities, class_names)
    test_actual, test_probabilities, test_loss = split_outputs["test"]
    test_predictions = test_probabilities.argmax(axis=1).tolist()
    baseline = report(test_actual, test_predictions, class_names, test_loss,
                      majority_class=checkpoint.get("majority_class"))
    thresholded = threshold_metrics(test_actual, test_probabilities, class_names,
                                    selected["threshold"])
    result = {
        "threshold_selection": "maximum validation unknown-vs-known F1; ties prefer fewer known rejections, then lower threshold",
        "validation": selected,
        "test_argmax": {
            "accuracy": baseline["accuracy"],
            "macro_f1": baseline["macro_f1"],
            "unknown_vs_known": baseline.get("unknown_vs_known"),
        },
        "test_thresholded": thresholded,
        "test_confidence": confidence_metrics(test_actual, test_probabilities),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
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
    train_parser.add_argument("--selection-metric", choices=("accuracy", "macro_f1"),
                              default="macro_f1",
                              help="validation metric used to select the best checkpoint")
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

    calibration_parser = commands.add_parser(
        "calibrate", help="select an unknown-rejection confidence threshold on validation data",
    )
    calibration_parser.add_argument("--checkpoint", type=Path, required=True)
    calibration_parser.add_argument("--data", type=Path, default=Path("dataset"))
    calibration_parser.add_argument("--batch-size", type=int, default=32)
    calibration_parser.add_argument("--workers", type=int, default=4)
    calibration_parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    calibration_parser.add_argument("--output", type=Path,
                                    help="optional path to write calibration and test metrics JSON")
    calibration_parser.set_defaults(func=calibrate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 2
