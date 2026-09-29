#!/usr/bin/env python3
"""Train CIFAR ResNet-18 with dynamically scheduled IES re-evaluations."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

# Permit direct execution from a source checkout without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dynamic_ies import (  # noqa: E402
    AdaptiveRatioSchedule,
    MasteryTracker,
    period_for_epoch,
    period_from_learning_rate,
)


@dataclass
class RunResult:
    seed: int
    best_accuracy_pct: float
    saved_backprop_ratio_pct: float
    elapsed_seconds: float


class IndexedSubset(Dataset):
    """Expose stable original sample IDs for an arbitrary index subset."""

    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        sample_id = self.indices[position]
        image, label = self.dataset[sample_id]
        return image, label, sample_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument(
        "--schedule",
        choices=("constant", "linear", "log", "sqrt", "adaptive-ratio", "lr-aware"),
        default="lr-aware",
    )
    parser.add_argument("--period", type=int, default=1, help="Period for constant schedule")
    parser.add_argument("--max-period", type=int, default=5)
    parser.add_argument("--lr-decay", choices=("fixed", "linear", "exponential"), default="linear")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--adaptive-threshold", type=float, default=0.02)
    parser.add_argument("--moving-average", type=int, default=3)
    parser.add_argument("--criterion-window", type=int, default=1)
    parser.add_argument("--switch-to-first-order", action="store_true")
    parser.add_argument("--first-order-threshold", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/summary.json"))
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_datasets(name: str, root: Path):
    stats = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        "cifar100": ((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    }
    mean, std = stats[name]
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    dataset_type = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    train = dataset_type(root=root, train=True, download=True, transform=train_transform)
    train_eval = dataset_type(root=root, train=True, download=False, transform=eval_transform)
    test = dataset_type(root=root, train=False, download=True, transform=eval_transform)
    return train, train_eval, test


def make_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def make_optimizer(model: nn.Module, args: argparse.Namespace):
    default_lr = 0.001 if args.lr_decay == "fixed" else 0.1
    learning_rate = args.learning_rate if args.learning_rate is not None else default_lr
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    if args.lr_decay == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.01, total_iters=min(150, args.epochs)
        )
    elif args.lr_decay == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    return optimizer, scheduler


def make_loader(
    dataset: Dataset,
    indices: set[int],
    args: argparse.Namespace,
    *,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        IndexedSubset(dataset, sorted(indices)),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
    return 100.0 * correct / total


def schedule_period(
    args: argparse.Namespace,
    epoch: int,
    learning_rate: float,
    adaptive: AdaptiveRatioSchedule,
) -> int:
    if args.schedule == "lr-aware":
        if args.lr_decay == "fixed":
            raise ValueError("lr-aware schedule requires linear or exponential LR decay")
        return period_from_learning_rate(
            learning_rate, args.lr_decay, max_period=args.max_period
        )
    if args.schedule == "adaptive-ratio":
        return adaptive.period
    return period_for_epoch(
        epoch,
        args.schedule,
        constant_period=args.period,
        max_period=args.max_period,
    )


def run_once(args: argparse.Namespace, seed: int, device: torch.device) -> RunResult:
    set_seed(seed)
    train_data, train_eval_data, test_data = make_datasets(args.dataset, args.data_dir)
    all_ids = set(range(len(train_data)))
    active_ids = all_ids.copy()
    mastered_ids: set[int] = set()

    model = make_model(10 if args.dataset == "cifar10" else 100).to(device)
    optimizer, scheduler = make_optimizer(model, args)
    criterion = nn.CrossEntropyLoss(reduction="none")
    tracker = MasteryTracker(
        threshold=args.threshold,
        derivative_order=2,
        smoothing_window=args.moving_average,
        criterion_window=args.criterion_window,
    )
    adaptive = AdaptiveRatioSchedule(
        threshold=args.adaptive_threshold, max_period=args.max_period
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    best_accuracy = 0.0
    skipped_backprops = 0
    switched = False
    start = time.perf_counter()

    for epoch in range(args.epochs):
        learning_rate = optimizer.param_groups[0]["lr"]
        period = schedule_period(args, epoch, learning_rate, adaptive)

        if args.switch_to_first_order and period == args.max_period and not switched:
            tracker.switch_to_first_order(args.first_order_threshold)
            switched = True

        model.train()
        if active_ids:
            active_loader = make_loader(train_data, active_ids, args, shuffle=True)
            for images, labels, sample_ids in active_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                losses = criterion(model(images), labels)
                losses.mean().backward()
                optimizer.step()
                for sample_id, loss in zip(sample_ids.tolist(), losses.detach().cpu().tolist()):
                    tracker.observe(sample_id, loss)

        # Re-evaluate mastered samples without backpropagation only when scheduled.
        if mastered_ids and (epoch + 1) % period == 0:
            mastered_loader = make_loader(train_eval_data, mastered_ids, args, shuffle=False)
            model.eval()
            with torch.no_grad():
                for images, labels, sample_ids in mastered_loader:
                    losses = criterion(model(images.to(device)), labels.to(device))
                    for sample_id, loss in zip(
                        sample_ids.tolist(), losses.detach().cpu().tolist()
                    ):
                        tracker.observe(sample_id, loss)

        if args.threshold > 0:
            mastered_ids = tracker.mastered(all_ids)
            active_ids = all_ids - mastered_ids
        else:
            mastered_ids = set()
            active_ids = all_ids.copy()

        skipped_backprops += len(mastered_ids)
        if args.schedule == "adaptive-ratio" and (epoch + 1) % 10 == 0:
            adaptive.update(len(mastered_ids) / len(all_ids))

        scheduler.step()
        best_accuracy = max(best_accuracy, evaluate(model, test_loader, device))
        print(
            f"run={seed + 1}/{args.runs} epoch={epoch + 1:03d}/{args.epochs} "
            f"acc={best_accuracy:.2f}% active={len(active_ids)} "
            f"period={period} lr={learning_rate:.5g}"
        )

    elapsed = time.perf_counter() - start
    return RunResult(
        seed=seed,
        best_accuracy_pct=best_accuracy,
        saved_backprop_ratio_pct=100.0 * skipped_backprops / (len(all_ids) * args.epochs),
        elapsed_seconds=elapsed,
    )


def mean_ci(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    ci = 0.0 if len(values) == 1 else 1.96 * statistics.pstdev(values) / math.sqrt(len(values))
    return {"mean": mean, "ci_95": ci}


def main() -> None:
    args = parse_args()
    if args.runs < 1 or args.epochs < 1:
        raise ValueError("runs and epochs must be positive")
    device = resolve_device(args.device)
    print(f"device={device} dataset={args.dataset} schedule={args.schedule}")

    runs = [run_once(args, seed, device) for seed in range(args.runs)]
    summary = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": str(device),
        "runs": [asdict(run) for run in runs],
        "aggregate": {
            "best_accuracy_pct": mean_ci([run.best_accuracy_pct for run in runs]),
            "saved_backprop_ratio_pct": mean_ci(
                [run.saved_backprop_ratio_pct for run in runs]
            ),
            "elapsed_seconds": mean_ci([run.elapsed_seconds for run in runs]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
