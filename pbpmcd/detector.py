"""Training, monitoring, adaptation, and output handling for PBPMCD."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ExperimentConfig, PARAMETER_PROVENANCE
from .data import (
    DatasetStore,
    FeatureNormalizer,
    Sublog,
    concatenate,
    temporal_case_split,
)
from .metrics import (
    class_performance_distance,
    classification_metrics,
    evaluate_radius_free,
    merge_alarm_bursts,
    resolve_alarm_merge_windows,
)
from .model import PBPMCDPredictor


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and enable deterministic kernels."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    """Resolve `auto`, CPU, or CUDA and reject unavailable CUDA requests."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def make_model(config: ExperimentConfig, num_classes: int) -> PBPMCDPredictor:
    """Construct the next-activity predictor for one detection run."""
    return PBPMCDPredictor(
        num_classes=num_classes,
        activity_embedding_size=config.activity_embedding_size,
        conv_filters=config.conv_filters,
        conv_kernel_size=config.conv_kernel_size,
        hidden_size=config.hidden_size,
        lstm_layers=config.lstm_layers,
        dropout=config.dropout,
    )


def make_loader(
    sublog: Sublog,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create a deterministic mini-batch loader and remove stored CaseIDs."""
    # CaseID is retained in files for leakage-safe splitting but excluded here.
    features = torch.from_numpy(sublog.data[:, :, 1:].astype(np.float32, copy=False))
    labels = torch.from_numpy(sublog.labels.astype(np.int64, copy=False))
    lengths = torch.from_numpy(sublog.lengths.astype(np.int64, copy=False))
    dataset = TensorDataset(features, labels, lengths)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator if shuffle else None,
    )


def train_model(
    model: nn.Module,
    training: Sublog,
    config: ExperimentConfig,
    device: torch.device,
    seed_offset: int = 0,
) -> List[float]:
    """Train one predictor and return the mean loss from each epoch."""
    loader = make_loader(
        training, config.batch_size, True, config.seed + int(seed_offset)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = []
    model.train()
    for epoch in range(config.epochs):
        total_loss, total_count = 0.0, 0
        for features, labels, lengths in loader:
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad()
            logits = model(features, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.size(0)
            total_count += labels.size(0)
        history.append(total_loss / max(total_count, 1))
    return history


def evaluate_model(
    model: nn.Module,
    sublog: Sublog,
    num_classes: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, object]:
    """Return accuracy, per-class recall, labels, and predictions."""
    loader = make_loader(sublog, batch_size, False, 0)
    truth, predictions = [], []
    model.eval()
    with torch.no_grad():
        for features, labels, lengths in loader:
            logits = model(features.to(device), lengths.to(device))
            predicted = torch.argmax(logits, dim=1)
            truth.extend(labels.numpy().astype(int).tolist())
            predictions.extend(predicted.cpu().numpy().astype(int).tolist())
    result = classification_metrics(truth, predictions, num_classes)
    result["truth"] = truth
    result["predictions"] = predictions
    return result


def prodrift_ground_truth(log_length: int) -> List[int]:
    """Return the nine equally spaced reference drifts in a PRODRIFT log."""
    return [round(int(log_length) * ratio / 10) for ratio in range(1, 10)]


def runtime_manifest(device: torch.device) -> dict:
    """Capture software versions, device details, and hashes of core modules."""
    code_hashes = {}
    for name in ("model.py", "data.py", "detector.py", "metrics.py"):
        path = Path(__file__).resolve().parent / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        code_hashes[name] = digest
    gpu_name = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "gpu": gpu_name,
        "code_sha256": code_hashes,
    }


@dataclass
class DetectionRun:
    """Execute the complete PBPMCD lifecycle for one prepared dataset."""

    config: ExperimentConfig
    store: DatasetStore
    output_dir: Path
    ground_truth: List[int]

    def execute(self) -> dict:
        """Train, monitor, adapt, consolidate alarms, evaluate, and persist."""
        set_seed(self.config.seed)
        device = resolve_device(self.config.device)
        num_classes = int(self.store.manifest["activity_count"])
        log_length = int(self.store.manifest["trace_count"])
        effective_merge_windows = resolve_alarm_merge_windows(
            total_windows=len(self.store),
            merge_proportion=self.config.alarm_merge_proportion,
            fixed_windows=self.config.alarm_merge_windows,
        )
        initialization_windows = max(
            1, math.ceil(len(self.store) * self.config.initialization_proportion)
        )
        if initialization_windows >= len(self.store):
            raise ValueError("initialization uses every sublog; no detection windows remain")

        initialization = concatenate(
            [self.store.load(index) for index in range(initialization_windows)]
        )
        initialization_train, initialization_evaluation = temporal_case_split(
            initialization, self.config.train_proportion
        )
        normalizer = FeatureNormalizer.fit(initialization_train)
        initialization_train = normalizer.transform(initialization_train)
        initialization_evaluation = normalizer.transform(initialization_evaluation)

        model = make_model(self.config, num_classes).to(device)
        initial_loss = train_model(
            model, initialization_train, self.config, device, seed_offset=0
        )
        reference = evaluate_model(
            model,
            initialization_evaluation,
            num_classes,
            self.config.batch_size,
            device,
        )
        reference_accuracy = float(reference["accuracy"])
        reference_classes = np.asarray(reference["class_recall"], dtype=float)
        reference_support = np.asarray(reference["class_support"], dtype=int)

        raw_window_indices, window_rows = [], []
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for index in range(initialization_windows, len(self.store)):
            current_raw = self.store.load(index)
            current = normalizer.transform(current_raw)
            current_metrics = evaluate_model(
                model,
                current,
                num_classes,
                self.config.batch_size,
                device,
            )
            current_accuracy = float(current_metrics["accuracy"])
            current_classes = np.asarray(current_metrics["class_recall"], dtype=float)
            current_support = np.asarray(current_metrics["class_support"], dtype=int)
            active_classes = (reference_support > 0) | (current_support > 0)
            accuracy_drop = reference_accuracy - current_accuracy
            class_distance = class_performance_distance(
                reference_classes[active_classes], current_classes[active_classes]
            )
            accuracy_trigger = accuracy_drop >= self.config.theta_p
            class_trigger = class_distance >= self.config.theta_v
            drift = bool(accuracy_trigger or class_trigger)

            localized = index * self.config.window_size
            reported = min((index + 1) * self.config.window_size, log_length)
            row = {
                "window_index": index,
                "trace_start": localized,
                "trace_end": reported,
                "reference_accuracy": reference_accuracy,
                "current_accuracy": current_accuracy,
                "accuracy_drop": accuracy_drop,
                "class_distance": class_distance,
                "accuracy_trigger": accuracy_trigger,
                "class_trigger": class_trigger,
                "drift": drift,
            }

            if drift:
                raw_window_indices.append(index)
                torch.save(
                    {
                        "state_dict": copy.deepcopy(model.state_dict()),
                        "window_index": index,
                        "config": self.config.to_dict(),
                    },
                    checkpoint_dir / f"model_before_drift_{index:04d}.pt",
                )
                adaptation_train, adaptation_evaluation = temporal_case_split(
                    current_raw, self.config.train_proportion
                )
                adaptation_train = normalizer.transform(adaptation_train)
                adaptation_evaluation = normalizer.transform(adaptation_evaluation)
                if self.config.adaptation_mode == "reinitialize":
                    set_seed(self.config.seed + index)
                    model = make_model(self.config, num_classes).to(device)
                elif self.config.adaptation_mode != "finetune":
                    raise ValueError(
                        "adaptation_mode must be 'reinitialize' or 'finetune'"
                    )
                row["adaptation_final_loss"] = train_model(
                    model,
                    adaptation_train,
                    self.config,
                    device,
                    seed_offset=index,
                )[-1]
                adapted = evaluate_model(
                    model,
                    adaptation_evaluation,
                    num_classes,
                    self.config.batch_size,
                    device,
                )
                reference_accuracy = float(adapted["accuracy"])
                reference_classes = np.asarray(adapted["class_recall"], dtype=float)
                reference_support = np.asarray(adapted["class_support"], dtype=int)
                row["adapted_evaluation_accuracy"] = reference_accuracy
            else:
                reference_accuracy = (reference_accuracy + current_accuracy) / 2.0
                reference_classes = (reference_classes + current_classes) / 2.0
                reference_support = np.maximum(reference_support, current_support)
                row["adaptation_final_loss"] = None
                row["adapted_evaluation_accuracy"] = None

            window_rows.append(row)
            print(
                f"window={index:03d}/{len(self.store)-1:03d} "
                f"acc={current_accuracy:.4f} drop={accuracy_drop:.4f} "
                f"class_dist={class_distance:.4f} drift={drift}"
            )

        consolidated_indices, alarm_clusters = merge_alarm_bursts(
            raw_window_indices, effective_merge_windows
        )
        raw_localized_points = [index * self.config.window_size for index in raw_window_indices]
        raw_reported_points = [
            min((index + 1) * self.config.window_size, log_length)
            for index in raw_window_indices
        ]
        localized_points = [
            index * self.config.window_size for index in consolidated_indices
        ]
        reported_points = [
            min((index + 1) * self.config.window_size, log_length)
            for index in consolidated_indices
        ]
        primary = evaluate_radius_free(
            self.ground_truth, reported_points, log_length=log_length
        )
        result = {
            "implementation": "PBPMCD",
            "dataset": self.config.dataset,
            "log_length": log_length,
            "ground_truth": self.ground_truth,
            "initialization_windows": initialization_windows,
            "initialization_trace_fraction": self.config.initialization_proportion,
            "effective_initialization_window_fraction": initialization_windows / len(self.store),
            "total_windows": len(self.store),
            "initialization_training_final_loss": initial_loss[-1],
            "initialization_evaluation_accuracy": float(reference["accuracy"]),
            "normalizer": normalizer.to_dict(),
            "config": self.config.to_dict(),
            "parameter_provenance": PARAMETER_PROVENANCE,
            "runtime": runtime_manifest(device),
            "alarm_consolidation": {
                "formula": "ceil(alpha * total_windows) when alpha > 0",
                "alarm_merge_proportion": self.config.alarm_merge_proportion,
                "fixed_alarm_merge_windows": self.config.alarm_merge_windows,
                "effective_alarm_merge_windows": effective_merge_windows,
                "ground_truth_used": False,
            },
            "raw_alarm_window_indices": raw_window_indices,
            "raw_localized_detections": raw_localized_points,
            "raw_reported_detections": raw_reported_points,
            "alarm_clusters_window_indices": alarm_clusters,
            "consolidated_alarm_window_indices": consolidated_indices,
            "localized_detections": localized_points,
            "reported_detections": reported_points,
            "primary_protocol": {
                "point_type": "reported: end of window required for decision",
                "radius_free": primary,
            },
        }
        self._write_outputs(result, window_rows)
        return result

    def _write_outputs(self, result: dict, window_rows: List[dict]) -> None:
        """Write the JSON result and machine-readable alarm tables."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "result.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        if window_rows:
            with (self.output_dir / "window_metrics.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(window_rows[0].keys()))
                writer.writeheader()
                writer.writerows(window_rows)
        with (self.output_dir / "detections.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["localized_trace", "reported_trace"])
            for localized, reported in zip(
                result["localized_detections"], result["reported_detections"]
            ):
                writer.writerow([localized, reported])
        cluster_by_window = {
            window_index: cluster_id
            for cluster_id, cluster in enumerate(
                result["alarm_clusters_window_indices"], start=1
            )
            for window_index in cluster
        }
        with (self.output_dir / "raw_detections.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["window_index", "localized_trace", "reported_trace", "cluster_id"]
            )
            for window_index, localized, reported in zip(
                result["raw_alarm_window_indices"],
                result["raw_localized_detections"],
                result["raw_reported_detections"],
            ):
                writer.writerow(
                    [window_index, localized, reported, cluster_by_window[window_index]]
                )
