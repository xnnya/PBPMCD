"""Dataset loading, leakage-safe trace splitting, and feature normalization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class Sublog:
    """Prefix-label instances belonging to one trace window."""

    data: np.ndarray
    labels: np.ndarray
    lengths: np.ndarray

    @property
    def case_ids(self) -> np.ndarray:
        # Valid prefixes end in the last row of the left-padded tensor.
        return self.data[:, -1, 0].astype(np.int64)


class DatasetStore:
    """Load prepared trace windows and their dataset manifest on demand."""

    def __init__(self, root: Path, dataset: str, window_size: int) -> None:
        self.dataset_dir = Path(root) / dataset.lower()
        self.window_dir = self.dataset_dir / f"w{int(window_size)}"
        if not self.window_dir.exists():
            raise FileNotFoundError(
                f"missing {self.window_dir}; run pbpmcd.prepare_data first"
            )
        with (self.dataset_dir / "manifest.json").open(encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.paths = sorted(self.window_dir.glob("sublog_*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"no sublogs in {self.window_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def load(self, index: int) -> Sublog:
        with np.load(self.paths[int(index)]) as payload:
            return Sublog(
                data=payload["data"].astype(np.float32, copy=False),
                labels=payload["labels"].astype(np.int64, copy=False),
                lengths=payload["lengths"].astype(np.int64, copy=False),
            )


def concatenate(sublogs: Sequence[Sublog]) -> Sublog:
    """Join multiple trace windows without changing instance order."""
    if not sublogs:
        raise ValueError("at least one sublog is required")
    return Sublog(
        data=np.concatenate([item.data for item in sublogs]),
        labels=np.concatenate([item.labels for item in sublogs]),
        lengths=np.concatenate([item.lengths for item in sublogs]),
    )


def temporal_case_split(sublog: Sublog, train_proportion: float) -> Tuple[Sublog, Sublog]:
    """Split chronologically by CaseID so one case never crosses partitions."""
    if not 0.0 < train_proportion < 1.0:
        raise ValueError("train_proportion must be between zero and one")
    cases = np.unique(sublog.case_ids)
    cases.sort()
    split_index = int(np.floor(cases.size * float(train_proportion)))
    split_index = min(max(split_index, 1), cases.size - 1)
    train_cases, evaluation_cases = cases[:split_index], cases[split_index:]
    train_mask = np.isin(sublog.case_ids, train_cases)
    evaluation_mask = np.isin(sublog.case_ids, evaluation_cases)

    def selected(mask: np.ndarray) -> Sublog:
        return Sublog(
            data=sublog.data[mask],
            labels=sublog.labels[mask],
            lengths=sublog.lengths[mask],
        )

    return selected(train_mask), selected(evaluation_mask)


@dataclass
class FeatureNormalizer:
    """Scale elapsed and gap seconds using training data only."""

    elapsed_scale: float
    gap_scale: float

    @classmethod
    def fit(cls, training: Sublog) -> "FeatureNormalizer":
        valid = training.data[:, :, :].any(axis=-1)
        elapsed = training.data[:, :, 2][valid]
        gap = training.data[:, :, 3][valid]
        return cls(
            elapsed_scale=max(float(elapsed.max(initial=0.0)), 1.0),
            gap_scale=max(float(gap.max(initial=0.0)), 1.0),
        )

    def transform(self, sublog: Sublog) -> Sublog:
        data = sublog.data.copy()
        data[:, :, 2] = np.clip(data[:, :, 2] / self.elapsed_scale, 0.0, 1.0)
        data[:, :, 3] = np.clip(data[:, :, 3] / self.gap_scale, 0.0, 1.0)
        return Sublog(data=data, labels=sublog.labels, lengths=sublog.lengths)

    def to_dict(self) -> dict:
        return {
            "elapsed_scale_seconds": self.elapsed_scale,
            "gap_scale_seconds": self.gap_scale,
            "fit_scope": "initialization training cases only",
        }
