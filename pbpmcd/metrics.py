"""Prediction, alarm consolidation, and causal detection metrics."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

import numpy as np


def resolve_alarm_merge_windows(
    total_windows: int,
    merge_proportion: float = 0.0,
    fixed_windows: int = 0,
) -> int:
    """Resolve a globally normalized alpha to a dataset-specific window gap."""
    total_windows = int(total_windows)
    merge_proportion = float(merge_proportion)
    fixed_windows = int(fixed_windows)
    if total_windows <= 0:
        raise ValueError("total_windows must be positive")
    if not 0.0 <= merge_proportion <= 1.0:
        raise ValueError("alarm_merge_proportion must be between 0 and 1")
    if fixed_windows < 0:
        raise ValueError("alarm_merge_windows must be non-negative")
    if merge_proportion > 0 and fixed_windows > 0:
        raise ValueError(
            "use either alarm_merge_proportion or alarm_merge_windows, not both"
        )
    if merge_proportion > 0:
        return max(1, int(math.ceil(merge_proportion * total_windows)))
    return fixed_windows


def merge_alarm_bursts(
    window_indices: Iterable[int], max_gap_windows: int
) -> Tuple[List[int], List[List[int]]]:
    """Merge nearby raw alarms without consulting ground-truth labels.

    Alarms whose consecutive window-index gap is at most ``max_gap_windows``
    form one burst.  The first alarm is retained as the event-level alarm.
    A value of zero disables merging (apart from exact duplicates).
    """
    max_gap_windows = int(max_gap_windows)
    if max_gap_windows < 0:
        raise ValueError("alarm_merge_windows must be non-negative")
    ordered = sorted(int(value) for value in window_indices)
    if any(value < 0 for value in ordered):
        raise ValueError("alarm window indices must be non-negative")
    if not ordered:
        return [], []
    clusters: List[List[int]] = [[ordered[0]]]
    for value in ordered[1:]:
        if value - clusters[-1][-1] <= max_gap_windows:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [cluster[0] for cluster in clusters], clusters


def classification_metrics(
    truth: Iterable[int], predictions: Iterable[int], num_classes: int
) -> dict:
    """Compute global accuracy and a recall value for every activity."""
    y_true = np.asarray(list(truth), dtype=np.int64)
    y_pred = np.asarray(list(predictions), dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("truth and predictions must have the same shape")
    if y_true.size == 0:
        raise ValueError("classification inputs cannot be empty")

    class_recall = np.zeros(int(num_classes), dtype=np.float64)
    class_support = np.zeros(int(num_classes), dtype=np.int64)
    for class_id in range(int(num_classes)):
        selected = y_true == class_id
        class_support[class_id] = int(selected.sum())
        if class_support[class_id]:
            class_recall[class_id] = float((y_pred[selected] == class_id).mean())
    accuracy = float((y_true == y_pred).mean())
    return {
        "accuracy": accuracy,
        "class_recall": class_recall,
        "class_support": class_support,
    }


def class_performance_distance(reference: np.ndarray, current: np.ndarray) -> float:
    """Return Euclidean distance between two per-activity recall vectors."""
    reference = np.asarray(reference, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if reference.shape != current.shape:
        raise ValueError("class performance vectors must have equal shapes")
    return float(np.linalg.norm(reference - current))


def _sorted_ints(values: Iterable[int]) -> list:
    result = sorted(int(value) for value in values)
    if any(value < 0 for value in result):
        raise ValueError("drift points must be non-negative")
    return result


def evaluate_radius_free(
    ground_truth: Iterable[int],
    detections: Iterable[int],
    log_length: Optional[int] = None,
) -> dict:
    """Causal one-to-one matching without a detection radius.

    The first detection after a ground-truth drift and before the next drift
    is a TP.  Extra detections are FP.  Mean Delay is calculated on TPs only.
    """
    truth = _sorted_ints(ground_truth)
    detected = _sorted_ints(detections)
    matched = [None] * len(truth)
    false_positives = []
    for point in detected:
        match_index = None
        for index, start in enumerate(truth):
            end = truth[index + 1] if index + 1 < len(truth) else log_length
            before_end = end is None or point < end
            if index + 1 == len(truth) and end is not None:
                before_end = point <= end
            if point >= start and before_end:
                match_index = index
                break
        if match_index is None or matched[match_index] is not None:
            false_positives.append(point)
        else:
            matched[match_index] = point
    matches = [
        {
            "ground_truth": truth[index],
            "detection": int(point),
            "delay": int(point) - truth[index],
        }
        for index, point in enumerate(matched)
        if point is not None
    ]
    missed = [truth[index] for index, point in enumerate(matched) if point is None]
    tp, fp, fn = len(matches), len(false_positives), len(missed)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_delay = float(np.mean([item["delay"] for item in matches])) if tp else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_delay": mean_delay,
        "delay_unit": "traces",
        "matches": matches,
        "false_positive_points": false_positives,
        "missed_ground_truth_points": missed,
    }
