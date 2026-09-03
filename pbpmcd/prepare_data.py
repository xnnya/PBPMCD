"""Convert MXML or event-log CSV files into PBPMCD trace windows.

Output is written under the selected data root. Raw elapsed and gap seconds
are retained; normalization is fitted later on initialization-training cases.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
from xml.etree import ElementTree as ET

import numpy as np


Event = Tuple[str, str, str]
Trace = List[Event]


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_timestamp(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def parse_mxml(path: Path) -> List[Trace]:
    """Read an MXML file and return traces ordered by first timestamp."""
    traces: List[Trace] = []
    for _, element in ET.iterparse(str(path), events=("end",)):
        if _local_name(element.tag) != "ProcessInstance":
            continue
        trace: Trace = []
        for entry in element:
            if _local_name(entry.tag) != "AuditTrailEntry":
                continue
            values: Dict[str, str] = {}
            for item in entry:
                if item.text:
                    values[_local_name(item.tag)] = item.text.strip()
            activity = values.get("WorkflowModelElement")
            event_type = values.get("EventType", "")
            timestamp = values.get("Timestamp") or values.get("timestamp")
            if activity and timestamp and event_type.lower() != "assign":
                trace.append((activity, event_type, timestamp))
        if trace:
            traces.append(trace)
        element.clear()
    if not traces:
        raise ValueError(f"no usable traces found in {path}")
    # Monitoring order is determined by each trace's first event.
    traces.sort(key=lambda trace: parse_timestamp(trace[0][2]))
    return traces


def parse_csv(path: Path) -> List[Trace]:
    """Read a CSV event log with CaseID, Activity, Type, and Timestamp columns."""
    traces_by_case: Dict[str, Trace] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"CaseID", "Activity", "Type", "Timestamp"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            case_id = (row.get("CaseID") or "").strip()
            activity = (row.get("Activity") or "").strip()
            event_type = (row.get("Type") or "").strip()
            timestamp = (row.get("Timestamp") or "").strip()
            if not case_id or not activity or not timestamp:
                raise ValueError(f"{path}:{row_number} contains an empty required value")
            parse_timestamp(timestamp)
            if event_type.casefold() != "assign":
                traces_by_case.setdefault(case_id, []).append(
                    (activity, event_type, timestamp)
                )
    traces = [trace for trace in traces_by_case.values() if trace]
    if not traces:
        raise ValueError(f"no usable traces found in {path}")
    traces.sort(key=lambda trace: parse_timestamp(trace[0][2]))
    return traces


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_traces(
    traces: Sequence[Trace], output_csv: Path
) -> Tuple[List[List[int]], List[List[datetime]], Dict[str, int]]:
    names = sorted({event[0] for trace in traces for event in trace})
    mapping = {name: index for index, name in enumerate(names)}
    activities: List[List[int]] = []
    timestamps: List[List[datetime]] = []
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["CaseID", "Activity", "Type", "Timestamp"])
        for case_id, trace in enumerate(traces):
            activity_trace, timestamp_trace = [], []
            for activity, event_type, timestamp_text in trace:
                activity_id = mapping[activity]
                writer.writerow([case_id, activity_id, event_type, timestamp_text])
                activity_trace.append(activity_id)
                timestamp_trace.append(parse_timestamp(timestamp_text))
            activities.append(activity_trace)
            timestamps.append(timestamp_trace)
    return activities, timestamps, mapping


def build_window(
    case_start: int,
    activity_traces: Sequence[Sequence[int]],
    timestamp_traces: Sequence[Sequence[datetime]],
    max_trace_length: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples, labels, lengths = [], [], []
    feature_count = 35
    for offset, (activities, timestamps) in enumerate(
        zip(activity_traces, timestamp_traces)
    ):
        case_id = case_start + offset
        for prefix_length in range(1, len(activities)):
            prefix = np.zeros((max_trace_length, feature_count), dtype=np.float32)
            rows = []
            for index in range(prefix_length):
                current = timestamps[index]
                previous = timestamps[index - 1] if index else current
                elapsed = max(0.0, (current - timestamps[0]).total_seconds())
                gap = max(0.0, (current - previous).total_seconds())
                hour = [1.0 if value == current.hour else 0.0 for value in range(24)]
                weekday = [
                    1.0 if value == current.weekday() else 0.0
                    for value in range(7)
                ]
                rows.append(
                    [
                        float(case_id),
                        float(activities[index]),
                        float(elapsed),
                        float(gap),
                        *hour,
                        *weekday,
                    ]
                )
            prefix[-prefix_length:, :] = np.asarray(rows, dtype=np.float32)
            samples.append(prefix)
            labels.append(int(activities[prefix_length]))
            lengths.append(prefix_length)
    return (
        np.asarray(samples, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(lengths, dtype=np.int64),
    )


def prepare_one(
    source: Path, output_root: Path, window_sizes: Sequence[int]
) -> None:
    dataset = source.stem.lower()
    dataset_dir = output_root / dataset
    print(f"Parsing {source}")
    suffix = source.suffix.casefold()
    if suffix == ".mxml":
        traces = parse_mxml(source)
    elif suffix == ".csv":
        traces = parse_csv(source)
    else:
        raise ValueError(f"unsupported event-log format: {source.suffix}")
    activities, timestamps, mapping = encode_traces(
        traces, dataset_dir / f"{dataset}.csv"
    )
    max_trace_length = max(len(trace) for trace in activities)

    window_manifests = {}
    for window_size in window_sizes:
        if window_size <= 0:
            raise ValueError("window sizes must be positive")
        window_dir = dataset_dir / f"w{window_size}"
        window_dir.mkdir(parents=True, exist_ok=True)
        sublog_count = (len(traces) + window_size - 1) // window_size
        total_prefixes = 0
        for index in range(sublog_count):
            start = index * window_size
            end = min((index + 1) * window_size, len(traces))
            data, labels, lengths = build_window(
                start,
                activities[start:end],
                timestamps[start:end],
                max_trace_length,
            )
            np.savez_compressed(
                window_dir / f"sublog_{index:04d}.npz",
                data=data,
                labels=labels,
                lengths=lengths,
            )
            total_prefixes += int(labels.size)
            print(
                f"  {dataset} w={window_size}: {index + 1}/{sublog_count}, "
                f"traces={start}:{end}, prefixes={labels.size}"
            )
        window_manifests[str(window_size)] = {
            "sublog_count": sublog_count,
            "total_prefixes": total_prefixes,
        }

    manifest = {
        "status": "ready",
        "source": str(source),
        "source_sha256": _sha256(source),
        "dataset": dataset,
        "trace_count": len(traces),
        "event_count": sum(len(trace) for trace in traces),
        "activity_count": len(mapping),
        "activity_mapping": mapping,
        "max_trace_length": max_trace_length,
        "window_sizes": window_manifests,
        "ordering": "ascending first-event timestamp; stable for ties",
        "numeric_features": "raw seconds; normalization fitted on init train only",
        "activity_ids": "zero-based; zero padding identified by full-row mask",
    }
    with (dataset_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"Completed {dataset}: traces={len(traces)}, activities={len(mapping)}")


def discover_sources(log_root: Path, patterns: Iterable[str], sizes: Iterable[str]):
    selected_patterns = list(patterns)
    if len(selected_patterns) == 1 and selected_patterns[0].lower() == "all":
        selected_patterns = sorted(
            (path.name for path in log_root.iterdir() if path.is_dir()),
            key=str.casefold,
        )
    sources = []
    for pattern in selected_patterns:
        matching_folders = [
            path
            for path in log_root.iterdir()
            if path.is_dir() and path.name.casefold() == pattern.casefold()
        ]
        if len(matching_folders) != 1:
            raise FileNotFoundError(
                f"expected exactly one case-insensitive directory for {pattern} "
                f"below {log_root}; found {[path.name for path in matching_folders]}"
            )
        folder = matching_folders[0]
        for size in sizes:
            stem = f"{pattern.lower()}{size.lower()}"
            matches = [
                path
                for path in folder.iterdir()
                if path.is_file()
                and path.suffix.lower() == ".mxml"
                and path.stem.lower() == stem
            ]
            if len(matches) != 1:
                raise FileNotFoundError(f"expected exactly one {stem}.mxml in {folder}")
            sources.append(matches[0])
    return sources


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        help="one or more MXML/CSV event logs; bypasses pattern discovery",
    )
    parser.add_argument("--log-root", type=Path, default=project_root / "logs")
    parser.add_argument(
        "--output-root", type=Path, default=Path(__file__).resolve().parent / "datasets"
    )
    parser.add_argument("--patterns", nargs="+", default=["cm"])
    parser.add_argument("--sizes", nargs="+", default=["10k"])
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[100])
    args = parser.parse_args()
    sources = args.input or discover_sources(args.log_root, args.patterns, args.sizes)
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        prepare_one(source, args.output_root, args.window_sizes)


if __name__ == "__main__":
    main()
