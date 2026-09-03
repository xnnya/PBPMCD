"""Generate the released five-phase gradual-drift logs from PRODRIFT MXML.

Each output contains 5,000 traces in five 1,000-trace phases:

1. base regime;
2. a linear base-to-modified mixture;
3. modified regime;
4. a linear modified-to-base mixture; and
5. base regime.

PRODRIFT alternates base and modified regimes in ten equal source segments.
This script separates those pools, samples traces deterministically, reassigns
case identifiers, and shifts timestamps so output cases remain chronological.
Sampling with replacement is intentional because a five-phase log requires
more base-regime traces than one 5k source log contains.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

from .prepare_data import Trace, parse_mxml, parse_timestamp


DEFAULT_PATTERNS = (
    "cb", "cd", "cf", "cm", "cp", "fr", "ior", "iro", "oir",
    "ori", "pl", "pm", "rio", "roi", "rp", "sw",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_source(log_root: Path, pattern: str, size: str) -> Path:
    folders = [
        path for path in log_root.iterdir()
        if path.is_dir() and path.name.casefold() == pattern.casefold()
    ]
    if len(folders) != 1:
        raise FileNotFoundError(f"cannot resolve pattern directory: {pattern}")
    stem = f"{pattern}{size}".casefold()
    files = [
        path for path in folders[0].iterdir()
        if path.is_file()
        and path.suffix.casefold() == ".mxml"
        and path.stem.casefold() == stem
    ]
    if len(files) != 1:
        raise FileNotFoundError(f"cannot resolve source {pattern}{size}.mxml")
    return files[0]


def regime_pools(traces: Sequence[Trace]) -> tuple[list[Trace], list[Trace]]:
    if len(traces) % 10:
        raise ValueError(
            f"expected ten equal PRODRIFT segments, found {len(traces)} traces"
        )
    segment_size = len(traces) // 10
    base, modified = [], []
    for index in range(10):
        target = base if index % 2 == 0 else modified
        target.extend(traces[index * segment_size:(index + 1) * segment_size])
    return base, modified


def phase_regime(case_index: int, rng: random.Random) -> str:
    phase, offset = divmod(case_index, 1000)
    if phase in (0, 4):
        return "base"
    if phase == 2:
        return "modified"
    progress = offset / 999.0
    modified_probability = progress if phase == 1 else 1.0 - progress
    return "modified" if rng.random() < modified_probability else "base"


def write_dataset(
    traces: Sequence[Trace],
    source: Path,
    output_dir: Path,
    pattern: str,
    seed: int,
) -> None:
    base, modified = regime_pools(traces)
    activity_names = sorted(
        {event[0] for trace in traces for event in trace},
        key=str.casefold,
    )
    activity_ids = {name: index for index, name in enumerate(activity_names)}
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{pattern}_gradual_5k.csv"
    provenance_path = output_dir / f"{pattern}_case_provenance.csv"
    phase_counts = {
        "base": [0, 0, 0, 0, 0],
        "modified": [0, 0, 0, 0, 0],
    }
    output_start = datetime(2004, 1, 1, tzinfo=timezone.utc)

    with log_path.open("w", encoding="utf-8", newline="") as log_handle, \
         provenance_path.open("w", encoding="utf-8", newline="") as provenance_handle:
        log_writer = csv.writer(log_handle)
        provenance_writer = csv.writer(provenance_handle)
        log_writer.writerow(["CaseID", "Activity", "Type", "Timestamp"])
        provenance_writer.writerow(
            ["CaseID", "Phase", "SourceRegime", "SourceTraceIndex"]
        )
        for case_id in range(5000):
            regime = phase_regime(case_id, rng)
            pool = base if regime == "base" else modified
            source_index = rng.randrange(len(pool))
            trace = pool[source_index]
            phase = case_id // 1000
            phase_counts[regime][phase] += 1
            provenance_writer.writerow([case_id, phase + 1, regime, source_index])

            source_start = parse_timestamp(trace[0][2])
            case_start = output_start + timedelta(hours=case_id)
            for activity, event_type, timestamp_text in trace:
                offset = parse_timestamp(timestamp_text) - source_start
                shifted = case_start + offset
                log_writer.writerow(
                    [
                        case_id,
                        activity_ids[activity],
                        event_type,
                        shifted.isoformat(timespec="milliseconds"),
                    ]
                )

    manifest = {
        "dataset": f"{pattern}_gradual_5k",
        "source_file": source.name,
        "source_sha256": sha256(source),
        "seed": seed,
        "trace_count": 5000,
        "phase_size": 1000,
        "phases": [
            "base",
            "linear_base_to_modified",
            "modified",
            "linear_modified_to_base",
            "base",
        ],
        "ground_truth_intervals": [[1000, 2000], [3000, 4000]],
        "sampling": "deterministic resampling with replacement",
        "phase_regime_counts": phase_counts,
        "activity_mapping": activity_ids,
        "case_provenance_file": provenance_path.name,
        "event_log_file": log_path.name,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def parse_patterns(values: Iterable[str]) -> list[str]:
    patterns = [value.casefold() for value in values]
    return list(DEFAULT_PATTERNS) if patterns == ["all"] else patterns


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=project_root / "logs")
    parser.add_argument("--patterns", nargs="+", default=["all"])
    parser.add_argument("--size", default="5k")
    parser.add_argument("--seed", type=int, default=3447)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "data" / "gradual_released",
    )
    args = parser.parse_args()

    for index, pattern in enumerate(parse_patterns(args.patterns)):
        source = locate_source(args.log_root, pattern, args.size)
        traces = parse_mxml(source)
        output_dir = args.output_root / pattern
        write_dataset(
            traces,
            source,
            output_dir,
            pattern,
            args.seed + index,
        )
        print(f"generated {pattern}: {output_dir}")


if __name__ == "__main__":
    main()
