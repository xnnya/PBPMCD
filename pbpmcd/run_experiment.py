"""Run PBPMCD on one prepared event log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from .config import ExperimentConfig
from .data import DatasetStore
from .detector import DetectionRun, prodrift_ground_truth


def parse_ground_truth(text: Optional[str], log_length: int) -> List[int]:
    if text is None:
        return prodrift_ground_truth(log_length)
    return [int(value) for value in text.replace(",", ";").split(";") if value.strip()]


def run(config: ExperimentConfig, data_root: Path, output_dir: Path, truth=None):
    store = DatasetStore(data_root, config.dataset, config.window_size)
    ground_truth = parse_ground_truth(truth, int(store.manifest["trace_count"]))
    return DetectionRun(config, store, output_dir, ground_truth).execute()


def main() -> None:
    module_dir = Path(__file__).resolve().parent
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=defaults.dataset)
    parser.add_argument("--window-size", type=int, default=defaults.window_size)
    parser.add_argument("--initialization-proportion", type=float, default=defaults.initialization_proportion)
    parser.add_argument("--train-proportion", type=float, default=defaults.train_proportion)
    parser.add_argument("--theta-p", type=float, default=defaults.theta_p)
    parser.add_argument("--theta-v", type=float, default=defaults.theta_v)
    parser.add_argument("--alarm-merge-windows", type=int, default=defaults.alarm_merge_windows)
    parser.add_argument("--alarm-merge-proportion", type=float, default=defaults.alarm_merge_proportion)
    parser.add_argument("--activity-embedding-size", type=int, default=defaults.activity_embedding_size)
    parser.add_argument("--conv-filters", type=int, default=defaults.conv_filters)
    parser.add_argument("--conv-kernel-size", type=int, default=defaults.conv_kernel_size)
    parser.add_argument("--hidden-size", type=int, default=defaults.hidden_size)
    parser.add_argument("--lstm-layers", type=int, default=defaults.lstm_layers)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--adaptation-mode", choices=["reinitialize", "finetune"], default=defaults.adaptation_mode)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--ground-truth", help="semicolon-separated trace positions; omit for PRODRIFT")
    parser.add_argument("--data-root", type=Path, default=module_dir / "datasets")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    config = ExperimentConfig(
        dataset=args.dataset,
        window_size=args.window_size,
        initialization_proportion=args.initialization_proportion,
        train_proportion=args.train_proportion,
        theta_p=args.theta_p,
        theta_v=args.theta_v,
        alarm_merge_windows=args.alarm_merge_windows,
        alarm_merge_proportion=args.alarm_merge_proportion,
        activity_embedding_size=args.activity_embedding_size,
        conv_filters=args.conv_filters,
        conv_kernel_size=args.conv_kernel_size,
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        seed=args.seed,
        adaptation_mode=args.adaptation_mode,
        device=args.device,
    )
    output_dir = args.output_dir or (
        module_dir
        / "outputs"
        / (
            f"{config.dataset}_w{config.window_size}_rho{config.initialization_proportion:g}"
            f"_split{config.train_proportion:g}_tp{config.theta_p:g}_tv{config.theta_v:g}"
            f"_merge{config.alarm_merge_windows}_alpha{config.alarm_merge_proportion:g}"
        )
    )
    result = run(config, args.data_root, output_dir, args.ground_truth)
    summary = {
        "output_dir": str(output_dir),
        "raw_alarm_count": len(result["raw_alarm_window_indices"]),
        "consolidated_alarm_count": len(result["consolidated_alarm_window_indices"]),
        "alarm_clusters_window_indices": result["alarm_clusters_window_indices"],
        "alarm_merge_proportion": result["config"]["alarm_merge_proportion"],
        "effective_alarm_merge_windows": result["alarm_consolidation"]["effective_alarm_merge_windows"],
        "localized_detections": result["localized_detections"],
        "reported_detections": result["reported_detections"],
        "radius_free_f1": result["primary_protocol"]["radius_free"]["f1"],
        "mean_delay": result["primary_protocol"]["radius_free"]["mean_delay"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
