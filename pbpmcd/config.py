"""Configuration defaults for PBPMCD detection runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "cm10k"
    window_size: int = 100
    initialization_proportion: float = 0.05
    train_proportion: float = 0.80

    # Detection and alarm-consolidation thresholds.
    theta_p: float = 0.02
    theta_v: float = 0.50
    # A fixed window gap is optional; proportional merging is the default.
    alarm_merge_windows: int = 0
    # Scale-normalized merging: h_d = ceil(alpha_m * number_of_windows).
    alarm_merge_proportion: float = 0.005

    activity_embedding_size: int = 32
    conv_filters: int = 32
    conv_kernel_size: int = 3
    hidden_size: int = 64
    lstm_layers: int = 1
    dropout: float = 0.30

    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 20
    seed: int = 3447
    adaptation_mode: str = "reinitialize"
    device: str = "auto"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


PARAMETER_PROVENANCE = {
    "initialization_proportion": "first 5% of trace windows",
    "train_proportion": "chronological 80% training / 20% evaluation split",
    "activity_embedding_size": "32-dimensional activity embedding",
    "hidden_size": "64 hidden units per LSTM direction",
    "batch_size": "64 prefix-label instances",
    "learning_rate": "Adam learning rate 0.001",
    "epochs": "20 epochs per training call",
    "seed": "deterministic seed 3447",
    "conv_filters": "32 temporal convolution filters",
    "conv_kernel_size": "temporal kernel size 3",
    "lstm_layers": "one bidirectional LSTM layer",
    "dropout": "dropout probability 0.30",
    "theta_p": "global prediction-accuracy drop threshold",
    "theta_v": "per-class recall-vector distance threshold",
    "alarm_merge_windows": "optional fixed alarm-merging gap",
    "alarm_merge_proportion": "scale-normalized alarm-merging proportion",
    "adaptation_mode": "reinitialize the predictor after a detected drift",
}
