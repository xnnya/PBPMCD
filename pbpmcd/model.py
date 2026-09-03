"""CNN-BiLSTM-self-attention predictor used by PBPMCD."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PBPMCDPredictor(nn.Module):
    """Predict the next activity from a padded process-trace prefix.

    Input feature 0 is the integer activity identifier. Remaining features
    contain elapsed time, inter-event time, hour-of-day, and weekday values.
    """

    def __init__(
        self,
        num_classes: int,
        activity_embedding_size: int = 32,
        numeric_feature_size: int = 33,
        conv_filters: int = 32,
        conv_kernel_size: int = 3,
        hidden_size: int = 64,
        lstm_layers: int = 1,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        if conv_kernel_size <= 0 or conv_kernel_size % 2 == 0:
            raise ValueError("conv_kernel_size must be a positive odd integer")
        if lstm_layers <= 0:
            raise ValueError("lstm_layers must be positive")

        self.num_classes = int(num_classes)
        self.numeric_feature_size = int(numeric_feature_size)
        self.activity_embedding = nn.Embedding(
            self.num_classes, int(activity_embedding_size)
        )
        encoded_size = int(activity_embedding_size) + self.numeric_feature_size
        self.convolution = nn.Conv1d(
            encoded_size,
            int(conv_filters),
            kernel_size=int(conv_kernel_size),
            padding=int(conv_kernel_size) // 2,
        )
        self.activation = nn.ReLU()
        self.feature_dropout = nn.Dropout(float(dropout))
        self.bilstm = nn.LSTM(
            input_size=int(conv_filters),
            hidden_size=int(hidden_size),
            num_layers=int(lstm_layers),
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout) if int(lstm_layers) > 1 else 0.0,
        )
        representation_size = 2 * int(hidden_size)
        self.attention_score = nn.Linear(representation_size, 1)
        self.output_dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(representation_size, self.num_classes)

    @staticmethod
    def _left_to_right_padded(
        x: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Move valid rows from left padding to packed-sequence right padding."""
        batch_size, sequence_length, feature_size = x.shape
        positions = torch.arange(sequence_length, device=x.device).unsqueeze(0)
        valid_mask = positions < lengths.unsqueeze(1)
        source = sequence_length - lengths.unsqueeze(1) + positions
        source = source.clamp(0, sequence_length - 1)
        gathered = x.gather(
            1, source.unsqueeze(-1).expand(batch_size, sequence_length, feature_size)
        )
        gathered = gathered * valid_mask.unsqueeze(-1).to(gathered.dtype)
        return gathered, valid_mask

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [batch, sequence, features], got {x.shape}")
        expected_features = 1 + self.numeric_feature_size
        if x.size(-1) != expected_features:
            raise ValueError(
                f"expected {expected_features} features after CaseID removal, "
                f"got {x.size(-1)}"
            )
        if lengths is None:
            mask = x.abs().sum(dim=-1).ne(0)
            lengths = mask.sum(dim=1).clamp(min=1).long()
        else:
            lengths = lengths.long().clamp(min=1, max=x.size(1))
        x, mask = self._left_to_right_padded(x, lengths)

        activity = x[..., 0].long().clamp(0, self.num_classes - 1)
        numeric = x[..., 1:].float()
        encoded = torch.cat((self.activity_embedding(activity), numeric), dim=-1)
        encoded = encoded * mask.unsqueeze(-1).to(encoded.dtype)

        convolved = self.convolution(encoded.transpose(1, 2)).transpose(1, 2)
        convolved = self.feature_dropout(self.activation(convolved))
        convolved = convolved * mask.unsqueeze(-1).to(convolved.dtype)

        packed = pack_padded_sequence(
            convolved,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.bilstm(packed)
        sequence, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=x.size(1)
        )

        scores = self.attention_score(sequence).squeeze(-1)
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return self.classifier(self.output_dropout(context))
