"""Run a CPU/GPU forward-and-backward check of the PBPMCD predictor."""

import torch

from .model import PBPMCDPredictor


def main() -> None:
    torch.manual_seed(3447)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, sequence, classes = 8, 18, 19
    lengths = torch.tensor([1, 2, 3, 5, 8, 10, 15, 18], device=device)
    x = torch.zeros(batch, sequence, 34, device=device)
    for index, length in enumerate(lengths.tolist()):
        x[index, -length:, 0] = torch.randint(0, classes, (length,), device=device).float()
        x[index, -length:, 1:3] = torch.rand(length, 2, device=device)
        x[index, -length:, 3] = 1.0
    labels = torch.randint(0, classes, (batch,), device=device)
    model = PBPMCDPredictor(classes).to(device)
    logits = model(x, lengths)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    if logits.shape != (batch, classes):
        raise AssertionError(f"unexpected output shape {tuple(logits.shape)}")
    print(f"input={tuple(x.shape)}, output={tuple(logits.shape)}")
    print(f"loss={loss.item():.6f}, device={device}")
    print("PBPMCD model smoke test passed.")


if __name__ == "__main__":
    main()
