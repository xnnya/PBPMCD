# PBPMCD

PBPMCD detects concept drift in business-process event logs by monitoring a
next-activity predictor. The implementation combines a temporal convolution,
a bidirectional LSTM, and self-attention. A drift alarm is issued when either
global prediction accuracy or the per-activity recall vector changes beyond
its configured threshold.

The repository contains the PBPMCD implementation, event-log preprocessing,
causal alarm evaluation, released gradual-drift data, tests, and an executable
English notebook.

## Repository layout

```text
pbpmcd/
  config.py                    Default configuration
  prepare_data.py              MXML/CSV preprocessing
  model.py                     CNN-BiLSTM-attention predictor
  detector.py                  Monitoring and adaptation workflow
  metrics.py                   Prediction and causal alarm metrics
  run_experiment.py            Command-line detector entry point
  generate_gradual_dataset.py  Gradual-drift data generator
  smoke_test.py                Model execution check
  tests/                       Unit tests
data/gradual_released/          Released gradual-drift event logs
docs/DATA.md                    Data format and generation details
notebooks/3.drift_detection.ipynb
```

Generated datasets, checkpoints, and outputs are ignored by Git.

## Installation

Python 3.9 or later is required. From the repository root:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Install PBPMCD and its dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

PyTorch builds depend on the operating system and CUDA version. If the command
above cannot install a suitable build, install PyTorch using the selector at
<https://pytorch.org/get-started/locally/> and run `python -m pip install -e .`
again.

## Verify the model

Run a forward-and-backward pass on the available CPU or GPU:

```bash
python -m pbpmcd.smoke_test
```

Expected final line:

```text
PBPMCD model smoke test passed.
```

## Run PBPMCD with an included event log

The released CSV logs already use the required columns: `CaseID`, `Activity`,
`Type`, and `Timestamp`. Prepare one log as non-overlapping trace windows:

```bash
python -m pbpmcd.prepare_data \
  --input data/gradual_released/cm/cm_gradual_5k.csv \
  --window-sizes 100
```

On PowerShell, the same command can be entered on one line:

```powershell
python -m pbpmcd.prepare_data --input data/gradual_released/cm/cm_gradual_5k.csv --window-sizes 100
```

Run a short end-to-end check with one training epoch:

```bash
python -m pbpmcd.run_experiment \
  --dataset cm_gradual_5k \
  --epochs 1 \
  --ground-truth "1000;3000"
```

Use `--epochs 20` for the standard training configuration. The two reference
positions above are the starts of the gradual intervals documented in the
dataset manifest.

## Run PBPMCD with PRODRIFT MXML logs

Place downloaded files under pattern folders, for example:

```text
logs/
  cm/cm10k.mxml
  cb/cb10k.mxml
```

Prepare a log:

```bash
python -m pbpmcd.prepare_data \
  --log-root logs \
  --patterns cm \
  --sizes 10k \
  --window-sizes 100
```

Run the detector:

```bash
python -m pbpmcd.run_experiment --dataset cm10k
```

For PRODRIFT logs, omitting `--ground-truth` uses drift positions at every
one-tenth of the trace sequence. For another dataset, supply semicolon-separated
trace positions explicitly.

## Configuration

The command-line defaults are:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--window-size` | 100 | Traces per monitoring window |
| `--initialization-proportion` | 0.05 | Initial windows used to initialize the predictor |
| `--train-proportion` | 0.80 | Chronological case-level training share |
| `--theta-p` | 0.02 | Global accuracy-drop threshold |
| `--theta-v` | 0.50 | Per-class recall-vector distance threshold |
| `--alarm-merge-proportion` | 0.005 | Scale-normalized alarm-merging gap |
| `--epochs` | 20 | Epochs per training call |
| `--seed` | 3447 | Random seed |
| `--device` | `auto` | Select CUDA when available, otherwise CPU |

All incoming windows are scored before model adaptation. Cases are split
chronologically, prefixes from one case remain in one partition, and numeric
normalization is fitted only on initialization-training cases.

## Outputs

Each run writes to `pbpmcd/outputs/` unless `--output-dir` is supplied:

- `result.json`: configuration, runtime information, alarms, and metrics;
- `window_metrics.csv`: window-level accuracy changes and alarm decisions;
- `detections.csv`: consolidated localized and reported alarm positions;
- `raw_detections.csv`: raw alarm positions and cluster identifiers;
- `checkpoints/`: predictor state immediately before each detected drift.

The reported alarm position is the end of the window required to make the
decision. Radius-free evaluation matches the first causal alarm after each
reference drift and reports precision, recall, F1, and mean delay.

## Notebook and tests

Start the notebook interface:

```bash
python -m pip install -e ".[notebook]"
jupyter notebook notebooks/3.drift_detection.ipynb
```

Run the automated checks:

```bash
python -m unittest discover -s pbpmcd/tests -v
```

The notebook calls the same package modules used by the command line.

## Gradual-drift data

`data/gradual_released/` contains 16 deterministic, five-phase event logs.
Each dataset directory includes the event log, case-level provenance, and a
manifest with source hashes, generation seed, phase counts, and ground-truth
intervals. See [docs/DATA.md](docs/DATA.md) for the complete schema and the
regeneration command.
