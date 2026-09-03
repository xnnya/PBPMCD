# PBPMCD data documentation

PBPMCD accepts MXML files and event-log CSV files. CSV input must contain the
columns `CaseID`, `Activity`, `Type`, and `Timestamp`. Timestamps must be valid
ISO-8601 values. Events whose type is `assign` are ignored.

## PRODRIFT

Download the Business Process Drift collection from 4TU.ResearchData record
12712436. Preserve its pattern folders and MXML file names. The preprocessing
module records a SHA-256 hash for every source file and writes trace, event,
activity, and window counts to a manifest.

Nominal size denotes the number of traces in a complete source log. Window size
denotes the number of traces in each non-overlapping monitoring sub-log. A
trace of length l creates l minus one next-activity prefix-label instances.

## Released gradual logs

`data/gradual_released` contains sixteen five-phase logs. Each CSV uses the
columns CaseID, Activity, Type, and Timestamp. Activity is a zero-based integer
whose source label appears in manifest.json.

The five phases are:

1. traces 0--999: base;
2. traces 1000--1999: linearly increasing modified-process probability;
3. traces 2000--2999: modified;
4. traces 3000--3999: linearly increasing base-process probability;
5. traces 4000--4999: base.

The ground-truth gradual intervals are [1000, 2000] and [3000, 4000]. Sampling
uses replacement because the requested five-phase construction needs more base
traces than one PRODRIFT 5k source contains. Every manifest records the seed,
source hash, phase counts, activity mapping, and the associated case-provenance
file.

Regenerate all released logs from PRODRIFT 5k MXML files:

```bash
python -m pbpmcd.generate_gradual_dataset --log-root logs --patterns all
```

Every dataset directory contains:

- `<pattern>_gradual_5k.csv`: the event log;
- `<pattern>_case_provenance.csv`: the selected source regime and trace index
  for every generated case;
- `manifest.json`: source hash, seed, phase counts, activity mapping, and
  ground-truth intervals.

Prepare any released CSV for PBPMCD with:

```bash
python -m pbpmcd.prepare_data --input data/gradual_released/cm/cm_gradual_5k.csv --window-sizes 100
```
