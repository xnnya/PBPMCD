import csv
import json
import random
import unittest
from pathlib import Path

from pbpmcd.generate_gradual_dataset import DEFAULT_PATTERNS, phase_regime


class GradualReleaseTests(unittest.TestCase):
    def test_phase_regime_has_fixed_outer_phases(self):
        rng = random.Random(3447)
        self.assertEqual(phase_regime(0, rng), "base")
        self.assertEqual(phase_regime(999, rng), "base")
        self.assertEqual(phase_regime(2000, rng), "modified")
        self.assertEqual(phase_regime(2999, rng), "modified")
        self.assertEqual(phase_regime(4000, rng), "base")
        self.assertEqual(phase_regime(4999, rng), "base")

    def test_released_logs_have_complete_manifests_and_provenance(self):
        release_root = Path(__file__).resolve().parents[2] / "data" / "gradual_released"
        directories = sorted(path.name for path in release_root.iterdir() if path.is_dir())
        self.assertEqual(directories, sorted(DEFAULT_PATTERNS))
        for pattern in DEFAULT_PATTERNS:
            dataset_dir = release_root / pattern
            with (dataset_dir / "manifest.json").open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["trace_count"], 5000)
            self.assertEqual(
                manifest["ground_truth_intervals"],
                [[1000, 2000], [3000, 4000]],
            )
            base_counts = manifest["phase_regime_counts"]["base"]
            modified_counts = manifest["phase_regime_counts"]["modified"]
            self.assertEqual(sum(base_counts) + sum(modified_counts), 5000)
            self.assertEqual(base_counts[0], 1000)
            self.assertEqual(base_counts[2], 0)
            self.assertEqual(base_counts[4], 1000)
            self.assertEqual(modified_counts[0], 0)
            self.assertEqual(modified_counts[2], 1000)
            self.assertEqual(modified_counts[4], 0)
            provenance = dataset_dir / manifest["case_provenance_file"]
            event_log = dataset_dir / manifest["event_log_file"]
            self.assertTrue(provenance.is_file())
            self.assertTrue(event_log.is_file())
            with provenance.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5000)
            self.assertEqual(int(rows[0]["CaseID"]), 0)
            self.assertEqual(int(rows[-1]["CaseID"]), 4999)


if __name__ == "__main__":
    unittest.main()
