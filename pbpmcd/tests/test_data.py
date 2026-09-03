import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from pbpmcd.data import FeatureNormalizer, Sublog, temporal_case_split
from pbpmcd.prepare_data import discover_sources, parse_csv


def sample_sublog():
    data = np.zeros((8, 3, 35), dtype=np.float32)
    labels = np.arange(8, dtype=np.int64) % 2
    lengths = np.ones(8, dtype=np.int64)
    for index in range(8):
        data[index, -1, 0] = index // 2
        data[index, -1, 1] = labels[index]
        data[index, -1, 2] = 10.0 * (index + 1)
        data[index, -1, 3] = 2.0 * (index + 1)
        data[index, -1, 4] = 1.0
    return Sublog(data, labels, lengths)


class DataTests(unittest.TestCase):
    def test_temporal_split_is_by_case(self):
        train, evaluation = temporal_case_split(sample_sublog(), 0.5)
        self.assertEqual(set(train.case_ids), {0, 1})
        self.assertEqual(set(evaluation.case_ids), {2, 3})
        self.assertFalse(set(train.case_ids) & set(evaluation.case_ids))

    def test_normalizer_fits_training_only(self):
        train, evaluation = temporal_case_split(sample_sublog(), 0.5)
        normalizer = FeatureNormalizer.fit(train)
        self.assertEqual(normalizer.elapsed_scale, 40.0)
        transformed = normalizer.transform(evaluation)
        self.assertLessEqual(float(transformed.data[:, :, 2].max()), 1.0)

    def test_source_discovery_is_case_insensitive_on_linux(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_folder = root / "IOR"
            source_folder.mkdir()
            source = source_folder / "IOR2.5k.MXML"
            source.touch()
            self.assertEqual(discover_sources(root, ["all"], ["2.5k"]), [source])
            self.assertEqual(discover_sources(root, ["ior"], ["2.5k"]), [source])

    def test_csv_parser_groups_cases_and_sorts_chronologically(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "sample.csv"
            source.write_text(
                "CaseID,Activity,Type,Timestamp\n"
                "later,A,complete,2024-01-02T00:00:00+00:00\n"
                "later,B,complete,2024-01-02T00:01:00+00:00\n"
                "earlier,A,complete,2024-01-01T00:00:00+00:00\n"
                "earlier,C,complete,2024-01-01T00:01:00+00:00\n",
                encoding="utf-8",
            )
            traces = parse_csv(source)
            self.assertEqual([event[0] for event in traces[0]], ["A", "C"])
            self.assertEqual([event[0] for event in traces[1]], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
