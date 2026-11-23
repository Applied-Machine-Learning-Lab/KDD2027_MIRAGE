"""Checks for the training negative-sampling boundary."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mirage.data import build_datasets


class TestDataSampling(unittest.TestCase):
    def test_forbidden_items_cover_complete_observed_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            frames = {
                "train": pd.DataFrame(
                    {"user": [1], "history": [[1]], "target": [2]}
                ),
                "valid": pd.DataFrame(
                    {"user": [1], "history": [[1, 2]], "target": [3]}
                ),
                "test": pd.DataFrame(
                    {"user": [1], "history": [[1, 2, 3]], "target": [4]}
                ),
            }
            for split, frame in frames.items():
                frame.to_parquet(data_dir / f"{split}.parquet", index=False)

            datasets, _ = build_datasets(data_dir, negative_count=0)
            forbidden = datasets["train"].forbidden_items[0]

            self.assertEqual(forbidden, {0, 1, 2, 3, 4})


if __name__ == "__main__":
    unittest.main()
