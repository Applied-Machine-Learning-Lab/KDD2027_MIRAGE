from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DataStats:
    user_count: int
    item_count: int
    train_rows: int
    valid_rows: int
    test_rows: int


def _items(values: Sequence[int]) -> List[int]:
    return [int(value) for value in values]


def _load(data_dir: Path, split: str) -> pd.DataFrame:
    path = Path(data_dir) / f"{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"missing MIRAGE split: {path}")
    frame = pd.read_parquet(path, columns=["user", "history", "target"])
    if frame.empty:
        raise ValueError(f"empty MIRAGE split: {path}")
    return frame


def validate_split_contract(data_dir: Path) -> DataStats:
    train, valid, test = (_load(data_dir, name) for name in ("train", "valid", "test"))
    users = train.user.astype(int).tolist()
    if users != valid.user.astype(int).tolist() or users != test.user.astype(int).tolist():
        raise ValueError("train, validation, and test users must have identical order")
    observed = set()
    for index in range(len(train)):
        train_history = _items(train.iloc[index].history)
        train_target = int(train.iloc[index].target)
        valid_history = _items(valid.iloc[index].history)
        valid_target = int(valid.iloc[index].target)
        test_history = _items(test.iloc[index].history)
        test_target = int(test.iloc[index].target)
        if valid_history != train_history + [train_target]:
            raise ValueError(f"validation prefix contract violated at row {index}")
        if test_history != valid_history + [valid_target]:
            raise ValueError(f"test prefix contract violated at row {index}")
        observed.update(train_history)
        observed.update((train_target, valid_target, test_target))
    if not observed or min(observed) <= 0:
        raise ValueError("real item IDs must be positive; item 0 is padding")
    item_count = max(observed)
    if observed != set(range(1, item_count + 1)):
        raise ValueError("real item IDs must be contiguous from 1")
    return DataStats(len(users), item_count, len(train), len(valid), len(test))


def _pad(history: Sequence[int], max_len: int) -> torch.Tensor:
    recent = _items(history)[-max_len:]
    return torch.tensor([0] * (max_len - len(recent)) + recent, dtype=torch.long)


class MIRAGETrainDataset(Dataset):
    def __init__(
        self,
        path: Path,
        max_len: int,
        item_count: int,
        forbidden_items: Sequence[Set[int]],
        negative_count: int,
    ):
        frame = pd.read_parquet(path, columns=["history", "target"])
        self.sequences = [
            _items(history) + [int(target)]
            for history, target in zip(frame.history, frame.target)
        ]
        self.samples = [
            (sequence_index, target_position)
            for sequence_index, sequence in enumerate(self.sequences)
            for target_position in range(1, len(sequence))
        ]
        self.max_len = max_len
        self.item_count = item_count
        self.forbidden_items = [set(items) | {0} for items in forbidden_items]
        self.negative_count = negative_count
        if len(self.forbidden_items) != len(self.sequences):
            raise ValueError("one forbidden-item set is required per source sequence")
        if negative_count < 0:
            raise ValueError("negative_count must be non-negative")
        for forbidden in self.forbidden_items:
            if self.item_count - len(forbidden - {0}) < self.negative_count:
                raise ValueError("not enough unobserved items for negative sampling")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sequence_index, target_position = self.samples[index]
        sequence = self.sequences[sequence_index]
        return (
            _pad(sequence[:target_position], self.max_len),
            torch.tensor(sequence[target_position], dtype=torch.long),
            self._sample_negatives(sequence_index),
        )

    def _sample_negatives(self, sequence_index: int) -> torch.Tensor:
        if self.negative_count == 0:
            return torch.empty(0, dtype=torch.long)
        forbidden = self.forbidden_items[sequence_index]
        selected = []
        selected_set = set()
        while len(selected) < self.negative_count:
            candidates = torch.randint(
                1, self.item_count + 1, (self.negative_count * 2,)
            )
            for candidate in candidates.tolist():
                if candidate not in forbidden and candidate not in selected_set:
                    selected.append(candidate)
                    selected_set.add(candidate)
                if len(selected) == self.negative_count:
                    break
        return torch.tensor(selected, dtype=torch.long)


class MIRAGEEvaluationDataset(Dataset):
    def __init__(self, path: Path, max_len: int):
        frame = pd.read_parquet(path, columns=["history", "target"])
        self.histories = [_items(history) for history in frame.history]
        self.targets = [int(target) for target in frame.target]
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return _pad(self.histories[index], self.max_len), torch.tensor(
            self.targets[index], dtype=torch.long
        )


def build_datasets(
    data_dir: Path,
    max_len: int = 20,
    stats: DataStats = None,
    negative_count: int = 10,
) -> Tuple[Dict[str, Dataset], DataStats]:
    data_dir = Path(data_dir)
    stats = stats or validate_split_contract(data_dir)
    source_frame = _load(data_dir, "test")
    forbidden_items = [
        set(_items(history)) | {int(target)}
        for history, target in zip(source_frame.history, source_frame.target)
    ]
    return {
        "train": MIRAGETrainDataset(
            data_dir / "train.parquet",
            max_len,
            stats.item_count,
            forbidden_items,
            negative_count,
        ),
        "valid": MIRAGEEvaluationDataset(data_dir / "valid.parquet", max_len),
        "test": MIRAGEEvaluationDataset(data_dir / "test.parquet", max_len),
    }, stats
