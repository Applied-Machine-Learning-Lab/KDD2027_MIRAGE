import math
from typing import Dict, Iterable

import torch


class RankingMetricAccumulator:
    def __init__(self, ks: Iterable[int]):
        self.ks = tuple(sorted(set(int(k) for k in ks)))
        if not self.ks or min(self.ks) <= 0:
            raise ValueError("ks must contain positive integers")
        self.count = 0
        self.hit_sums = {k: 0.0 for k in self.ks}
        self.ndcg_sums = {k: 0.0 for k in self.ks}

    def update(self, scores: torch.Tensor, labels: torch.Tensor) -> None:
        labels = labels.view(-1)
        if scores.shape[0] != labels.shape[0]:
            raise ValueError("scores and labels must have the same batch size")
        top = scores.topk(min(max(self.ks), scores.shape[1]), dim=1).indices
        matches = top.eq(labels[:, None])
        self.count += labels.shape[0]
        for k in self.ks:
            limited = matches[:, :k]
            self.hit_sums[k] += limited.any(dim=1).sum().item()
            positions = torch.arange(
                1, limited.shape[1] + 1, device=scores.device, dtype=torch.float32
            )
            discounts = 1.0 / torch.log2(positions + 1)
            self.ndcg_sums[k] += (limited.float() * discounts).sum().item()

    def compute(self) -> Dict[str, float]:
        if self.count == 0:
            raise RuntimeError("no ranking examples were accumulated")
        result = {}
        for k in self.ks:
            result[f"HR@{k}"] = self.hit_sums[k] / self.count
            result[f"NDCG@{k}"] = self.ndcg_sums[k] / self.count
        return result
