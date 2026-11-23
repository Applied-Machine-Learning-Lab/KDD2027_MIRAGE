#!/usr/bin/env python3
"""Build co-occurrence KNN graph for MIRAGE graph regularizer.

Usage:
  python scripts/mirage/build_knn.py --dataset beauty [--K 100] [--window 5]
  python scripts/mirage/build_knn.py --all  # build for all 4 datasets

The KNN graph is saved to scripts/output/mirage/_cache/<dataset>/cooccur_knn_<K>.pt
"""
import argparse, sys, torch, numpy as np, pandas as pd
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mirage.data import validate_split_contract

DATASET_PATHS = {
    "beauty": "dataset/beauty",
    "sports": "dataset/sports",
    "toys": "dataset/toys",
    "cds": "dataset/cds",
}
DEFAULT_WINDOWS = {
    "cds": 5,
    "beauty": 5,
    "sports": 5,
    "toys": 5,
}
OUTPUT_BASE = Path("scripts/output/mirage/_cache")


def build_knn(dataset: str, K: int = 100, window: int = None):
    window = window or DEFAULT_WINDOWS.get(dataset, 5)
    data_dir = Path(DATASET_PATHS[dataset])
    stats = validate_split_contract(data_dir)
    V = stats.item_count + 1  # 0 = padding
    print(f"[{dataset}] item_count={stats.item_count}, V={V}, K={K}, window={window} (per Appendix B.3)")

    train_df = pd.read_parquet(data_dir / "train.parquet", columns=["history", "target"])
    cooccur = Counter()

    for _, row in train_df.iterrows():
        items = [int(x) for x in row["history"]] + [int(row["target"])]
        items = [i for i in items if 0 < i < V]
        for i in range(len(items)):
            for j in range(max(0, i - window), min(len(items), i + window + 1)):
                if i != j:
                    cooccur[(items[i], items[j])] += 1

    knn = torch.zeros(V, K, dtype=torch.long)
    for i in range(1, V):
        # Only include items with actual co-occurrence — prevent fake neighbors
        scores = [(j, cooccur[(i, j)]) for j in range(1, V) if (i, j) in cooccur]
        scores.sort(key=lambda x: -x[1])
        for rank, (j, _) in enumerate(scores[:K]):
            knn[i, rank] = j
        # Remaining slots stay at item 0, matching the paper checkpoints.

    out_dir = OUTPUT_BASE / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"cooccur_knn_{K}.pt"
    torch.save(knn, path)

    # Validation
    nonzero = (knn[1:] != 0).float().mean()
    unique_ratio = len(set(knn[1:].flatten().tolist())) / (V * K)
    print(f"  saved: {path}  shape={knn.shape}  nonzero={nonzero:.1%}  unique_ratio={unique_ratio:.1%}")
    if nonzero < 0.5:
        print(f"  ⚠️  WARNING: >50% KNN entries are zero (padding). Graph may be too sparse.")
    if V < 100:
        print(f"  ⚠️  WARNING: V={V} is very small.")
    return path


def main():
    parser = argparse.ArgumentParser(description="Build co-occurrence KNN for MIRAGE")
    parser.add_argument("--dataset", choices=list(DATASET_PATHS), default="beauty")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--window", type=int, default=None, help="Context window (default: 5)")
    args = parser.parse_args()

    datasets = list(DATASET_PATHS) if args.all else [args.dataset]
    for ds in datasets:
        build_knn(ds, K=args.K, window=args.window)


if __name__ == "__main__":
    main()
