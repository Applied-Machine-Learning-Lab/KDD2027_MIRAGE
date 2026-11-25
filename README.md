# MIRAGE

Official implementation of the paper:
**[Escaping the euclidean void: Manifold-informed flow matching for sequential recommendation](https://arxiv.org/abs/2607.23762)**.

MIRAGE is a manifold-informed rectification framework for accelerated generation of embeddings in sequential recommendation. This repository contains the core model implementation, processed splits for the four benchmark datasets (Beauty, Sports, Toys, CDs), prebuilt item-graph KNN caches, and reproduction scripts.

## Directory layout

```text
mirage/                             # Core MIRAGE model, flow matcher, data loader, trainer
scripts/mirage/run_mirage.sh         # Shared reproduction entrypoint
scripts/mirage/run_{dataset}.sh      # Per-dataset reproduction scripts
scripts/mirage/run_all_datasets.sh   # Run Beauty, Sports, Toys, CDs sequentially
scripts/mirage/build_knn.py          # Rebuild co-occurrence KNN graphs if needed
dataset/{beauty,sports,toys,cds}/    # Dataset splits (train, valid, test)
scripts/output/mirage/_cache/*/      # Prebuilt co-occurrence KNN caches
tests/                              # Unit test suite
```

## Environment

Recommended with `uv`:

```bash
cd MIRAGE
uv sync
source ./.venv/bin/activate
```

Alternatively, with standard `pip`:

```bash
pip install torch pandas pyarrow scipy
pip install -e .
```

If you already have a compatible Python environment activated, you can also run this project directly from the repository root because the scripts use the active `python` when `./.venv/bin/python` is not present.

## Reproduction scripts

All scripts use CUDA and seed `2026`. Select the target GPU with `DEVICE=cuda:<id>`.

### Running experiments

```bash
# Single dataset
DEVICE=cuda:0 bash scripts/mirage/run_beauty.sh
DEVICE=cuda:0 bash scripts/mirage/run_sports.sh
DEVICE=cuda:0 bash scripts/mirage/run_toys.sh
DEVICE=cuda:0 bash scripts/mirage/run_cds.sh

# All datasets sequentially
DEVICE=cuda:0 bash scripts/mirage/run_all_datasets.sh
```

Dry run without launching training:

```bash
DRY_RUN=1 bash scripts/mirage/run_all_datasets.sh
```

Useful overrides:

```bash
DEVICE=cuda:3 RUN_NAME=my_beauty_run bash scripts/mirage/run_beauty.sh
EPOCHS=1 BATCH_SIZE=64 EVAL_BATCH_SIZE=64 NUM_WORKERS=0 bash scripts/mirage/run_beauty.sh
```

Outputs and checkpoints are saved to `scripts/output/mirage/<dataset>/<run_name>/`.

## Rebuilding KNN caches

The repository includes prebuilt `cooccur_knn_100.pt` files for all benchmark datasets. To rebuild them from scratch if desired:

```bash
python scripts/mirage/build_knn.py --all --K 100
```

MIRAGE dynamically loads the first `K` columns from `cooccur_knn_100.pt`, so the same cache supports $K=20$ and $K=40$ configurations.

## Direct Python entrypoint

You can directly invoke the training module via Python:

```bash
# Beauty
python -m mirage.train \
  --dataset beauty \
  --device cuda:0 \
  --run-name manual_beauty \
  --sample-steps 1 \
  --graph-weight 0.1 \
  --graph-k 20 \
  --recommendation-weight 0.2
```

For a rapid CPU smoke test:

```bash
python -m mirage.train \
  --dataset beauty --device cpu --run-name smoke_cpu \
  --num-workers 0 --epochs 1 --eval-interval 1 --patience 1 \
  --batch-size 8 --eval-batch-size 8 \
  --max-train-batches 1 --max-eval-batches 1 \
  --sample-steps 1 --graph-weight 0
```

## Running tests

Run the unit test suite to verify gradient flows, loss stability, and model components:

```bash
python -m unittest discover -s tests -v
```

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{fang2026escaping,
  title={Escaping the euclidean void: Manifold-informed flow matching for sequential recommendation},
  author={Fang, Dengzhao and Gao, Jingtong and Li, Yu and Zhao, Xiangyu and Chang, Yi},
  journal={arXiv preprint arXiv:2607.23762},
  year={2026}
}
```
