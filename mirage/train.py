import argparse
from dataclasses import asdict
from datetime import datetime
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import (
    DATASET_PATHS,
    DEFAULT_OUTPUT_ROOT,
    PROJECT_ROOT,
    LOSS_TYPES,
    NEGATIVE_SAMPLING_MODES,
    SAMPLING_METHODS,
    ModelConfig,
    TrainConfig,
    parse_int_tuple,
)
from .data import build_datasets, validate_split_contract
from .model import MIRAGEModel
from .trainer import MIRAGETrainer


RELEASE_TAG = "anonymous-artifact"
PAPER_VERSION = "anonymous-submission"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train MIRAGE on processed sequential recommendation parquet datasets"
    )
    parser.add_argument("--dataset", choices=sorted(DATASET_PATHS), default="beauty")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-len", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--last-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embedding-dropout", type=float, default=0.3)
    parser.add_argument("--decoder-dims", default="512,2048")
    parser.add_argument("--eps", type=float, default=0.001)
    parser.add_argument("--lambda-uncertainty", type=float, default=0.001)
    parser.add_argument("--sample-steps", type=int, default=30)
    parser.add_argument("--sampling-method", choices=SAMPLING_METHODS, default="mode")
    parser.add_argument("--mode-scale", type=float, default=1.0)
    parser.add_argument("--lognorm-mean", type=float, default=1.0)
    parser.add_argument("--lognorm-std", type=float, default=0.6)
    parser.add_argument("--mask-ratio", type=float, default=1.0)
    parser.add_argument("--recommendation-weight", type=float, default=0.2)
    parser.add_argument("--reconstruction-weight", type=float, default=0.4)
    parser.add_argument(
        "--loss-type", choices=LOSS_TYPES, default="sampled_softmax"
    )
    parser.add_argument("--negative-count", type=int, default=10)
    parser.add_argument(
        "--negative-sampling",
        choices=NEGATIVE_SAMPLING_MODES,
        default="shared_pool",
    )
    parser.add_argument("--graph-weight", type=float, default=0.0)
    parser.add_argument("--graph-k", type=int, default=20)
    parser.add_argument("--graph-warmup-epochs", type=int, default=0)
    parser.add_argument("--use-history-anchor", action="store_true", default=False)
    parser.add_argument("--graph-source", type=str, default="cooccur")
    parser.add_argument("--graph-anchor-mode", type=str, default="target")
    parser.add_argument("--no-graph-time-mod", action="store_true", default=False)
    parser.add_argument(
        "--graph-distance-mode", type=str, default="hardmin",
        choices=("hardmin", "softmin"),
    )
    parser.add_argument("--detach-flow-state", action="store_true", default=False)
    parser.add_argument("--detach-items", action="store_true", default=False)
    parser.add_argument(
        "--endpoint-only-reg", action="store_true", default=False,
        help="Ablation: regularize clean target embedding at t=1 with constant weight (w/ Endpoint Reg.)",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--decay-step", type=int, default=100)
    parser.add_argument("--decay-gamma", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--metric-ks", default="5,10,20")
    parser.add_argument("--early-stopping-metric", default="NDCG@20")
    parser.add_argument("--gradient-clip", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    return parser


def fix_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _display_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def _logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("mirage")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    for handler in (
        logging.FileHandler(run_dir / "train.log"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _loader(dataset, batch_size, shuffle, args, device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=torch.Generator().manual_seed(args.seed) if shuffle else None,
    )


def run(args: argparse.Namespace):
    fix_random_seed(args.seed)
    device = _resolve_device(args.device)
    data_dir = args.data_dir or DATASET_PATHS[args.dataset]
    stats = validate_split_contract(data_dir)
    datasets, _ = build_datasets(
        data_dir,
        args.max_len,
        stats,
        negative_count=(
            args.negative_count if args.loss_type == "sampled_softmax" else 0
        ),
    )
    metric_ks = parse_int_tuple(args.metric_ks)
    decoder_dims = parse_int_tuple(args.decoder_dims)
    allowed_metrics = {
        f"{name}@{k}" for name in ("HR", "NDCG") for k in metric_ks
    }
    if args.early_stopping_metric not in allowed_metrics:
        raise ValueError("early_stopping_metric must be included in metric_ks")
    model_config = ModelConfig(
        item_count=stats.item_count,
        max_len=args.max_len,
        hidden_size=args.hidden_size,
        num_blocks=args.num_blocks,
        attention_heads=args.attention_heads,
        last_blocks=args.last_blocks,
        dropout=args.dropout,
        embedding_dropout=args.embedding_dropout,
        decoder_dims=decoder_dims,
        eps=args.eps,
        lambda_uncertainty=args.lambda_uncertainty,
        sample_steps=args.sample_steps,
        sampling_method=args.sampling_method,
        mode_scale=args.mode_scale,
        lognorm_mean=args.lognorm_mean,
        lognorm_std=args.lognorm_std,
        mask_ratio=args.mask_ratio,
        recommendation_weight=args.recommendation_weight,
        reconstruction_weight=args.reconstruction_weight,
        loss_type=args.loss_type,
        negative_count=args.negative_count,
        negative_sampling=args.negative_sampling,
        graph_weight=args.graph_weight,
        graph_k=args.graph_k,
        graph_warmup_epochs=args.graph_warmup_epochs,
        use_history_anchor=args.use_history_anchor,
        graph_anchor_mode=args.graph_anchor_mode,
        graph_time_mod=not args.no_graph_time_mod,
        graph_distance_mode=args.graph_distance_mode,
        detach_flow_state=args.detach_flow_state,
        detach_items=args.detach_items,
        endpoint_only_reg=args.endpoint_only_reg,
    )
    train_config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        decay_step=args.decay_step,
        decay_gamma=args.decay_gamma,
        eval_interval=args.eval_interval,
        patience=args.patience,
        metric_ks=metric_ks,
        early_stopping_metric=args.early_stopping_metric,
        gradient_clip=args.gradient_clip,
        log_interval=args.log_interval,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_root) / args.dataset / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = _logger(run_dir)
    payload = {
        "method": "MIRAGE",
        "source": {
            "paper": PAPER_VERSION,
            "release": RELEASE_TAG,
        },
        "dataset": args.dataset,
        "data_dir": _display_path(data_dir),
        "output_dir": _display_path(run_dir),
        "device": str(device),
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "candidate_set": "full_catalog",
        "train_samples": len(datasets["train"]),
        "data_stats": asdict(stats),
        "model": asdict(model_config),
        "training": asdict(train_config),
    }
    (run_dir / "config.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    logger.info(
        "dataset=%s users=%d items=%d train_samples=%d device=%s max_len=%d seed=%d loss_type=%s negative_count=%d",
        args.dataset,
        stats.user_count,
        stats.item_count,
        len(datasets["train"]),
        device,
        args.max_len,
        args.seed,
        args.loss_type,
        args.negative_count,
    )
    loaders = (
        _loader(datasets["train"], args.batch_size, True, args, device),
        _loader(datasets["valid"], args.eval_batch_size, False, args, device),
        _loader(datasets["test"], args.eval_batch_size, False, args, device),
    )
    model = MIRAGEModel(model_config)
    if args.graph_weight > 0:
        knn_name = f"{args.graph_source}_knn_100.pt"
        knn_path = Path(args.output_root) / "_cache" / args.dataset / knn_name
        if not knn_path.is_file():
            logger.info("KNN graph not found, auto-building with build_knn.py ...")
            import subprocess, sys as _sys
            build_script = PROJECT_ROOT / "scripts" / "mirage" / "build_knn.py"
            result = subprocess.run(
                [_sys.executable, str(build_script), "--dataset", args.dataset, "--K", "100"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"KNN build failed:\n{result.stderr}")
            logger.info("KNN build output:\n%s", result.stdout)
        if knn_path.is_file():
            knn = torch.load(knn_path, map_location="cpu", weights_only=True)
            # Validate: check shape and nonzero ratio
            V_expected = stats.item_count + 1
            if knn.shape[0] != V_expected:
                raise ValueError(f"KNN rows {knn.shape[0]} != expected V={V_expected}")
            if knn.shape[1] < args.graph_k:
                raise ValueError(f"KNN has {knn.shape[1]} cols, need at least graph_k={args.graph_k}")
            nonzero = (knn[1:] != 0).float().mean().item()
            if nonzero < 0.5:
                logger.warning("KNN graph is sparse (%.1f%% nonzero), graph may be weak", nonzero * 100)
            model.set_graph(knn)
            logger.info("graph KNN loaded [%s]: %s (shape=%s, nonzero=%.1f%%)",
                        args.graph_source, _display_path(knn_path), knn.shape, nonzero * 100)
        else:
            logger.warning("KNN graph still missing after build attempt, running without graph")
    logger.info("parameters=%d", sum(parameter.numel() for parameter in model.parameters()))
    trainer = MIRAGETrainer(
        model,
        *loaders,
        train_config,
        device,
        run_dir,
        logger,
        args.eval_seed,
        payload,
    )
    return trainer.fit()


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
