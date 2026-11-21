from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch

from .metrics import RankingMetricAccumulator


class MIRAGETrainer:
    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        test_loader,
        config,
        device: torch.device,
        run_dir: Path,
        logger: Optional[logging.Logger] = None,
        eval_seed: int = 2026,
        resolved_config: Optional[Dict] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("mirage")
        self.eval_seed = eval_seed
        self.resolved_config = resolved_config or {}

    def _checkpoint(self, epoch, optimizer, scheduler, validation):
        return {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "validation": validation,
            "model_config": asdict(self.model.config),
            "train_config": asdict(self.config),
            "resolved_config": self.resolved_config,
        }

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        metrics = RankingMetricAccumulator(self.config.metric_ks)
        generator = torch.Generator(device=self.device).manual_seed(self.eval_seed)
        for batch_index, (sequence, target) in enumerate(loader):
            if (
                self.config.max_eval_batches is not None
                and batch_index >= self.config.max_eval_batches
            ):
                break
            sequence = sequence.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            scores = self.model.score(sequence, generator)
            metrics.update(scores, target)
        return metrics.compute()

    def _train_epoch(self, optimizer, epoch: int):
        self.model.train()
        totals = {}
        sample_count = 0
        for batch_index, batch in enumerate(self.train_loader):
            if (
                self.config.max_train_batches is not None
                and batch_index >= self.config.max_train_batches
            ):
                break
            sequence, target = batch[:2]
            negatives = batch[2] if len(batch) > 2 else None
            sequence = sequence.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            negative_pool = (
                torch.unique(negatives.to(self.device, non_blocking=True))
                if negatives is not None and negatives.numel()
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            losses = self.model.compute_loss(sequence, target, negative_pool, epoch=epoch)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("non-finite MIRAGE training loss")
            losses["loss"].backward()
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
            optimizer.step()
            batch_size = sequence.shape[0]
            sample_count += batch_size
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + value.detach().item() * batch_size
            if batch_index == 0 or (batch_index + 1) % self.config.log_interval == 0:
                self.logger.info(
                    "epoch=%d batch=%d loss=%.6f flow=%.6f recommendation=%.6f reconstruction=%.6f",
                    epoch,
                    batch_index + 1,
                    losses["loss"].item(),
                    losses["flow"].item(),
                    losses["recommendation"].item(),
                    losses["reconstruction"].item(),
                )
        if sample_count == 0:
            raise RuntimeError("training loader produced no samples")
        return {name: value / sample_count for name, value in totals.items()}

    def fit(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.config.decay_step,
            gamma=self.config.decay_gamma,
        )
        best_value = float("-inf")
        best_epoch = 0
        stale = 0
        history = []
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self._train_epoch(optimizer, epoch)
            scheduler.step()
            should_evaluate = (
                epoch % self.config.eval_interval == 0
                or epoch == self.config.epochs
            )
            validation = self.evaluate(self.valid_loader) if should_evaluate else {}
            record = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation,
            }
            history.append(record)
            if should_evaluate:
                value = validation[self.config.early_stopping_metric]
                if value > best_value:
                    best_value = value
                    best_epoch = epoch
                    stale = 0
                    torch.save(
                        self._checkpoint(epoch, optimizer, scheduler, validation),
                        self.run_dir / "best_model.pt",
                    )
                else:
                    stale += 1
            torch.save(
                self._checkpoint(epoch, optimizer, scheduler, validation),
                self.run_dir / "last_model.pt",
            )
            self.logger.info(
                "epoch=%d train=%s validation=%s", epoch, train_metrics, validation
            )
            if should_evaluate and stale >= self.config.patience:
                break
        checkpoint = torch.load(
            self.run_dir / "best_model.pt",
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = self.evaluate(self.test_loader)
        result = {
            "best_epoch": best_epoch,
            "best_metric_name": self.config.early_stopping_metric,
            "best_metric_value": best_value,
            "history": history,
            "test": test_metrics,
        }
        (self.run_dir / "metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        self.logger.info("test=%s", test_metrics)
        return result
