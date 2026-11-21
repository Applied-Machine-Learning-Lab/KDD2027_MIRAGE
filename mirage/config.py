from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATHS = {
    "beauty": PROJECT_ROOT
    / "dataset/beauty",
    "sports": PROJECT_ROOT
    / "dataset/sports",
    "toys": PROJECT_ROOT
    / "dataset/toys",
    "cds": PROJECT_ROOT
    / "dataset/cds",
}
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "scripts/output/mirage"
SAMPLING_METHODS = ("mode", "uniform", "logit_normal", "cosmap")
LOSS_TYPES = ("sampled_softmax", "full_softmax")
NEGATIVE_SAMPLING_MODES = ("shared_pool",)


@dataclass(frozen=True)
class ModelConfig:
    item_count: int
    max_len: int = 20
    hidden_size: int = 128
    num_blocks: int = 4
    attention_heads: int = 4
    last_blocks: int = 2
    dropout: float = 0.1
    embedding_dropout: float = 0.3
    decoder_dims: Tuple[int, ...] = (512, 2048)
    eps: float = 0.001
    lambda_uncertainty: float = 0.001
    sample_steps: int = 1
    sampling_method: str = "mode"
    mode_scale: float = 1.0
    lognorm_mean: float = 1.0
    lognorm_std: float = 0.6
    mask_ratio: float = 1.0
    recommendation_weight: float = 0.2
    reconstruction_weight: float = 0.4
    loss_type: str = "sampled_softmax"
    negative_count: int = 10
    negative_sampling: str = "shared_pool"
    graph_weight: float = 0.0
    graph_k: int = 20
    graph_warmup_epochs: int = 0
    use_history_anchor: bool = False
    graph_anchor_mode: str = "target"  # "target" or "user"
    graph_time_mod: bool = True  # parabolic time weight 4t(1-t) when True; constant 1.0 when False
    graph_distance_mode: str = "hardmin"  # "hardmin" or "softmin"
    detach_flow_state: bool = False  # stop-gradient on continuous flow state x_t
    detach_items: bool = False  # stop-gradient on item table embeddings e
    endpoint_only_reg: bool = False  # regularize target embedding directly at t=1 with constant weight

    def __post_init__(self) -> None:
        if self.item_count <= 0 or self.max_len <= 0 or self.hidden_size <= 0:
            raise ValueError("item_count, max_len, and hidden_size must be positive")
        if self.attention_heads <= 0 or self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if self.num_blocks <= 0 or not 0 <= self.last_blocks < self.num_blocks:
            raise ValueError("last_blocks must be in [0, num_blocks)")
        if not 0 <= self.dropout < 1 or not 0 <= self.embedding_dropout < 1:
            raise ValueError("dropout probabilities must be in [0, 1)")
        if not self.decoder_dims or min(self.decoder_dims) <= 0:
            raise ValueError("decoder_dims must contain positive dimensions")
        if not 0 < self.eps < 1 or self.sample_steps <= 0:
            raise ValueError("eps must be in (0, 1) and sample_steps must be positive")
        if self.sampling_method not in SAMPLING_METHODS:
            raise ValueError(f"sampling_method must be one of {SAMPLING_METHODS}")
        if self.lambda_uncertainty < 0 or self.lognorm_std < 0:
            raise ValueError("uncertainty scales must be non-negative")
        if self.mask_ratio < 0:
            raise ValueError("mask_ratio must be non-negative")
        if self.recommendation_weight < 0 or self.reconstruction_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if self.loss_type not in LOSS_TYPES:
            raise ValueError(f"loss_type must be one of {LOSS_TYPES}")
        if self.negative_count <= 0:
            raise ValueError("negative_count must be positive")
        if (
            self.loss_type == "sampled_softmax"
            and self.item_count <= self.negative_count
        ):
            raise ValueError("item_count must be greater than negative_count")
        if self.negative_sampling not in NEGATIVE_SAMPLING_MODES:
            raise ValueError(
                f"negative_sampling must be one of {NEGATIVE_SAMPLING_MODES}"
            )


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 500
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    decay_step: int = 100
    decay_gamma: float = 0.1
    eval_interval: int = 20
    patience: int = 5
    metric_ks: Tuple[int, ...] = (5, 10, 20)
    early_stopping_metric: str = "NDCG@20"
    gradient_clip: float = 0.0
    log_interval: int = 100
    max_train_batches: Optional[int] = None
    max_eval_batches: Optional[int] = None

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer configuration")
        if self.decay_step <= 0 or not 0 < self.decay_gamma <= 1:
            raise ValueError("invalid learning-rate decay configuration")
        if self.eval_interval <= 0 or self.patience <= 0:
            raise ValueError("eval_interval and patience must be positive")
        if not self.metric_ks or min(self.metric_ks) <= 0:
            raise ValueError("metric_ks must be positive")


def parse_int_tuple(value: str) -> Tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed or min(parsed) <= 0:
        raise ValueError("expected comma-separated positive integers")
    return parsed
