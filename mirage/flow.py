import math
from typing import Callable, Optional

import torch


class FlowMatcher:
    def __init__(
        self,
        eps: float,
        sample_steps: int,
        sampling_method: str = "mode",
        mode_scale: float = 1.0,
        lognorm_mean: float = 1.0,
        lognorm_std: float = 0.6,
    ):
        self.eps = eps
        self.sample_steps = sample_steps
        self.sampling_method = sampling_method
        self.mode_scale = mode_scale
        self.lognorm_mean = lognorm_mean
        self.lognorm_std = lognorm_std

    @staticmethod
    def mode_time(uniform: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        correction = scale * (
            torch.cos((math.pi / 2) * uniform).square() - 1 + uniform
        )
        return 1 - uniform - correction

    def sample_time(
        self,
        batch_size: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.sampling_method == "logit_normal":
            normal = torch.randn(batch_size, device=device, generator=generator)
            sampled = torch.sigmoid(
                self.lognorm_mean + self.lognorm_std * normal
            )
        else:
            uniform = torch.rand(batch_size, device=device, generator=generator)
            if self.sampling_method == "mode":
                sampled = self.mode_time(uniform, self.mode_scale)
            elif self.sampling_method == "uniform":
                sampled = uniform
            elif self.sampling_method == "cosmap":
                sampled = 1 - 1 / (torch.tan((math.pi / 2) * uniform) + 1)
            else:
                raise ValueError(f"unsupported sampling method: {self.sampling_method}")
        return sampled * (1 - self.eps) + self.eps

    @staticmethod
    def interpolate(
        target: torch.Tensor, source: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        while time.ndim < target.ndim:
            time = time.unsqueeze(-1)
        return time * target + (1 - time) * source

    def scale_time(self, time: torch.Tensor) -> torch.Tensor:
        return time * (1 / self.eps - 1)

    @staticmethod
    def sample_source(
        shape,
        device: torch.device,
        dtype: torch.dtype,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=dtype, generator=generator)

    @torch.no_grad()
    def euler(
        self,
        source: torch.Tensor,
        predictor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        value = source.clone()
        step_size = 1 / self.sample_steps
        for index in range(self.sample_steps):
            scalar_time = (
                index / self.sample_steps * (1 - self.eps) + self.eps
            )
            time = torch.full(
                (source.shape[0],),
                scalar_time,
                device=source.device,
                dtype=source.dtype,
            )
            prediction = predictor(value, time)
            value = value + (prediction - source) * step_size
        return value
