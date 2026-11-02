import math

import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean = inputs.mean(dim=-1, keepdim=True)
        variance = (inputs - mean).square().mean(dim=-1, keepdim=True)
        return self.weight * (inputs - mean) / torch.sqrt(variance + self.eps) + self.bias


class SublayerConnection(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor, sublayer) -> torch.Tensor:
        return inputs + self.dropout(sublayer(self.norm(inputs)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(hidden_size, hidden_size * 4)
        self.output = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        nn.init.xavier_normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.input(hidden)
        hidden = 0.5 * hidden * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (hidden + 0.044715 * hidden.pow(3))
            )
        )
        return self.output(self.dropout(hidden))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads: int, hidden_size: int, dropout: float):
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        self.heads = heads
        self.head_size = hidden_size // heads
        self.projections = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size) for _ in range(3)
        )
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_normal_(self.output.weight)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = query.shape[0]
        query, key, value = [
            projection(tensor)
            .view(batch_size, -1, self.heads, self.head_size)
            .transpose(1, 2)
            for projection, tensor in zip(self.projections, (query, key, value))
        ]
        correlation = query @ key.transpose(-2, -1) / math.sqrt(self.head_size)
        if mask is not None:
            correlation = correlation.masked_fill(
                mask[:, None, :, None].eq(0), -1e9
            )
        probabilities = self.dropout(F.softmax(correlation, dim=-1))
        hidden = probabilities @ value
        hidden = hidden.transpose(1, 2).contiguous().view(
            batch_size, -1, self.heads * self.head_size
        )
        return self.output(hidden)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, dropout: float):
        super().__init__()
        self.attention = MultiHeadedAttention(heads, hidden_size, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, dropout)
        self.input_sublayer = SublayerConnection(hidden_size, dropout)
        self.output_sublayer = SublayerConnection(hidden_size, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, hidden: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        del condition
        hidden = self.input_sublayer(
            hidden,
            lambda normalized: self.attention(
                normalized, normalized, normalized, mask
            ),
        )
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class TransformerRepresentation(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.capture_after = config.num_blocks - config.last_blocks
        self.blocks = nn.ModuleList(
            TransformerBlock(
                config.hidden_size, config.attention_heads, config.dropout
            )
            for _ in range(config.num_blocks)
        )

    def forward(
        self, hidden: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor
    ):
        intermediate = None
        for index, block in enumerate(self.blocks, start=1):
            hidden = block(hidden, condition, mask)
            if index == self.capture_after:
                intermediate = hidden
        if intermediate is None:
            raise RuntimeError("Transformer intermediate representation was not captured")
        return hidden, intermediate


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def sinusoidal(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.hidden_size // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / half
        )
        arguments = timesteps[:, None].float() * frequencies[None]
        embedding = torch.cat((arguments.cos(), arguments.sin()), dim=-1)
        if self.hidden_size % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
        return embedding

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.network(self.sinusoidal(timesteps))


class HistoryDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dimensions = (
            config.hidden_size,
            *config.decoder_dims,
            config.item_count + 1,
        )
        layers = []
        for index, (input_size, output_size) in enumerate(
            zip(dimensions[:-1], dimensions[1:])
        ):
            layers.append(nn.Linear(input_size, output_size))
            if index < len(dimensions) - 2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        return self.network(representation)
