from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .flow import FlowMatcher
from .modules import (
    HistoryDecoder,
    LayerNorm,
    TimestepEmbedding,
    TransformerRepresentation,
)


def sampled_softmax_loss(
    queries: torch.Tensor,
    positive_embeddings: torch.Tensor,
    positive_ids: torch.Tensor,
    negative_embeddings: torch.Tensor,
    negative_ids: torch.Tensor,
) -> torch.Tensor:
    positive_logits = (queries * positive_embeddings).sum(dim=-1, keepdim=True)
    negative_logits = queries @ negative_embeddings.t()
    accidental_hits = positive_ids[:, None].eq(negative_ids[None, :])
    negative_logits = negative_logits.masked_fill(accidental_hits, -torch.inf)
    logits = torch.cat((positive_logits, negative_logits), dim=1)
    labels = torch.zeros(queries.shape[0], dtype=torch.long, device=queries.device)
    return F.cross_entropy(logits, labels)


class MIRAGEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.item_embedding = nn.Embedding(
            config.item_count + 1, config.hidden_size
        )
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.embedding_norm = LayerNorm(config.hidden_size)
        self.time_embedding = TimestepEmbedding(config.hidden_size)
        self.transformer = TransformerRepresentation(config)
        self.output_dropout = nn.Dropout(config.dropout)
        self.output_norm = LayerNorm(config.hidden_size)
        self.decoder = HistoryDecoder(config)
        self.flow = FlowMatcher(
            eps=config.eps,
            sample_steps=config.sample_steps,
            sampling_method=config.sampling_method,
            mode_scale=config.mode_scale,
            lognorm_mean=config.lognorm_mean,
            lognorm_std=config.lognorm_std,
        )
        # Graph regularizer
        self.knn_graph = None

    def set_graph(self, knn):
        self.knn_graph = knn

    def embed_history(self, sequence: torch.Tensor):
        embeddings = self.embedding_dropout(self.item_embedding(sequence))
        return self.embedding_norm(embeddings), sequence.gt(0)

    def predict_clean(
        self,
        history: torch.Tensor,
        noisy_target: torch.Tensor,
        scaled_time: torch.Tensor,
        mask: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ):
        time_embedding = self.time_embedding(scaled_time)
        uncertainty = self.config.lambda_uncertainty * (
            1
            + torch.randn(
                history.shape,
                device=history.device,
                dtype=history.dtype,
                generator=generator,
            )
        )
        perturbed_history = history + uncertainty * (
            noisy_target + time_embedding
        ).unsqueeze(1)
        final, intermediate = self.transformer(
            perturbed_history, history, mask.float()
        )
        final = self.output_norm(self.output_dropout(final))
        prediction = final[:, -1]
        reconstruction = self.decoder(intermediate[:, -1])
        return prediction, reconstruction

    def history_multi_hot(self, sequence: torch.Tensor) -> torch.Tensor:
        target = torch.zeros(
            sequence.shape[0],
            self.config.item_count + 1,
            device=sequence.device,
        )
        target.scatter_(1, sequence, 1.0)
        target[:, 0] = 0
        return target

    def balanced_reconstruction_loss(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        positive = target.eq(1)
        zero = target.eq(0)
        zero[:, 0] = False
        zero_count = min(
            int(zero.sum().item()),
            int(positive.sum().item() * self.config.mask_ratio),
        )
        selected = positive.clone()
        if zero_count:
            candidates = zero.nonzero(as_tuple=False)
            indices = torch.randint(
                0, candidates.shape[0], (zero_count,), device=output.device
            )
            coordinates = candidates[indices]
            selected[coordinates[:, 0], coordinates[:, 1]] = True
        return ((output - target).square() * selected).mean()

    def recommendation_loss(
        self,
        representation: torch.Tensor,
        target: torch.Tensor,
        negative_items: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.config.loss_type == "sampled_softmax":
            if negative_items is None or negative_items.numel() == 0:
                raise ValueError("sampled_softmax requires negative items")
            negative_items = torch.unique(negative_items)
            return sampled_softmax_loss(
                representation,
                self.item_embedding(target),
                target,
                self.item_embedding(negative_items),
                negative_items,
            )
        scores = representation @ self.item_embedding.weight.t()
        scores[:, 0] = -torch.inf
        return F.cross_entropy(scores, target)

    def compute_loss(
        self,
        sequence: torch.Tensor,
        target: torch.Tensor,
        negative_items: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        epoch: int = 0,
    ):
        history, mask = self.embed_history(sequence)
        target_embedding = self.item_embedding(target)

        # Source: Gaussian noise or user history anchor
        if self.config.use_history_anchor:
            # Use the last history token's embedding after transformer as source
            embedded = self.embedding_dropout(self.item_embedding(sequence))
            normed = self.embedding_norm(embedded)
            final, _ = self.transformer(normed, normed, mask.float())
            source = self.output_norm(self.output_dropout(final))[:, -1]
        else:
            source = self.flow.sample_source(
                target_embedding.shape, target_embedding.device,
                target_embedding.dtype, generator,
            )
        time = self.flow.sample_time(
            sequence.shape[0], sequence.device, generator
        )
        noisy_target = self.flow.interpolate(target_embedding, source, time)

        # Time-modulated Graph Flow (Stage 2.x)
        graph_loss = torch.tensor(0.0, device=sequence.device)
        if self.config.graph_weight > 0 and self.knn_graph is not None:
            knn = self.knn_graph.to(target.device)
            K = min(self.config.graph_k, knn.shape[1])

            if self.config.graph_anchor_mode == "user":
                B = sequence.shape[0]
                anchors = torch.zeros((B, 3), dtype=torch.long, device=target.device)
                anchors[:, 0] = target
                anchors[:, 1] = sequence[:, -1]
                anchors[:, 2] = sequence[:, -2]
                anchors = anchors.clamp(0, knn.shape[0] - 1)
                nb = knn[anchors, :K].view(B, -1)  # (B, 3*K)
                invalid = (
                    (anchors == 0).unsqueeze(-1).expand(-1, -1, K).reshape(B, -1)
                )
                invalid = invalid | nb.eq(0)
            else:
                nb = knn[target.clamp(0, knn.shape[0] - 1), :K]  # (B, K)
                invalid = nb.eq(0)

            nb_emb = self.item_embedding(nb)
            if getattr(self.config, "detach_items", False):
                nb_emb = nb_emb.detach()

            if getattr(self.config, "endpoint_only_reg", False):
                flow_state = target_embedding
                time_weight = torch.ones_like(time)
            else:
                flow_state = noisy_target
                if self.config.graph_time_mod:
                    time_weight = 4.0 * time * (1.0 - time)
                else:
                    time_weight = torch.ones_like(time)

            if getattr(self.config, "detach_flow_state", False):
                flow_state = flow_state.detach()

            dist_sq = (flow_state.unsqueeze(1) - nb_emb).square().sum(dim=-1)
            dist_sq.masked_fill_(invalid, float('inf'))

            if self.config.graph_distance_mode == "softmin":
                # Temperature-scaled softmin: weighted centroid of K neighbors
                tau = 0.1  # temperature for softmin
                neg_dist = -dist_sq / tau
                neg_dist_safe = torch.where(invalid, torch.full_like(neg_dist, -1e9), neg_dist)
                weights = F.softmax(neg_dist_safe, dim=-1)  # (B, K)
                dist_sq_zeroed = torch.where(invalid, torch.zeros_like(dist_sq), dist_sq)
                agg_dist_sq = (dist_sq_zeroed * weights).sum(dim=-1)
            else:
                # Hard min: anchor to the single closest neighbor
                agg_dist_sq = dist_sq.min(dim=-1)[0]
                agg_dist_sq = torch.where(
                    torch.isinf(agg_dist_sq), torch.zeros_like(agg_dist_sq), agg_dist_sq
                )

            graph_loss = (agg_dist_sq * time_weight).mean()

        prediction, decoded = self.predict_clean(
            history,
            noisy_target,
            self.flow.scale_time(time),
            mask,
            generator,
        )
        flow_loss = F.mse_loss(prediction, target_embedding)
        recommendation_loss = self.recommendation_loss(
            prediction, target, negative_items
        )
        reconstruction_loss = self.balanced_reconstruction_loss(
            decoded, self.history_multi_hot(sequence)
        )
        # Curriculum warmup for graph weight
        gw = self.config.graph_weight
        if self.config.graph_warmup_epochs > 0 and gw > 0:
            gw = gw * min(1.0, epoch / self.config.graph_warmup_epochs)

        total = (
            flow_loss
            + self.config.recommendation_weight * recommendation_loss
            + self.config.reconstruction_weight * reconstruction_loss
            + gw * graph_loss
        )
        return {
            "loss": total,
            "flow": flow_loss,
            "recommendation": recommendation_loss,
            "reconstruction": reconstruction_loss,
            "graph": graph_loss,
        }

    @torch.no_grad()
    def predict_representation(
        self,
        sequence: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        history, mask = self.embed_history(sequence)

        # Consistent source: history anchor or Gaussian noise (matches training)
        if self.config.use_history_anchor:
            embedded = self.embedding_dropout(self.item_embedding(sequence))
            normed = self.embedding_norm(embedded)
            final, _ = self.transformer(normed, normed, mask.float())
            source = self.output_norm(self.output_dropout(final))[:, -1]
        else:
            source = self.flow.sample_source(
                (sequence.shape[0], self.config.hidden_size),
                sequence.device, history.dtype, generator,
            )

        # 1-step bypass: skip ODE solver (A6 proved 30-step = 1-step)
        if self.config.sample_steps == 1:
            dummy_t = torch.full((source.shape[0],), self.flow.eps, device=source.device)
            prediction, _ = self.predict_clean(
                history, source, self.flow.scale_time(dummy_t), mask, generator,
            )
            return prediction

        def predictor(value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            prediction, _ = self.predict_clean(
                history, value, self.flow.scale_time(time), mask, generator,
            )
            return prediction

        return self.flow.euler(source, predictor)

    def rank_scores(self, representation: torch.Tensor) -> torch.Tensor:
        scores = representation @ self.item_embedding.weight.t()
        scores[:, 0] = -torch.inf
        return scores

    @torch.no_grad()
    def score(
        self,
        sequence: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        return self.rank_scores(self.predict_representation(sequence, generator))
