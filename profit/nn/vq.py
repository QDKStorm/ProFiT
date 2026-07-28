"""
Vector Quantization module with EMA codebook updates.

The interface is aligned with existing quantizers in this repo:
- forward(x) -> (quantized, indices, aux_loss)
- indices_to_codes(indices) -> quantized codes
"""

from __future__ import annotations
from contextlib import nullcontext
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast


def _sample_vectors(samples: torch.Tensor, num: int) -> torch.Tensor:
    n = samples.shape[0]
    if n >= num:
        indices = torch.randperm(n, device=samples.device)[:num]
    else:
        indices = torch.randint(0, n, (num,), device=samples.device)
    return samples[indices]


def _kmeans(
    samples: torch.Tensor,
    num_clusters: int,
    num_iters: int = 10,
    use_cosine_sim: bool = False,
):
    means = _sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            logits = samples @ means.t()
            buckets = logits.argmax(dim=-1)
        else:
            distances = (
                samples.pow(2).sum(dim=1, keepdim=True)
                - 2 * (samples @ means.t())
                + means.pow(2).sum(dim=1, keepdim=False).unsqueeze(0)
            )
            buckets = distances.argmin(dim=-1)

        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = torch.zeros(
            num_clusters, samples.shape[-1], device=samples.device, dtype=samples.dtype
        )
        new_means.scatter_add_(0, buckets.unsqueeze(-1).expand_as(samples), samples)
        new_means = new_means / bins_min_clamped.unsqueeze(-1)

        if use_cosine_sim:
            new_means = F.normalize(new_means, dim=-1)

        means = torch.where(zero_mask.unsqueeze(-1), means, new_means)

    return means, bins


class VQ(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        codebook_size: int,
        decay: float = 0.99,
        eps: float = 1e-5,
        commitment_loss_weight: float = 1.0,
        use_cosine_sim: bool = False,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        force_quantization_f32: bool = True,
        threshold_ema_dead_code: Optional[float] = None,
        expired_code_reset_size: Optional[float] = None,
        sync_kmeans: bool = True,
    ):
        super().__init__()

        if codebook_size <= 0:
            raise ValueError("codebook_size must be > 0")

        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.eps = eps
        self.commitment_loss_weight = commitment_loss_weight
        self.use_cosine_sim = use_cosine_sim
        self.kmeans_iters = kmeans_iters
        self.sync_kmeans = sync_kmeans
        self.force_quantization_f32 = force_quantization_f32
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.expired_code_reset_size = (
            expired_code_reset_size
            if expired_code_reset_size is not None
            else (
                threshold_ema_dead_code * 10
                if threshold_ema_dead_code is not None
                else None
            )
        )

        if kmeans_init:
            embed = torch.zeros(codebook_size, dim)
        else:
            embed = torch.randn(codebook_size, dim)
            if use_cosine_sim:
                embed = F.normalize(embed, dim=-1)

        self.register_buffer("embed", embed)
        self.register_buffer("ema_embed", embed.clone())
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("initted", torch.tensor(not kmeans_init))
        self._dist_synced = False

    @staticmethod
    def _is_distributed():
        return (
            dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        )

    def _sync_state_across_ranks(self):
        if self._dist_synced or not self._is_distributed():
            return

        # Use rank-0 initialization for codebook state so all workers start from identical codes.
        dist.broadcast(self.embed, src=0)
        dist.broadcast(self.ema_embed, src=0)
        dist.broadcast(self.ema_cluster_size, src=0)
        self._dist_synced = True

    def _quantize(self, x_flat: torch.Tensor):
        if self.use_cosine_sim:
            x_n = F.normalize(x_flat, dim=-1)
            e_n = F.normalize(self.embed, dim=-1)
            logits = x_n @ e_n.t()
            indices = logits.argmax(dim=-1)
        else:
            distances = (
                x_flat.pow(2).sum(dim=1, keepdim=True)
                - 2 * (x_flat @ self.embed.t())
                + self.embed.pow(2).sum(dim=1, keepdim=False).unsqueeze(0)
            )
            indices = distances.argmin(dim=-1)

        quantized = F.embedding(indices, self.embed)
        return quantized, indices

    @torch.no_grad()
    def _gather_samples_for_kmeans(self, x_flat: torch.Tensor) -> torch.Tensor:
        if not (self._is_distributed() and self.sync_kmeans):
            return x_flat

        world_size = dist.get_world_size()
        local_size = torch.tensor(
            [x_flat.shape[0]], device=x_flat.device, dtype=torch.long
        )
        all_sizes = [
            torch.zeros(1, device=x_flat.device, dtype=torch.long)
            for _ in range(world_size)
        ]
        dist.all_gather(all_sizes, local_size)

        max_size = max(s.item() for s in all_sizes)
        padded = torch.zeros(
            max_size, self.dim, device=x_flat.device, dtype=x_flat.dtype
        )
        padded[: x_flat.shape[0]] = x_flat

        all_padded = [torch.zeros_like(padded) for _ in range(world_size)]
        dist.all_gather(all_padded, padded)

        return torch.cat([p[: s.item()] for p, s in zip(all_padded, all_sizes)], dim=0)

    @torch.no_grad()
    def _init_embed_(self, x_flat: torch.Tensor):
        if bool(self.initted.item()):
            return

        data = self._gather_samples_for_kmeans(x_flat)
        kmeans_embed, cluster_size = _kmeans(
            data,
            self.codebook_size,
            num_iters=self.kmeans_iters,
            use_cosine_sim=self.use_cosine_sim,
        )

        if self._is_distributed() and self.sync_kmeans:
            dist.broadcast(kmeans_embed, src=0)
            dist.broadcast(cluster_size, src=0)

        embed_sum = kmeans_embed * cluster_size.unsqueeze(1).to(kmeans_embed.dtype)
        self.ema_embed.copy_(embed_sum)
        self.ema_cluster_size.copy_(cluster_size.to(self.ema_cluster_size.dtype))

        denom = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + self.eps) / (
            denom + self.codebook_size * self.eps
        )
        smoothed = smoothed * denom

        new_embed = self.ema_embed / smoothed.unsqueeze(1).clamp(min=self.eps)
        if self.use_cosine_sim:
            new_embed = F.normalize(new_embed, dim=-1)

        self.embed.copy_(new_embed)
        self.initted.copy_(torch.tensor(True, device=self.initted.device))

    @torch.no_grad()
    def _ema_update(self, x_flat: torch.Tensor, indices: torch.Tensor):
        one_hot = F.one_hot(indices, num_classes=self.codebook_size).to(x_flat.dtype)

        cluster_size = one_hot.sum(dim=0)
        embed_sum = one_hot.t() @ x_flat

        if self._is_distributed():
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)
            dist.all_reduce(embed_sum, op=dist.ReduceOp.SUM)

        self.ema_cluster_size.mul_(self.decay).add_(
            cluster_size, alpha=1.0 - self.decay
        )
        self.ema_embed.mul_(self.decay).add_(embed_sum, alpha=1.0 - self.decay)

        denom = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + self.eps) / (
            denom + self.codebook_size * self.eps
        )
        smoothed = smoothed * denom

        new_embed = self.ema_embed / smoothed.unsqueeze(1).clamp(min=self.eps)
        if self.use_cosine_sim:
            new_embed = F.normalize(new_embed, dim=-1)

        self.embed.copy_(new_embed)

        self._expire_codes(x_flat)

    @torch.no_grad()
    def _expire_codes(self, x_flat: torch.Tensor):
        if self.threshold_ema_dead_code is None:
            return

        expired = self.ema_cluster_size < self.threshold_ema_dead_code
        num_expired = expired.sum().item()
        if num_expired == 0:
            return

        # Gather samples from all ranks so every GPU replaces with the same vectors.
        if self._is_distributed():
            world_size = dist.get_world_size()
            # Exchange sizes first — x_flat may have different lengths across ranks.
            local_size = torch.tensor(
                [x_flat.shape[0]], device=x_flat.device, dtype=torch.long
            )
            all_sizes = [
                torch.zeros(1, device=x_flat.device, dtype=torch.long)
                for _ in range(world_size)
            ]
            dist.all_gather(all_sizes, local_size)

            max_size = max(s.item() for s in all_sizes)
            padded = torch.zeros(
                max_size, self.dim, device=x_flat.device, dtype=x_flat.dtype
            )
            padded[: x_flat.shape[0]] = x_flat

            all_padded = [torch.zeros_like(padded) for _ in range(world_size)]
            dist.all_gather(all_padded, padded)

            x_pool = torch.cat(
                [p[: s.item()] for p, s in zip(all_padded, all_sizes)], dim=0
            )
        else:
            x_pool = x_flat

        n = x_pool.shape[0]
        if n >= num_expired:
            indices = torch.randperm(n, device=x_pool.device)[:num_expired]
        else:
            indices = torch.randint(0, n, (num_expired,), device=x_pool.device)

        # Broadcast so all ranks pick the same indices.
        if self._is_distributed():
            dist.broadcast(indices, src=0)

        sampled = x_pool[indices]
        if self.use_cosine_sim:
            sampled = F.normalize(sampled, dim=-1)

        self.embed.data[expired] = sampled
        self.ema_embed.data[expired] = sampled * self.expired_code_reset_size
        self.ema_cluster_size.data[expired] = self.expired_code_reset_size

    def indices_to_codes(self, indices: torch.Tensor):
        return F.embedding(indices.long(), self.embed)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {x.shape[-1]}")

        self._sync_state_across_ranks()

        x_dtype = x.dtype
        q_ctx = (
            autocast("cuda", enabled=False)
            if self.force_quantization_f32 and x.is_cuda
            else nullcontext()
        )

        with q_ctx:
            x_in = x.float() if self.force_quantization_f32 else x
            x_flat = x_in.reshape(-1, self.dim)

            if mask is not None:
                mask_flat = mask.reshape(-1).bool()
                x_valid = x_flat[mask_flat]
            else:
                mask_flat = None
                x_valid = x_flat

            # Init / EMA / expire must only see valid positions so that padded
            # zero vectors do not pollute codebook statistics.
            self._init_embed_(x_valid)

            quantized_flat, indices_flat = self._quantize(x_flat)

            if self.training:
                if mask_flat is not None:
                    self._ema_update(x_valid, indices_flat[mask_flat])
                else:
                    self._ema_update(x_flat, indices_flat)

            quantized = quantized_flat.view_as(x_in)
            if mask is not None:
                # Commitment loss only on valid positions.
                mse = (quantized.detach() - x_in).pow(2).mean(dim=-1)
                denom = mask.float().sum().clamp(min=1.0)
                commit_loss = (
                    (mse * mask.float()).sum() / denom * self.commitment_loss_weight
                )
            else:
                commit_loss = (
                    F.mse_loss(quantized.detach(), x_in) * self.commitment_loss_weight
                )

            # Straight-through estimator.
            quantized = x_in + (quantized - x_in).detach()

        quantized = quantized.to(x_dtype)
        indices = indices_flat.view(*x.shape[:-1])
        return quantized, indices, commit_loss
