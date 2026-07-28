# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from math import prod
from typing import Dict
import random

import torch
from scipy.spatial.transform import Rotation
from einops import rearrange

from profit.flow_matching import R3NFlowMatcher
from profit.nn.transformer import ProteinTransformerAF3
from profit.nn.vq import VQ
from profit.trainer import FlowMatchingModule
from profit.coordinates import ang_to_nm, trans_nm_to_atom37
from profit.geometry import extract_ca


def sample_uniform_rotation(shape=tuple(), dtype=None, device=None):
    """
    Samples rotations distributed uniformly.

    Args:
        shape: tuple (if empty then samples single rotation)
        dtype: used for samples
        device: torch.device

    Returns:
        Uniformly samples rotation matrices [*shape, 3, 3]
    """
    return torch.tensor(
        Rotation.random(prod(shape)).as_matrix(),
        device=device,
        dtype=dtype,
    ).reshape(*shape, 3, 3)


class ProFiT(FlowMatchingModule):
    def __init__(self, cfg_exp):
        super().__init__(cfg_exp=cfg_exp)
        self.save_hyperparameters({"cfg_exp": cfg_exp})
        self.fm = R3NFlowMatcher(zero_com=True, scale_ref=1.0, dim=3)
        self.noise_tau = cfg_exp.training.noise_tau
        self.token_noise_ratio = cfg_exp.training.token_noise_ratio

        self.encoder = ProteinTransformerAF3(
            **cfg_exp.model.ae.encoder,
            ca_only=False,
        )
        self.decoder = ProteinTransformerAF3(
            **cfg_exp.model.ae.decoder,
            ca_only=False,
            latent_add_place="cond",
        )

        self.quantizer = self._build_quantizer(cfg_exp)

    def _build_quantizer(self, cfg_exp):
        dim_latent = cfg_exp.model.ae.encoder.dim_latent
        vq_cfg = cfg_exp.model.ae.vq
        self.codebook_size = vq_cfg.codebook_size
        return VQ(
            dim=dim_latent,
            codebook_size=vq_cfg.codebook_size,
            decay=vq_cfg.decay,
            eps=vq_cfg.eps,
            commitment_loss_weight=vq_cfg.commitment_loss_weight,
            use_cosine_sim=vq_cfg.use_cosine_sim,
            kmeans_init=vq_cfg.kmeans_init,
            kmeans_iters=vq_cfg.kmeans_iters,
            force_quantization_f32=vq_cfg.force_quantization_f32,
            threshold_ema_dead_code=vq_cfg.threshold_ema_dead_code,
            expired_code_reset_size=vq_cfg.expired_code_reset_size,
            sync_kmeans=vq_cfg.sync_kmeans,
        )

    def predict_clean(self, batch: Dict):
        output = self.decoder(batch)
        return self._nn_out_to_x_clean(output, batch), output

    def predict_clean_n_v_w_guidance(
        self,
        batch: Dict,
        guidance_weight: float = 1.0,
        autoguidance_ratio: float = 0.0,
    ):
        del guidance_weight, autoguidance_ratio
        output = self.decoder(batch)
        prediction = self._nn_out_to_x_clean(output, batch)
        velocity = self.fm.xt_dot(
            prediction, batch["x_t"], batch["t"], batch["coords_mask"]
        )
        return prediction, velocity

    def extract_clean_sample(self, batch):
        """Extract backbone coordinates and apply global rotation augmentation."""
        x_1, mask, coords_mask = self._extract_backbone_coordinates(batch)
        x_1, coords_mask = self.apply_random_rotation(x_1, coords_mask)
        mask = rearrange(coords_mask, "b (n atoms) -> b n atoms", atoms=4)[..., 1]
        return (
            ang_to_nm(x_1),
            mask,
            coords_mask,
            x_1.shape[:-2],
            x_1.shape[-2],
            x_1.dtype,
        )

    def _extract_backbone_coordinates(self, batch):
        """Extract backbone coordinates (N, CA, C, O) and masks."""
        backbone_indices = [0, 1, 2, 4]
        x_1 = rearrange(
            batch["coords"][:, :, backbone_indices],
            "b n atoms xyz -> b (n atoms) xyz",
        )
        coords_mask = batch["mask_dict"]["coords"][..., backbone_indices, 0]
        mask = coords_mask[..., 1]
        coords_mask = rearrange(coords_mask, "b n atoms -> b (n atoms)")
        return x_1, mask, coords_mask

    def apply_random_rotation(self, x, mask):
        """Rotate and center each backbone in a batch."""
        rots = sample_uniform_rotation(
            shape=x.shape[:-2], dtype=x.dtype, device=x.device
        )
        return self.fm._mask_and_zero_com(torch.matmul(x, rots), mask), mask

    def training_step(self, batch, batch_idx):
        """Compute one flow-matching training or validation step."""
        val_step = batch_idx == -1
        log_prefix = "validation_loss" if val_step else "train"
        self._prepare_batch_data(batch)
        single_repr = self._encode_and_prepare_flow_matching(batch)
        self._add_noise_to_single_repr(single_repr, batch)
        self._apply_self_conditioning(batch)
        x_1_pred, _ = self.predict_clean(batch)
        train_loss = self._compute_all_losses(batch, x_1_pred, log_prefix)
        self._log_training_metrics(train_loss, batch, log_prefix, val_step)
        return train_loss

    def _prepare_batch_data(self, batch):
        """Prepare batch data by extracting and processing coordinates."""
        x_1, mask, coords_mask, batch_shape, n, dtype = self.extract_clean_sample(batch)
        x_1 = self.fm._mask_and_zero_com(x_1, coords_mask)
        batch.update(
            {
                "x_1": x_1,
                "mask": mask,
                "coords_mask": coords_mask,
                "batch_shape": batch_shape,
                "n": n,
                "dtype": dtype,
            }
        )

    def quantize_single_repr(self, single_repr, mask=None):
        """Quantize valid latent tokens with the VQ codebook."""
        single_repr, indices, vq_loss = self.quantizer(single_repr, mask=mask)
        self._quantizer_aux_loss = vq_loss
        return single_repr, indices

    def _apply_token_noise(self, single_repr, quantizer_indices):
        """Zero hidden vectors at a random subset of token positions.

        Noise ratio is sampled from U(0, token_noise_ratio) per batch.
        Only active during training when token_noise_ratio > 0 and quantizer indices exist.
        """
        if self.token_noise_ratio <= 0 or not self.training:
            return single_repr

        noise_ratio = (
            torch.rand(1, device=quantizer_indices.device).item()
            * self.token_noise_ratio
        )
        noise_mask = torch.rand_like(quantizer_indices.float()) < noise_ratio
        while noise_mask.ndim < single_repr.ndim:
            noise_mask = noise_mask.unsqueeze(-1)
        return single_repr.masked_fill(noise_mask, 0.0)

    def _encode_and_prepare_flow_matching(self, batch):
        """Encode input and prepare flow matching interpolation."""
        single_repr = self.encoder(batch)["single_repr"]
        single_repr, quantizer_indices = self.quantize_single_repr(
            single_repr, mask=batch.get("mask")
        )
        self._last_quantizer_indices = quantizer_indices
        single_repr = self._apply_token_noise(single_repr, quantizer_indices)
        t = self.sample_t(batch["batch_shape"])
        x_0 = self.fm.sample_reference(
            n=batch["n"],
            shape=batch["batch_shape"],
            device=self.device,
            dtype=batch["dtype"],
            mask=batch["coords_mask"],
        )
        x_t = self.fm.interpolate(x_0, batch["x_1"], t)
        batch.update({"t": t, "x_t": x_t})
        return single_repr

    def _add_noise_to_single_repr(self, single_repr, batch):
        """Add per-sample Gaussian noise to latent vectors."""
        shape = (single_repr.size(0),) + (1,) * (single_repr.ndim - 1)
        sigma = self.noise_tau * torch.rand(shape, device=single_repr.device)
        batch["single_repr"] = single_repr + torch.randn_like(single_repr) * sigma

    def _apply_self_conditioning(self, batch):
        """Apply self-conditioning if enabled."""
        if random.random() > 0.5:
            x_pred_sc, _ = self.predict_clean(batch)
            batch["x_sc"] = extract_ca(x_pred_sc.detach())

    def _compute_all_losses(self, batch, x_1_pred, log_prefix):
        """Compute all loss components."""
        x_1, mask, coords_mask = batch["x_1"], batch["mask"], batch["coords_mask"]
        x_t, t = batch["x_t"], batch["t"]
        fm_loss = self.compute_fm_loss(
            x_1, x_1_pred, x_t, t, mask, coords_mask, log_prefix=log_prefix
        )
        train_loss = torch.mean(fm_loss)
        if getattr(self, "_quantizer_aux_loss", None) is not None:
            aux_loss = self._quantizer_aux_loss
            self._log_metric(
                f"{log_prefix}/vq_commitment_loss", aux_loss, mask.shape[0]
            )
            train_loss = train_loss + aux_loss

        return train_loss

    def _log_codebook_utilization(self, log_prefix, batch_size):
        """Log codebook utilization metrics."""
        indices = getattr(self, "_last_quantizer_indices", None)
        if indices is None or self.codebook_size is None:
            return

        unique_codes = indices.unique().numel()
        utilization = unique_codes / self.codebook_size
        self._log_metric(
            f"{log_prefix}/codebook_usage", float(unique_codes), batch_size
        )
        self._log_metric(f"{log_prefix}/codebook_utilization", utilization, batch_size)

    def _log_training_metrics(self, train_loss, batch, log_prefix, val_step):
        """Log loss and codebook utilization."""
        mask = batch["mask"]

        self.log(
            f"{log_prefix}/loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=mask.shape[0],
            sync_dist=True,
            add_dataloader_idx=False,
        )

        self._log_codebook_utilization(log_prefix, mask.shape[0])

        if not val_step:
            self.log(
                "train_loss",
                train_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=mask.shape[0],
                sync_dist=True,
                add_dataloader_idx=False,
            )

    def validation_step(self, batch, batch_idx):
        del batch_idx
        return self.validation_step_data(batch)

    def compute_fm_loss(
        self,
        x_1,
        x_1_pred,
        x_t,
        t,
        mask,
        coords_mask,
        log_prefix: str,
    ):
        """
        Computes and logs flow matching loss.

        Args:
            x_1: True clean sample, shape [*, n, 3].
            x_1_pred: Predicted clean sample, shape [*, n, 3].
            x_t: Sample at interpolation time t (used as input to predict clean sample), shape [*, n, 3].
            t: Interpolation time, shape [*].
            mask: Boolean residue mask, shape [*, nres].

        Returns:
            Flow matching loss.
        """
        natoms = torch.sum(coords_mask, dim=-1) * 3  # [*]

        err = (x_1 - x_1_pred) * coords_mask[..., None]  # [*, n, 3]
        loss = torch.sum(err**2, dim=(-1, -2)) / natoms  # [*]

        total_loss_w = 1.0 / ((1.0 - t) ** 2 + 1e-5)

        loss = loss * total_loss_w  # [*]
        if log_prefix:
            self._log_metric(
                f"{log_prefix}/trans_loss",
                torch.mean(loss),
                mask.shape[0],
                prog_bar=True,
            )
        return loss

    def _log_metric(
        self,
        name,
        value,
        batch_size,
        prog_bar=False,
        on_step=True,
        on_epoch=True,
    ):
        """Helper method for consistent metric logging."""
        self.log(
            name,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def detach_gradients(self, x):
        """Detaches gradients from sample x"""
        return x.detach()

    def samples_to_atom37(self, samples):
        """
        Transforms samples to atom37 representation.

        Args:
            samples: Tensor of shape [b, n, 3]

        Returns:
            Samples in atom37 representation, shape [b, n, 37, 3].
        """
        return trans_nm_to_atom37(samples, ca_only=False)

    def predict_step(self, batch, batch_idx):
        """Encode a backbone, quantize it, and reconstruct its coordinates."""
        del batch_idx
        dt = self.inf_cfg.get("dt", 0.0025)
        x_1, mask, coords_mask, _, n, _ = self.extract_clean_sample(batch)
        x_1 = self.fm._mask_and_zero_com(x_1, coords_mask)
        batch.update({"x_1": x_1, "mask": mask, "coords_mask": coords_mask})
        single_repr = self.encoder(batch)["single_repr"]
        single_repr, indices = self.quantize_single_repr(single_repr, mask=mask)
        x = self.generate(
            nsamples=x_1.shape[0],
            n=n,
            dt=torch.scalar_tensor(dt, dtype=single_repr.dtype),
            self_cond=True,
            dtype=single_repr.dtype,
            mask=mask,
            coords_mask=coords_mask,
            single_repr=single_repr,
        )
        return {
            "id": batch.get("id", None),
            "pred_coords": self.samples_to_atom37(x),
            "tokens": indices,
            "gt_coords": self.samples_to_atom37(x_1),
        }
