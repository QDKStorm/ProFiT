"""Euclidean flow matching for centered protein backbone coordinates."""

import math

import torch

from profit.geometry import extract_ca


class R3NFlowMatcher:
    def __init__(self, zero_com=False, scale_ref=1.0, dim=3):
        self.zero_com = zero_com
        self.scale_ref = scale_ref
        self.dim = dim

    @staticmethod
    def _apply_mask(values, mask=None):
        return values if mask is None else values * mask[..., None]

    @staticmethod
    def _masked_mean(values, mask):
        weights = mask[..., None]
        count = weights.sum(dim=-2, keepdim=True).clamp(min=1)
        return (values * weights).sum(dim=-2, keepdim=True) / count

    def _mask_and_zero_com(self, values, mask=None):
        values = self._apply_mask(values, mask)
        if not self.zero_com:
            return values
        mean = (
            values.mean(dim=-2, keepdim=True)
            if mask is None
            else self._masked_mean(values, mask)
        )
        return self._apply_mask(values - mean, mask)

    def interpolate(self, reference, target, time, mask=None):
        reference = self._mask_and_zero_com(reference, mask)
        target = self._mask_and_zero_com(target, mask)
        time = time[..., None, None]
        return (1 - time) * reference + time * target

    def xt_dot(self, target, current, time, mask=None):
        target = self._mask_and_zero_com(target, mask)
        current = self._mask_and_zero_com(current, mask)
        return (target - current) / (1 - time[..., None, None])

    def sample_reference(self, n, shape=(), dtype=None, device=None, mask=None):
        values = torch.randn(*shape, n, self.dim, dtype=dtype, device=device)
        return self._mask_and_zero_com(values * self.scale_ref, mask)

    def full_simulation(
        self,
        predictor,
        *,
        dt,
        nsamples,
        n,
        self_cond,
        device,
        mask,
        coords_mask,
        dtype,
        single_repr,
    ):
        """Integrate the learned vector field from time 0 to 1 with Euler steps."""
        if coords_mask.shape != (nsamples, n):
            raise ValueError(
                f"Expected coordinate mask {(nsamples, n)}, got {coords_mask.shape}"
            )

        steps = math.ceil(1 / float(dt))
        times = torch.linspace(0, 1, steps + 1, device=device)
        values = self.sample_reference(
            n,
            shape=(nsamples,),
            device=device,
            mask=coords_mask,
            dtype=dtype,
        )
        previous_prediction = None
        for step in range(steps):
            batch = {
                "x_t": values,
                "t": times[step].expand(nsamples),
                "mask": mask,
                "coords_mask": coords_mask,
                "single_repr": single_repr,
            }
            if self_cond and previous_prediction is not None:
                batch["x_sc"] = extract_ca(previous_prediction)
            previous_prediction, velocity = predictor(batch)
            step_size = times[step + 1] - times[step]
            values = self._mask_and_zero_com(values + velocity * step_size, coords_mask)
        return values
