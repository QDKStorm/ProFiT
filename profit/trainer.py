"""Lightning base class for flow-matching tokenizer training."""

from functools import partial

import lightning as L
import torch


class FlowMatchingModule(L.LightningModule):
    def __init__(self, cfg_exp):
        super().__init__()
        self.cfg_exp = cfg_exp
        self.inf_cfg = None
        self.fm = None

    def configure_optimizers(self):
        return torch.optim.Adam(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=self.cfg_exp.opt.lr,
        )

    def _nn_out_to_x_clean(self, nn_out, batch):
        prediction = nn_out["coors_pred"]
        if self.cfg_exp.model.target_pred == "v":
            return batch["x_t"] + (1 - batch["t"][..., None, None]) * prediction
        if self.cfg_exp.model.target_pred == "x_1":
            return prediction
        raise ValueError(
            f"Unsupported parameterization: {self.cfg_exp.model.target_pred}"
        )

    def sample_t(self, shape):
        config = self.cfg_exp.loss.t_distribution
        distribution = torch.distributions.beta.Beta(config.p1, config.p2)
        beta_samples = distribution.sample(shape).to(self.device)
        uniform_samples = torch.rand(shape, device=self.device)
        return torch.where(
            torch.rand(shape, device=self.device) < 0.02,
            uniform_samples,
            beta_samples,
        )

    def validation_step_data(self, batch):
        with torch.no_grad():
            return self.training_step(batch, batch_idx=-1)

    def configure_inference(self, config):
        self.inf_cfg = config

    def generate(
        self,
        *,
        nsamples,
        n,
        dt,
        self_cond,
        dtype,
        mask,
        coords_mask,
        single_repr,
        verbose=False,
        **_,
    ):
        predictor = partial(
            self.predict_clean_n_v_w_guidance,
            guidance_weight=1.0,
            autoguidance_ratio=0.0,
        )
        return self.fm.full_simulation(
            predictor,
            dt=dt,
            nsamples=nsamples,
            n=n,
            self_cond=self_cond,
            device=self.device,
            mask=mask,
            coords_mask=coords_mask,
            dtype=dtype,
            single_repr=single_repr,
        )
