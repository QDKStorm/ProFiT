"""Input features used by the ProFiT encoder and decoder."""

from typing import Literal

import torch

from profit.embeddings import get_time_embedding
from profit.geometry import extract_ca


def _one_hot_bins(values, boundaries):
    indices = torch.bucketize(values, boundaries)
    return torch.nn.functional.one_hot(indices, len(boundaries) + 1).float()


def _pairwise_distance_features(coords, minimum, maximum, dimension):
    distances = torch.norm(coords[:, :, None] - coords[:, None, :], dim=-1)
    boundaries = torch.linspace(minimum, maximum, dimension - 1, device=coords.device)
    return _one_hot_bins(distances, boundaries)


class _Feature(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    def get_dim(self):
        return self.dimension


class _ZeroFeature(_Feature):
    def __init__(self, dimension, mode):
        super().__init__(dimension)
        self.mode = mode

    def forward(self, batch):
        batch_size, length = batch["mask"].shape
        shape = (batch_size, length, self.dimension)
        if self.mode == "pair":
            shape = (batch_size, length, length, self.dimension)
        return torch.zeros(shape, device=batch["mask"].device)


class _TimeSequenceFeature(_Feature):
    def __init__(self, t_emb_dim, **_):
        super().__init__(t_emb_dim)

    def forward(self, batch):
        embedding = get_time_embedding(batch["t"], edim=self.dimension)
        return embedding[:, None].expand(-1, batch["mask"].shape[1], -1)


class _TimePairFeature(_Feature):
    def __init__(self, t_emb_dim, **_):
        super().__init__(t_emb_dim)

    def forward(self, batch):
        length = batch["mask"].shape[1]
        embedding = get_time_embedding(batch["t"], edim=self.dimension)
        return embedding[:, None, None].expand(-1, length, length, -1)


class _ChainBreakFeature(_Feature):
    def __init__(self, **_):
        super().__init__(1)

    def forward(self, batch):
        values = batch.get("chain_breaks_per_residue")
        if values is None:
            values = torch.zeros_like(batch["mask"], dtype=torch.float32)
        return values.float().unsqueeze(-1)


class _SelfConditioningFeature(_Feature):
    def __init__(self, **_):
        super().__init__(3)

    def forward(self, batch):
        values = batch.get("x_sc")
        if values is None:
            batch_size, length = batch["mask"].shape
            values = torch.zeros(batch_size, length, 3, device=batch["mask"].device)
        return values


class _SequenceSeparationFeature(_Feature):
    def __init__(self, seq_sep_dim, **_):
        if seq_sep_dim % 2 != 1:
            raise ValueError("seq_sep_dim must be odd")
        super().__init__(seq_sep_dim)

    def forward(self, batch):
        indices = batch.get("residue_pdb_idx")
        if indices is None:
            length = batch["mask"].shape[1]
            indices = torch.arange(1, length + 1, device=batch["mask"].device).expand(
                batch["mask"].shape[0], -1
            )
        separation = indices[:, :, None] - indices[:, None, :]
        boundaries = torch.linspace(
            -(self.dimension / 2 - 1),
            self.dimension / 2 - 1,
            self.dimension - 1,
            device=indices.device,
        )
        return _one_hot_bins(separation, boundaries)


class _CoordinateDistanceFeature(_Feature):
    def __init__(self, source, dimension, minimum, maximum):
        super().__init__(dimension)
        self.source = source
        self.minimum = minimum
        self.maximum = maximum

    def forward(self, batch):
        coords = batch.get(self.source)
        if coords is None:
            batch_size, length = batch["mask"].shape
            return torch.zeros(
                batch_size,
                length,
                length,
                self.dimension,
                device=batch["mask"].device,
            )
        if self.source != "x_sc":
            coords = extract_ca(coords)
        return _pairwise_distance_features(
            coords, self.minimum, self.maximum, self.dimension
        )


class FeatureFactory(torch.nn.Module):
    """Build and project the fixed set of model input features."""

    def __init__(
        self,
        feats,
        dim_feats_out,
        use_ln_out,
        mode: Literal["seq", "pair"],
        **kwargs,
    ):
        super().__init__()
        self.mode = mode
        self.ret_zero = not feats
        if self.ret_zero:
            self.zero_creator = _ZeroFeature(dim_feats_out, mode)
            return

        self.feat_creators = torch.nn.ModuleList(
            [self._create(name, kwargs) for name in feats]
        )
        self.ln_out = (
            torch.nn.LayerNorm(dim_feats_out) if use_ln_out else torch.nn.Identity()
        )
        self.linear_out = torch.nn.Linear(
            sum(feature.get_dim() for feature in self.feat_creators),
            dim_feats_out,
            bias=False,
        )

    def _create(self, name, config):
        if self.mode == "seq":
            creators = {
                "time_emb": _TimeSequenceFeature,
                "chain_break_per_res": _ChainBreakFeature,
                "x_sc": _SelfConditioningFeature,
            }
            if name not in creators:
                raise ValueError(f"Unsupported sequence feature: {name}")
            return creators[name](**config)

        if name == "rel_seq_sep":
            return _SequenceSeparationFeature(**config)
        if name == "time_emb":
            return _TimePairFeature(**config)
        distance_config = {
            "x1_pair_dists": ("x_1", "x1_pair_dist"),
            "xt_pair_dists": ("x_t", "xt_pair_dist"),
            "x_sc_pair_dists": ("x_sc", "x_sc_pair_dist"),
        }
        if name not in distance_config:
            raise ValueError(f"Unsupported pair feature: {name}")
        source, prefix = distance_config[name]
        return _CoordinateDistanceFeature(
            source=source,
            dimension=config[f"{prefix}_dim"],
            minimum=config[f"{prefix}_min"],
            maximum=config[f"{prefix}_max"],
        )

    def _mask(self, values, mask):
        if self.mode == "seq":
            return values * mask[..., None]
        pair_mask = mask[:, :, None] * mask[:, None, :]
        return values * pair_mask[..., None]

    def forward(self, batch):
        if self.ret_zero:
            return self.zero_creator(batch)
        values = torch.cat([creator(batch) for creator in self.feat_creators], dim=-1)
        values = self._mask(values, batch["mask"])
        return self._mask(self.ln_out(self.linear_out(values)), batch["mask"])
