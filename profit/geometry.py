"""Small geometry helpers shared by training and reconstruction."""

from einops import rearrange


def extract_ca(coords):
    if coords.shape[-2] % 4 != 0 or coords.shape[-1] != 3:
        raise ValueError(
            f"Expected backbone coordinates shaped [..., 4N, 3], got {coords.shape}"
        )
    return rearrange(coords, "b (n atoms) xyz -> b n atoms xyz", atoms=4)[..., 1, :]
