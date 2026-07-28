from pathlib import Path

from omegaconf import OmegaConf


def load_model_config(path=None):
    if path is None:
        path = Path(__file__).resolve().parents[1] / "configs" / "model.yaml"
    return OmegaConf.load(path)
