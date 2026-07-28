"""Reconstruct a protein backbone with a trained ProFiT checkpoint."""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from profit.batching import DensePaddingDataLoader
from profit.model import ProFiT
from profit.pdb import read_pdb, write_pdb


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint["hyper_parameters"]["cfg_exp"])
    model = ProFiT(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().to(device)


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct a protein backbone with ProFiT"
    )
    parser.add_argument("--input", required=True, help="Input single-chain PDB file")
    parser.add_argument("--output", required=True, help="Output reconstructed PDB file")
    parser.add_argument("--checkpoint", default="checkpoints/profit.ckpt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available; pass --device cpu for CPU reconstruction"
        )

    torch.manual_seed(0)
    device = torch.device(args.device)
    item = read_pdb(args.input)
    batch = next(iter(DensePaddingDataLoader([item]))).to(device)
    model = load_checkpoint(args.checkpoint, device)
    model.configure_inference(OmegaConf.create({"dt": 0.05}))
    with torch.inference_mode():
        result = model.predict_step(batch, 0)

    output = Path(args.output)
    coordinates = result["pred_coords"][0].cpu().numpy()
    residue_types = item.residue_type.cpu().numpy()
    write_pdb(output, coordinates, residue_types)
    torch.save(result["tokens"][0].cpu(), output.with_suffix(".tokens.pt"))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
