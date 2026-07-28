"""Train ProFiT on a directory of backbone PDB files."""

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from profit.config import load_model_config
from profit.data import PDBDataModule
from profit.model import ProFiT


def main():
    parser = argparse.ArgumentParser(description="Train ProFiT")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="runs/profit")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--devices", type=int, default=1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU")

    torch.set_float32_matmul_precision("medium")
    config = load_model_config()
    config.opt.lr = args.learning_rate
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint = checkpoint_dir / "last.ckpt"

    data = PDBDataModule(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = ProFiT(config)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch-{epoch:04d}-step-{step:08d}",
        save_last=True,
        save_top_k=0,
        every_n_epochs=1,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.devices,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        precision="bf16-mixed",
        callbacks=[checkpoint_callback],
        logger=CSVLogger(save_dir=output_dir, name="logs"),
        log_every_n_steps=1,
        default_root_dir=output_dir,
        gradient_clip_val=1.0,
        deterministic=True,
    )
    trainer.fit(
        model,
        data,
        ckpt_path=str(last_checkpoint) if last_checkpoint.exists() else None,
    )


if __name__ == "__main__":
    main()
