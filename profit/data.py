"""PDB dataset preparation and loading."""

import argparse
import hashlib
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import Dataset

from profit.batching import DensePaddingDataLoader
from profit.pdb import read_pdb


def discover_pdbs(data_dir):
    paths = sorted(Path(data_dir).rglob("*.pdb"))
    if not paths:
        raise FileNotFoundError(f"No .pdb files found under {data_dir}")
    return paths


class PDBDataset(Dataset):
    def __init__(self, paths, cache_dir):
        self.paths = list(paths)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.paths)

    def _cache_path(self, path):
        digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
        return self.cache_dir / f"{path.stem}-{digest}.pt"

    def __getitem__(self, index):
        path = self.paths[index]
        cache_path = self._cache_path(path)
        if not cache_path.exists():
            torch.save(read_pdb(path), cache_path)
        return torch.load(cache_path, map_location="cpu", weights_only=False)


class PDBDataModule(L.LightningDataModule):
    def __init__(self, data_dir, cache_dir, batch_size, num_workers, seed=42):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        del stage
        paths = discover_pdbs(self.data_dir)
        generator = torch.Generator().manual_seed(self.seed)
        order = torch.randperm(len(paths), generator=generator).tolist()
        paths = [paths[index] for index in order]
        validation_size = max(1, round(len(paths) * 0.01))
        if len(paths) == 1:
            train_paths = validation_paths = paths
        else:
            validation_size = min(validation_size, len(paths) - 1)
            validation_paths = paths[:validation_size]
            train_paths = paths[validation_size:]
        self.train_dataset = PDBDataset(train_paths, self.cache_dir)
        self.validation_dataset = PDBDataset(validation_paths, self.cache_dir)

    def _loader(self, dataset, shuffle):
        return DensePaddingDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.validation_dataset, shuffle=False)


def main():
    parser = argparse.ArgumentParser(
        description="Validate and cache training PDB files"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", default="data/cache")
    args = parser.parse_args()
    paths = discover_pdbs(args.data_dir)
    dataset = PDBDataset(paths, args.cache_dir)
    for index in range(len(dataset)):
        dataset[index]
    print(f"Prepared {len(dataset)} structures in {args.cache_dir}")


if __name__ == "__main__":
    main()
