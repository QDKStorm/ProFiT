# ProFiT

Flow-matching protein structure tokenizer with VQ latents.

## Environment

```bash
conda env create -f environment.yaml
conda activate profit
pip install -e . --no-deps
```

## Training data

Put one protein chain in each PDB file. Standard residues with complete `N`, `CA`, `C`, and `O` atoms are used; incomplete residues are skipped.

```text
data/raw/
├── protein_001.pdb
├── protein_002.pdb
└── protein_003.pdb
```

Validate and cache the files:

```bash
profit-prepare --data-dir data/raw --cache-dir data/cache
```

## Training

```bash
profit-train \
  --data-dir data/raw \
  --cache-dir data/cache \
  --output-dir runs/profit \
  --batch-size 8 \
  --max-epochs 100 \
  --devices 1
```

Training resumes automatically from `runs/profit/checkpoints/last.ckpt`.

## Reconstruction

```bash
profit-reconstruct \
  --input examples/example.pdb \
  --checkpoint checkpoints/profit.ckpt \
  --output outputs/example.pdb
```

The command writes the reconstructed PDB and `outputs/example.tokens.pt`.
