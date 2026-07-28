"""PDB input and output for backbone structure reconstruction."""

from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from openfold.np import protein

BACKBONE_ATOMS = (0, 1, 2, 4)


def read_pdb(path: str | Path) -> Data:
    path = Path(path)
    structure = protein.from_pdb_string(path.read_text())
    if len(np.unique(structure.chain_index)) != 1:
        raise ValueError(
            f"{path} contains multiple chains; provide one chain per PDB file"
        )

    atom_mask = torch.from_numpy(structure.atom_mask).bool()
    complete = atom_mask[:, BACKBONE_ATOMS].all(dim=-1)
    if not complete.any():
        raise ValueError(f"{path} has no residues with complete N, CA, C, and O atoms")

    keep = complete.numpy()
    length = int(complete.sum())
    return Data(
        id=path.stem,
        coords=torch.from_numpy(structure.atom_positions[keep]).float(),
        residue_type=torch.from_numpy(structure.aatype[keep]).long(),
        residue_pdb_idx=torch.from_numpy(structure.residue_index[keep]).long(),
        seq_pos=torch.arange(length).unsqueeze(-1),
    )


def write_pdb(path: str | Path, atom_positions, residue_types) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.asarray(atom_positions, dtype=np.float32)
    atom_mask = np.zeros(positions.shape[:2], dtype=np.float32)
    atom_mask[:, BACKBONE_ATOMS] = 1.0
    structure = protein.Protein(
        atom_positions=positions,
        atom_mask=atom_mask,
        aatype=np.asarray(residue_types, dtype=np.int32),
        residue_index=np.arange(1, positions.shape[0] + 1),
        chain_index=np.zeros(positions.shape[0], dtype=np.int32),
        b_factors=np.zeros(positions.shape[:2], dtype=np.float32),
    )
    contents = protein.to_pdb(structure)
    if contents.startswith("PARENT N/A\n"):
        contents = contents.removeprefix("PARENT N/A\n")
    path.write_text(contents)
