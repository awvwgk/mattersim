"""Build M3GNet graph input dict from LAMMPS MLIAP neighbor list data.

LAMMPS MLIAP provides a flat two-body neighbor list (pair_i, pair_j, rij).
M3GNet additionally requires three-body (angle) indices. This module constructs
the full M3GNet input dict by:
1. Filtering edges to the model cutoff (LAMMPS provides extended neighbor list)
2. Computing three-body indices from edges within the threebody_cutoff
3. Assembling all tensors in the format expected by M3GNet.forward()

The graph has ntotal nodes (nlocal real + nghost ghost atoms). Ghost atom
embeddings are kept in sync via LAMMPS exchange between MainBlock layers
(see m3gnet_lammps.py).

The pbc_offsets trick:
    With pos=0, cell=I, pbc_offsets=rij, M3GNet computes:
        edge_vector = 0 - (0 + rij @ I) = -rij
    which is the correct sign convention. Differentiating w.r.t.
    pbc_offsets yields pair forces.
"""

from __future__ import annotations

import torch

from mattersim.datasets.utils.converter import compute_threebody_indices_torch


def build_m3gnet_input_from_lammps(
    *,
    elems: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    rij: torch.Tensor,
    ntotal: int,
    cutoff: float = 5.0,
    threebody_cutoff: float = 4.0,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Convert LAMMPS MLIAP neighbor list data into an M3GNet input dict.

    The graph has ntotal nodes (local + ghost). Ghost embeddings are synced
    via LAMMPS exchange between MainBlock layers (see M3GnetLammps).

    Args:
        elems: Atomic numbers [ntotal] (1-based, already mapped by caller).
        pair_i: Receiver atom indices [npairs], range [0, ntotal).
        pair_j: Sender atom indices [npairs], range [0, ntotal).
        rij: Distance vectors from i to j [npairs, 3].
        ntotal: Total number of atoms (local + ghost).
        cutoff: Model cutoff in Angstrom (edges beyond this are dropped).
        threebody_cutoff: Cutoff for three-body interactions in Angstrom.
        device: Torch device.

    Returns:
        Dictionary with all keys required by M3GNet.forward(), plus:
        - "_sort_idx": indices that sort the filtered edges by receiver
        - "_edge_mask": boolean mask [npairs_lammps] for which pairs were kept
    """
    distances_all = torch.linalg.norm(rij, dim=1)

    # Filter to model cutoff (LAMMPS provides pairs out to extended cutoff)
    edge_mask = distances_all <= cutoff
    pair_i_f = pair_i[edge_mask]
    pair_j_f = pair_j[edge_mask]
    rij_f = rij[edge_mask]
    distances_f = distances_all[edge_mask]

    atom_attr = elems[:ntotal].unsqueeze(-1).to(torch.float32)

    # Edge index: [2, n_edges] with [receiver, sender], all in [0, ntotal)
    edge_index = torch.stack([pair_i_f, pair_j_f], dim=0).to(torch.int64)

    # Three-body computation requires edges sorted by receiver (central atom)
    sort_idx = torch.argsort(edge_index[0])
    edge_index = edge_index[:, sort_idx]
    distances_sorted = distances_f[sort_idx]
    rij_sorted = rij_f[sort_idx]

    # pbc_offsets trick: store rij as pbc_offsets with cell=I, pos=0
    pbc_offsets = rij_sorted.to(torch.float32)

    atom_pos = torch.zeros((ntotal, 3), dtype=torch.float32, device=device)
    cell = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0)

    # Single graph — all ntotal atoms (local + ghost)
    num_atoms = torch.tensor([ntotal], dtype=torch.int64, device=device)
    num_bonds = torch.tensor([edge_index.shape[1]], dtype=torch.int64, device=device)
    num_graphs = torch.tensor(1, dtype=torch.int64, device=device)
    batch = torch.zeros(ntotal, dtype=torch.int64, device=device)

    # Three-body indices
    (
        triple_bond_indices,
        n_triple_ij,
        _n_triple_i,
        n_triple_s,
    ) = compute_threebody_indices_torch(
        edge_indices=edge_index,
        distances=distances_sorted,
        num_atoms=num_atoms,
        threebody_cutoff=threebody_cutoff,
    )

    return {
        "atom_pos": atom_pos,
        "cell": cell,
        "pbc_offsets": pbc_offsets,
        "atom_attr": atom_attr,
        "edge_index": edge_index,
        "three_body_indices": triple_bond_indices,
        "num_three_body": n_triple_s,
        "num_bonds": num_bonds,
        "num_triple_ij": n_triple_ij.unsqueeze(-1),
        "num_atoms": num_atoms,
        "num_graphs": num_graphs,
        "batch": batch,
        "_sort_idx": sort_idx,
        "_edge_mask": edge_mask,
    }
