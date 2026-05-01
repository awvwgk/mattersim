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

from mattersim.datasets.utils.threebody_indices_torch import compute_threebody_torch


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
    ) = _compute_threebody_with_cutoff(
        edge_indices=edge_index.T,  # [n_edges, 2]
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


def _compute_threebody_with_cutoff(
    edge_indices: torch.Tensor,
    distances: torch.Tensor,
    num_atoms: torch.Tensor,
    threebody_cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute three-body indices with distance-based filtering.

    Mirrors the logic in converter.py:compute_threebody_indices_torch but
    operates on a single structure.

    Args:
        edge_indices: [n_edges, 2] sorted by first column (central atom).
        distances: [n_edges] edge lengths.
        num_atoms: [1] number of atoms (single structure).
        threebody_cutoff: Distance cutoff for three-body interactions.

    Returns:
        triple_bond_indices, n_triple_ij, n_triple_i, n_triple_s
    """
    num_edges = edge_indices.shape[0]
    device = edge_indices.device

    if num_edges > 0 and threebody_cutoff is not None:
        valid_mask = distances <= threebody_cutoff
        ij_reverse_map = torch.where(valid_mask)[0]
        original_index = torch.arange(num_edges, device=device)[valid_mask]
        valid_edge_indices = edge_indices[valid_mask]
    else:
        ij_reverse_map = None
        original_index = torch.arange(num_edges, device=device)
        valid_edge_indices = edge_indices

    if num_edges > 0 and valid_edge_indices.shape[0] > 0:
        (
            angle_indices,
            num_angles_per_edge,
            _num_edges_per_atom,
            num_angles_per_structure,
        ) = compute_threebody_torch(valid_edge_indices, num_atoms)

        if ij_reverse_map is not None:
            num_angles_per_edge_full = torch.zeros(
                num_edges, dtype=torch.long, device=device
            )
            num_angles_per_edge_full[ij_reverse_map] = num_angles_per_edge
            num_angles_per_edge = num_angles_per_edge_full

        # Map filtered indices back to original edge indices
        angle_indices = original_index[angle_indices]
    else:
        angle_indices = torch.zeros((0, 2), dtype=torch.long, device=device)
        num_angles_per_edge = torch.zeros(num_edges, dtype=torch.int32, device=device)
        _num_edges_per_atom = torch.zeros(
            int(num_atoms.sum().item()), dtype=torch.int32, device=device
        )
        num_angles_per_structure = torch.zeros(
            num_atoms.shape[0], dtype=torch.int32, device=device
        )

    return (
        angle_indices,
        num_angles_per_edge,
        _num_edges_per_atom,
        num_angles_per_structure,
    )
