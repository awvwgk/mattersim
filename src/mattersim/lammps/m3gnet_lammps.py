"""M3GNet wrapper for LAMMPS with inter-layer exchange.

Wraps a pre-trained M3GNet model and inserts LAMMPS exchange calls between
each MainBlock layer. This keeps ghost atom embeddings in sync with their
real counterparts, giving the correct multi-hop receptive field.

Ghost truncation (à la MACE):
    After each MainBlock's scatter, ghost atom features are discarded
    (they'll be overwritten by exchange anyway). Between layers, atom_attr
    is [nlocal, F]. Before the next conv, LammpsExchange pads zeros for
    ghosts and calls forward_exchange to populate them from local copies.
    This avoids redundant per-node compute on ghost atoms.

Works for both single-GPU (local copy) and multi-GPU (MPI + local copy)
via LAMMPS's data.forward_exchange / data.reverse_exchange.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn

from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
from mattersim.forcefield.m3gnet.modules.message_passing import polynomial


class LammpsExchange(torch.autograd.Function):
    """Pad ghost slots, exchange local→ghost, with matching backward.

    Forward:  [nlocal, F] → pad zeros → [ntotal, F] → forward_exchange
    Backward: [ntotal, F] grad → reverse_exchange → truncate → [nlocal, F]

    Both forward_exchange and reverse_exchange expect full [ntotal, vec_len]
    tensors as both copy_from and copy_to arguments.
    """

    @staticmethod
    def forward(
        ctx,
        atom_attr: torch.Tensor,
        forward_exchange_fn: Callable,
        reverse_exchange_fn: Callable,
        nghost: int,
    ) -> torch.Tensor:
        ctx.reverse_exchange_fn = reverse_exchange_fn
        ctx.nlocal = atom_attr.shape[0]

        # Pad ghost slots with zeros, then exchange local → ghost
        pad = torch.zeros(
            (nghost, atom_attr.shape[1]),
            dtype=atom_attr.dtype,
            device=atom_attr.device,
        )
        full = torch.cat([atom_attr, pad], dim=0)
        out = torch.empty_like(full)
        forward_exchange_fn(full, out, atom_attr.shape[1])
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Accumulate ghost gradients back into local atoms
        grad_input = torch.empty_like(grad_output)
        ctx.reverse_exchange_fn(grad_output, grad_input, grad_output.shape[1])
        # Truncate to local atoms only
        return grad_input[: ctx.nlocal], None, None, None


class M3GnetLammps(nn.Module):
    """M3GNet with inter-layer LAMMPS exchange and ghost truncation.

    Replicates M3GNet.forward() but:
    - Inserts exchange + ghost padding before each MainBlock
    - Truncates ghost atoms after each MainBlock's scatter
    - Only sums local atom energies in the readout

    Between layers, atom_attr is [nlocal, F] — per-node ops (embedding,
    transitions) only run on local atoms. Ghost features are populated
    by exchange right before each convolution needs them for gather.
    """

    def __init__(self, m3gnet: M3Gnet):
        super().__init__()
        self.m3gnet = m3gnet

    def forward(
        self,
        input: Dict[str, torch.Tensor],
        nlocal: int,
        forward_exchange_fn: Callable,
        reverse_exchange_fn: Callable,
    ) -> torch.Tensor:
        m = self.m3gnet
        ntotal = input["atom_attr"].shape[0]
        nghost = ntotal - nlocal

        # --- Unpack input (identical to M3GNet.forward) ---
        pos = input["atom_pos"]
        cell = input["cell"]
        pbc_offsets = input["pbc_offsets"]
        atom_attr = input["atom_attr"]
        edge_index = input["edge_index"].long()
        three_body_indices = input["three_body_indices"].long()
        batch = input["batch"]

        # --- Compute edge geometry (identical to M3GNet.forward) ---
        edge_batch = batch[edge_index[0]]
        edge_vector = pos[edge_index[0]] - (
            pos[edge_index[1]]
            + torch.einsum("bi, bij->bj", pbc_offsets, cell[edge_batch])
        )
        edge_length = torch.linalg.norm(edge_vector, dim=1)
        vij = edge_vector[three_body_indices[:, 0]]
        vik = edge_vector[three_body_indices[:, 1]]
        rij = edge_length[three_body_indices[:, 0]]
        rik = edge_length[three_body_indices[:, 1]]
        cos_jik = torch.sum(vij * vik, dim=1) / (rij * rik)
        cos_jik = torch.clamp(cos_jik, min=-1.0 + 1e-7, max=1.0 - 1e-7)
        triple_edge_length = rik.view(-1)
        edge_length = edge_length.unsqueeze(-1)
        atomic_numbers = atom_attr.squeeze(1).long()

        # --- Featurize (all ntotal atoms for initial embedding) ---
        atom_attr = m.atom_embedding(m.one_hot_atoms(atomic_numbers))
        edge_attr = m.rbf(edge_length.view(-1))
        edge_attr_zero = edge_attr
        edge_attr = m.edge_encoder(edge_attr)
        three_basis = m.sbf(triple_edge_length, torch.acos(cos_jik))

        # --- Main loop: exchange → conv (inlined) → truncate ---
        for i, conv in enumerate(m.graph_conv):
            if i > 0:
                # Between layers: atom_attr is [nlocal, F].
                # Pad + exchange to get [ntotal, F] with ghost copies.
                atom_attr = LammpsExchange.apply(
                    atom_attr,
                    forward_exchange_fn,
                    reverse_exchange_fn,
                    nghost,
                )

            # -- Inlined MainBlock.forward with inner truncation --

            # Three-body interaction (inlined from ThreeDInteraction.forward)
            tb = conv.three_body
            # atom_mlp only needs local atoms — edge_index[0] (receivers)
            # are always < nlocal, so the gather only touches local indices.
            atom_mlp_out = tb.atom_mlp(atom_attr[:nlocal])
            atom_mask = (
                atom_mlp_out[edge_index[0][three_body_indices[:, 1]]]
                * polynomial(edge_length[three_body_indices[:, 0]], tb.threebody_cutoff)
                * polynomial(edge_length[three_body_indices[:, 1]], tb.threebody_cutoff)
            )
            three_basis_masked = three_basis * atom_mask
            tb_output = torch.zeros(
                (edge_attr.shape[0], three_basis_masked.shape[1]),
                device=three_basis_masked.device,
                dtype=three_basis_masked.dtype,
            )
            tb_output.index_add_(0, three_body_indices[:, 0], three_basis_masked)
            edge_attr = edge_attr + tb.edge_gate_mlp(tb_output)

            # Update bond features (per-edge)
            feat = torch.concat(
                [atom_attr[edge_index[0]], atom_attr[edge_index[1]], edge_attr],
                dim=1,
            )
            edge_attr = edge_attr + conv.gated_mlp_edge(feat) * conv.edge_layer_edge(
                edge_attr_zero
            )

            # Update atom features: per-edge MLP, then scatter to receivers
            feat = torch.concat(
                [atom_attr[edge_index[0]], atom_attr[edge_index[1]], edge_attr],
                dim=1,
            )
            atom_attr_prime = conv.gated_mlp_atom(feat) * conv.edge_layer_atom(
                edge_attr_zero
            )
            # Scatter to local atoms only — edge_index[0] (receivers) are
            # always < nlocal, so ghost slots would remain zero anyway.
            output = torch.zeros(
                (nlocal, atom_attr_prime.shape[1]),
                device=atom_attr_prime.device,
                dtype=atom_attr_prime.dtype,
            )
            output.index_add_(0, edge_index[0], atom_attr_prime)
            atom_attr = atom_attr[:nlocal] + output

        # --- Readout (local atoms only) ---
        energies_i = m.final(atom_attr).view(-1)
        energies_i = m.normalizer(energies_i, atomic_numbers[:nlocal])
        energies = energies_i.sum().unsqueeze(0)

        return energies
