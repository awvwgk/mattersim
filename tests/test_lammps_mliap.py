"""Tests for the LAMMPS MLIAP wrapper.

 Verifies that the pbc_offsets trick and the M3GnetLammps wrapper
 reproduce the same energy and forces as vanilla M3Gnet.
 """

import numpy as np
import pytest
import torch
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.potential import Potential, batch_to_dict
from mattersim.lammps.graph_builder import build_m3gnet_input_from_lammps
from mattersim.lammps.m3gnet_lammps import M3GnetLammps


@pytest.fixture(scope="module")
def potential():
    return Potential.from_checkpoint(version="mattersim-v1.0.0-5M", device="cpu")


@pytest.fixture(scope="module")
def nacl_perturbed():
    """NaCl with perturbed positions — has nonzero forces."""
    struct = Structure.from_dict(
        {
            "@module": "pymatgen.core.structure",
            "@class": "Structure",
            "charge": 0.0,
            "lattice": {
                "matrix": [[0.0, 2.82, 2.82], [2.82, 0.0, 2.82], [2.82, 2.82, 0.0]],
                "pbc": (True, True, True),
            },
            "sites": [
                {
                    "species": [{"element": "Na", "occu": 1}],
                    "abc": [0.001, 0.018, 0.047],
                    "label": "Na",
                },
                {
                    "species": [{"element": "Cl", "occu": 1}],
                    "abc": [-0.515, 0.474, 0.505],
                    "label": "Cl",
                },
            ],
        }
    )
    return AseAtomsAdaptor.get_atoms(struct)


def _get_reference(potential, atoms, device="cpu"):
    """Standard Potential.forward() → energy + forces."""
    dataloader = build_dataloader(
        [atoms], batch_size=1, only_inference=True, model_type="m3gnet", shuffle=False
    )
    input_dict = batch_to_dict(
        next(iter(dataloader)), model_type="m3gnet", device=device
    )

    with torch.enable_grad():
        output = potential.forward(
            input_dict, include_forces=True, include_stresses=False
        )

    return output["total_energy"].detach(), output["forces"].detach(), input_dict


def _run_pbc_offsets_trick(model, input_dict, atoms, device="cpu"):
    """Re-run M3GNet with pos=0, cell=I, pbc_offsets=rij (simulating LAMMPS)."""
    # Extract what LAMMPS would give us: edge vectors from the existing graph
    pos = input_dict["atom_pos"]
    cell = input_dict["cell"]
    pbc_offsets = input_dict["pbc_offsets"]
    edge_index = input_dict["edge_index"]
    num_atoms = input_dict["num_atoms"]

    atoms_batch = torch.arange(len(num_atoms), device=device).repeat_interleave(
        num_atoms
    )
    edge_batch = atoms_batch[edge_index[0]]

    # This is what M3GNet computes internally (line 92 of m3gnet.py)
    edge_vector = pos[edge_index[0]] - (
        pos[edge_index[1]]
        + torch.einsum("bi, bij->bj", pbc_offsets.float(), cell[edge_batch])
    )
    # LAMMPS rij = -edge_vector (i→j convention)
    rij = -edge_vector

    ntotal = int(num_atoms.sum().item())
    elems = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int64, device=device)

    lammps_input = build_m3gnet_input_from_lammps(
        elems=elems,
        pair_i=edge_index[0].to(device),
        pair_j=edge_index[1].to(device),
        rij=rij.to(device),
        ntotal=ntotal,
        cutoff=model.model_args["cutoff"],
        threebody_cutoff=model.model_args["threebody_cutoff"],
        device=torch.device(device),
    )
    sort_idx = lammps_input.pop("_sort_idx")
    lammps_input.pop("_edge_mask")  # all edges within cutoff
    lammps_input["pbc_offsets"].requires_grad_(True)

    energy = model.forward(lammps_input)

    (pair_forces_sorted,) = torch.autograd.grad(
        outputs=energy,
        inputs=lammps_input["pbc_offsets"],
        grad_outputs=torch.ones_like(energy),
    )

    # Unsort
    inverse_sort = torch.argsort(sort_idx)
    pair_forces = pair_forces_sorted[inverse_sort]

    # Aggregate pair forces → atomic forces (what LAMMPS does)
    pair_i = edge_index[0]
    pair_j = edge_index[1]
    atomic_forces = torch.zeros((ntotal, 3), device=device)
    atomic_forces.scatter_add_(
        0, pair_i.unsqueeze(-1).expand_as(pair_forces), pair_forces
    )
    atomic_forces.scatter_add_(
        0, pair_j.unsqueeze(-1).expand_as(pair_forces), -pair_forces
    )

    return energy.detach(), atomic_forces.detach()


def test_energy_matches(potential, nacl_perturbed):
    """Energy from pbc_offsets trick must equal standard forward."""
    ref_energy, _, ref_input = _get_reference(potential, nacl_perturbed)
    trick_energy, _ = _run_pbc_offsets_trick(potential.model, ref_input, nacl_perturbed)
    np.testing.assert_allclose(
        trick_energy.numpy(),
        ref_energy.numpy(),
        atol=1e-5,
        err_msg="Energy mismatch between normal and pbc_offsets trick",
    )


def test_forces_match(potential, nacl_perturbed):
    """Atomic forces aggregated from pair forces must match standard forces."""
    ref_energy, ref_forces, ref_input = _get_reference(potential, nacl_perturbed)
    _, trick_forces = _run_pbc_offsets_trick(potential.model, ref_input, nacl_perturbed)
    np.testing.assert_allclose(
        trick_forces.numpy(),
        ref_forces.numpy(),
        atol=1e-4,
        err_msg="Force mismatch between normal and pbc_offsets trick",
    )


# ------------------------------------------------------------------
# M3GNetLammps wrapper tests (no LAMMPS needed)
# ------------------------------------------------------------------


def _noop_exchange(buf_from, buf_to, vec_len):
    """No-op exchange: just copies buf_from → buf_to (no ghosts to sync)."""
    buf_to[:] = buf_from


def test_m3gnet_lammps_energy_matches(potential, nacl_perturbed):
    """M3GNetLammps with no-op exchange must match vanilla M3GNet energy."""
    ref_energy, _, ref_input = _get_reference(potential, nacl_perturbed)
    ntotal = int(ref_input["num_atoms"].sum().item())

    wrapper = M3GnetLammps(potential.model)
    wrapper_energy = wrapper(
        ref_input,
        nlocal=ntotal,
        forward_exchange_fn=_noop_exchange,
        reverse_exchange_fn=_noop_exchange,
    )

    np.testing.assert_allclose(
        wrapper_energy.detach().numpy(),
        ref_energy.numpy(),
        atol=1e-5,
        err_msg="M3GNetLammps energy doesn't match vanilla M3GNet",
    )
