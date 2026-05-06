"""LAMMPS MLIAP wrapper for MatterSim (M3GNet).

Implements the MLIAPUnified interface so that a pre-trained M3GNet model can
be used as a pair_style in LAMMPS via Kokkos (single or multi-GPU).

Usage:
    # Deploy
    from mattersim.lammps.mliap_wrapper import MatterSimMLIAP
    mliap = MatterSimMLIAP.from_checkpoint("mattersim-v1.0.0-1M")
    mliap.save("mattersim-v1.0.0-1M_mliap.pt")

    # In LAMMPS input script:
    #   pair_style mliap unified mattersim-v1.0.0-1M_mliap.pt
    #   pair_coeff * * <element list>
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import ase.data
import torch
from lammps.mliap.mliap_unified_abc import MLIAPUnified

from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
from mattersim.forcefield.potential import Potential
from mattersim.lammps.graph_builder import build_m3gnet_input_from_lammps
from mattersim.lammps.m3gnet_lammps import M3GnetLammps

LOG = logging.getLogger(__name__)

# M3GNet supports elements 1..94
_MAX_ATOMIC_NUMBER = 94


def _freeze(model: torch.nn.Module) -> torch.nn.Module:
    """Freeze weights and switch to eval mode."""
    model = copy.deepcopy(model)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


class MatterSimMLIAP(MLIAPUnified):  # type: ignore[misc]
    """MatterSim (M3GNet) integration for LAMMPS via the MLIAP interface.

    Uses inter-layer exchange to keep ghost atom embeddings in sync,
    enabling correct multi-hop message passing without requiring atom tags.
    Works for both single-GPU and multi-GPU configurations.
    """

    def __init__(
        self,
        model: M3Gnet,
        *,
        cutoff: float = 5.0,
        threebody_cutoff: float = 4.0,
        compile: bool = True,
    ) -> None:
        super().__init__()

        self.m3gnet_lammps = M3GnetLammps(_freeze(model))
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        self._compile = compile

        # LAMMPS doubles rcutfac internally.
        self.rcutfac = 0.5 * cutoff

        self.element_types = [
            ase.data.chemical_symbols[z] for z in range(1, _MAX_ATOMIC_NUMBER + 1)
        ]
        self.ndescriptors = 1
        self.nparams = 1

        self.device = torch.device("cpu")
        self.initialized = False

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        version: str = "mattersim-v1.0.0-5M",
        *,
        device: str = "cpu",
        **kwargs: Any,
    ) -> "MatterSimMLIAP":
        """Load a pre-trained MatterSim model for use with LAMMPS.

        Args:
            version: Model version (e.g. ``"mattersim-v1.0.0-1M"``,
                ``"mattersim-v1.0.0-5M"``) or path to a local checkpoint file.
                Downloads automatically on first use.
            device: Device for model loading (default ``"cpu"``).
        """
        potential = Potential.from_checkpoint(load_path=version, device=device)
        m3gnet: M3Gnet = potential.model  # type: ignore[assignment]
        cutoff = m3gnet.model_args["cutoff"]
        threebody_cutoff = m3gnet.model_args["threebody_cutoff"]
        return cls(m3gnet, cutoff=cutoff, threebody_cutoff=threebody_cutoff, **kwargs)

    def save(self, path: str) -> None:
        """Save the MLIAP wrapper for use with LAMMPS pair_style mliap.

        Always saves with model on CPU so LAMMPS can load it on any device;
        _initialize_device moves to GPU at runtime if needed.
        """
        saved = copy.deepcopy(self)
        saved.m3gnet_lammps = M3GnetLammps(saved.m3gnet_lammps.m3gnet.cpu())
        torch.save(saved, path)

    # ------------------------------------------------------------------
    # Device handling
    # ------------------------------------------------------------------

    def _initialize_device(self, data) -> None:  # type: ignore[no-untyped-def]
        # Detect device from LAMMPS data tensors (Kokkos puts them on GPU)
        device_tensor = torch.as_tensor(data.elems)
        device = device_tensor.device if device_tensor.is_cuda else torch.device("cpu")

        self.device = device
        self.m3gnet_lammps = self.m3gnet_lammps.to(device)

        if self._compile and device.type == "cuda":
            LOG.info("Compiling M3GNetLammps with torch.compile ...")
            self.m3gnet_lammps = torch.compile(self.m3gnet_lammps)

        # MPI calls (forward/reverse exchange) happen inside autograd backward.
        # Autograd worker threads violate MPI/UCX thread ownership, so force
        # backward to run on the main thread.
        torch.autograd.set_multithreading_enabled(False)

        LOG.info(f"MatterSimMLIAP initialized on device: {device}")
        self.initialized = True

    # ------------------------------------------------------------------
    # MLIAP interface
    # ------------------------------------------------------------------

    def compute_forces(self, data) -> None:  # type: ignore[no-untyped-def]
        if not self.initialized:
            self._initialize_device(data)

        if data.nlocal == 0:
            return

        # Convert LAMMPS data → tensors
        elems = torch.as_tensor(data.elems, dtype=torch.int64, device=self.device) + 1
        pair_i = torch.as_tensor(data.pair_i, dtype=torch.int64, device=self.device)
        pair_j = torch.as_tensor(data.pair_j, dtype=torch.int64, device=self.device)
        rij = torch.as_tensor(data.rij, dtype=torch.float32, device=self.device)

        input_dict = build_m3gnet_input_from_lammps(
            elems=elems,
            pair_i=pair_i,
            pair_j=pair_j,
            rij=rij,
            ntotal=data.ntotal,
            cutoff=self.cutoff,
            threebody_cutoff=self.threebody_cutoff,
            device=self.device,
        )

        sort_idx = input_dict.pop("_sort_idx")
        edge_mask = input_dict.pop("_edge_mask")
        npairs_lammps = pair_i.shape[0]

        input_dict["pbc_offsets"].requires_grad_(True)

        energy = self.m3gnet_lammps(
            input_dict,
            nlocal=data.nlocal,
            forward_exchange_fn=data.forward_exchange,
            reverse_exchange_fn=data.reverse_exchange,
        )

        # pbc_offsets encode pair displacement vectors (rij). With atom_pos=0
        # and cell=I, the model sees positions solely through pbc_offsets.
        # ∂E/∂pbc_offsets then gives the pair force on each edge, which is
        # exactly what LAMMPS needs via data.update_pair_forces().
        (pair_forces_sorted,) = torch.autograd.grad(
            outputs=energy,
            inputs=input_dict["pbc_offsets"],
            grad_outputs=torch.ones_like(energy),
        )

        inverse_sort = torch.argsort(sort_idx)
        pair_forces_filtered = pair_forces_sorted[inverse_sort]

        pair_forces = torch.zeros(
            (npairs_lammps, 3), dtype=pair_forces_filtered.dtype, device=self.device
        )
        pair_forces[edge_mask] = pair_forces_filtered

        self._update_lammps_data(data, pair_forces.detach(), energy.detach())

    def _update_lammps_data(
        self,
        data,  # type: ignore[no-untyped-def]
        pair_forces: torch.Tensor,
        energy: torch.Tensor,
    ) -> None:
        pair_forces = pair_forces.double()

        if self.device.type == "cpu":
            data.update_pair_forces(pair_forces.numpy())
        else:
            data.update_pair_forces_gpu(pair_forces)
        data.energy = energy.item()

    def compute_descriptors(self, data) -> None:  # type: ignore[no-untyped-def]
        return

    def compute_gradients(self, data) -> None:  # type: ignore[no-untyped-def]
        return
