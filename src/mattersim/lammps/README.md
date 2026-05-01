# MatterSim LAMMPS Integration

Run MatterSim (M3GNet) molecular dynamics via LAMMPS using the MLIAP interface.
Supports single-GPU and multi-GPU domain decomposition.

## Prerequisites

- LAMMPS built with `ML-IAP` (with `MLIAP_ENABLE_PYTHON`), `PKG_PYTHON`, and
  Kokkos CUDA support
- Python packages: `mattersim`, `lammps`, `torch`

## Quick Start

### 1. Export the model

```python
from mattersim.lammps.mliap_wrapper import MatterSimMLIAP

# Choose your model: "mattersim-v1.0.0-1M" or "mattersim-v1.0.0-5M"
mliap = MatterSimMLIAP.from_checkpoint("mattersim-v1.0.0-1M", device="cpu")
mliap.save("mattersim-v1.0.0-1M-mliap.pt")
```

The checkpoint is downloaded automatically on first use.

### 2. Run LAMMPS

Use `pair_style mliap unified` with the exported model in your input file:

```lammps
pair_style      mliap unified mattersim-v1.0.0-1M-mliap.pt
pair_coeff      * * Cu
```

```bash
lmp -in input.in -k on g 1 -sf kk -pk kokkos newton on neigh half
```

### 3. Validate (optional)

Run the included example and verify LAMMPS output matches Python recomputation:

```bash
cd examples
lmp -in cu_nve.in -var MODEL_PATH /path/to/mattersim_mliap.pt \
    -k on g 1 -sf kk -pk kokkos newton on neigh half
python validate.py --version mattersim-v1.0.0-1M --plot
```

Expected errors: ~1e-6 eV/atom, ~1e-6 eV/Å, ~1e-6 GPa.

## Multi-GPU

Multi-GPU uses MPI domain decomposition. Each M3GNet layer syncs ghost atom
embeddings across ranks via LAMMPS exchange. Requires a CUDA-aware MPI stack.

```bash
mpirun -np 4 lmp -in input.in \
    -k on g 4 -sf kk \
    -pk kokkos newton on neigh half gpu/aware on \
    comm/pair/forward device comm/pair/reverse device
```

> **Note:** The exact MPI flags may vary depending on your MPI implementation
> and GPU-aware transport layer (e.g. UCX, libfabric). The above works with
> OpenMPI + Kokkos.

## How It Works

1. **Graph over all atoms:** The graph is built over `ntotal` atoms (local +
   ghost) using the LAMMPS neighbor list directly.

2. **The pbc_offsets trick:** With `pos=0`, `cell=I`, `pbc_offsets=rij`,
   M3GNet computes `edge_vector = -rij`. Differentiating energy w.r.t.
   `pbc_offsets` yields per-edge pair forces for LAMMPS.

3. **Inter-layer exchange:** In multi-GPU mode, `LammpsExchange` syncs ghost
   atom embeddings via `forward_exchange` in the forward pass and accumulates
   gradients via `reverse_exchange` in the backward pass.

## Limitations

- **CUDA-aware MPI required for multi-GPU.**
- **`mattersim` must be installed.** The `.pt` file contains pickled Python
  objects that require the `mattersim` package.
