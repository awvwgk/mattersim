"""Validate LAMMPS MatterSim output against Python Potential.

Reads the LAMMPS trajectory from the current directory, recomputes
energy/forces/stress with Potential.forward(), and prints MAE errors.

Usage:
    cd examples
    python validate.py          # reads dump.lammpstrj, pe.dat, stress.dat from cwd
    python validate.py --plot   # also save parity plots
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import typer

app = typer.Typer(pretty_exceptions_enable=False)

EXAMPLES_DIR = Path(__file__).parent


def parse_fix_print(path: Path, ncols: int) -> list[list[float]]:
    """Parse a LAMMPS fix-print file, returning rows of ncols floats (skipping step)."""
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 1 + ncols:
                rows.append([float(x) for x in parts[1 : 1 + ncols]])
    return rows


def read_lammps_trajectory(
    dump_path: Path,
    pe_path: Path,
    stress_path: Path,
) -> list[dict]:
    """Read LAMMPS dump + pe.dat + stress.dat into list of dicts."""
    energies = [r[0] for r in parse_fix_print(pe_path, 1)]

    # LAMMPS virial pressure in bars -> stress in GPa (stress = -pressure)
    # Voigt order: xx, yy, zz, yz, xz, xy
    raw_stress = parse_fix_print(stress_path, 6)  # pxx pyy pzz pxy pxz pyz
    stresses_gpa = []
    for row in raw_stress:
        pxx, pyy, pzz, pxy, pxz, pyz = row
        voigt = np.array([-pxx, -pyy, -pzz, -pyz, -pxz, -pxy]) / 1e4
        stresses_gpa.append(voigt)

    # Parse LAMMPS custom dump
    frames: list[dict] = []
    with open(dump_path) as f:
        lines = f.readlines()

    i = 0
    frame_idx = 0
    while i < len(lines):
        if "ITEM: TIMESTEP" in lines[i]:
            i += 1
            i += 1  # ITEM: NUMBER OF ATOMS
            i += 1
            natoms = int(lines[i].strip())
            i += 1  # ITEM: BOX BOUNDS
            i += 1
            box_bounds = []
            for _ in range(3):
                lo, hi = [float(x) for x in lines[i].strip().split()[:2]]
                box_bounds.append(hi - lo)
                i += 1
            cell = np.diag(box_bounds)
            i += 1  # ITEM: ATOMS ...
            positions = np.zeros((natoms, 3))
            forces = np.zeros((natoms, 3))
            for _ in range(natoms):
                parts = lines[i].strip().split()
                idx = int(parts[0]) - 1
                positions[idx] = [float(parts[2]), float(parts[3]), float(parts[4])]
                forces[idx] = [float(parts[5]), float(parts[6]), float(parts[7])]
                i += 1
            frames.append(
                {
                    "positions": positions,
                    "forces": forces,
                    "cell": cell,
                    "energy": energies[frame_idx]
                    if frame_idx < len(energies)
                    else None,
                    "stress": stresses_gpa[frame_idx]
                    if frame_idx < len(stresses_gpa)
                    else None,
                    "natoms": natoms,
                }
            )
            frame_idx += 1
        else:
            i += 1
    return frames


def recompute_with_potential(
    potential,  # type: ignore[no-untyped-def]
    frames: list[dict],
    device: str = "cpu",
) -> tuple[list[float], list[np.ndarray], list[np.ndarray]]:
    """Run frames through Potential -> energy/forces/stress."""
    from ase import Atoms

    from mattersim.datasets.utils.build import build_dataloader

    element = "Cu"
    atoms_list = [
        Atoms(element * f["natoms"], positions=f["positions"], cell=f["cell"], pbc=True)
        for f in frames
    ]

    dataloader = build_dataloader(atoms_list, batch_size=1, only_inference=True)
    predictions = potential.predict_properties(
        dataloader,
        include_forces=True,
        include_stresses=True,
    )
    energies = [float(e) for e in predictions[0]]
    all_forces = [np.array(f) for f in predictions[1]]
    # predict_properties returns 3x3 stress in GPa -> Voigt [xx,yy,zz,yz,xz,xy]
    all_stresses = []
    for s in predictions[2]:
        s = np.array(s)
        voigt = np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])
        all_stresses.append(voigt)

    return energies, all_forces, all_stresses


@app.command()
def validate(
    output_dir: Path = typer.Option(
        ".", help="Directory containing LAMMPS output files"
    ),
    version: str = typer.Option("mattersim-v1.0.0-5M", help="MatterSim version"),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        help="Device for Python recomputation",
    ),
    plot: bool = typer.Option(False, help="Save parity plots"),
) -> None:
    from mattersim.forcefield.potential import Potential

    out = Path(output_dir)

    # 1. Read LAMMPS output
    dump_path = out / "dump.lammpstrj"
    pe_path = out / "pe.dat"
    stress_path = out / "stress.dat"

    for p in [dump_path, pe_path, stress_path]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run lmp.py first.")

    frames = read_lammps_trajectory(dump_path, pe_path, stress_path)
    # Drop last frame (no forward call in LAMMPS for it)
    frames = frames[:-1]
    n = len(frames)
    print(f"Read {n} frames from LAMMPS output")

    lammps_energies = [f["energy"] for f in frames]
    lammps_forces = [f["forces"] for f in frames]
    lammps_stresses = [f["stress"] for f in frames]

    # 2. Recompute with Python Potential
    print(f"Recomputing with Potential on {device}...")
    potential = Potential.from_checkpoint(load_path=version, device=device)
    python_energies, python_forces, python_stresses = recompute_with_potential(
        potential,
        frames,
        device=device,
    )

    # 3. Compute errors
    e_diffs = [abs(lammps_energies[i] - python_energies[i]) for i in range(n)]
    f_diffs = [np.abs(lammps_forces[i] - python_forces[i]) for i in range(n)]
    s_diffs = [np.abs(lammps_stresses[i] - python_stresses[i]) for i in range(n)]

    natoms = frames[0]["natoms"]
    print(f"\n{'=' * 60}")
    print(f"  Validation Summary  ({natoms} atoms, {n} frames)")
    print(f"{'=' * 60}")
    e_mae = np.mean(e_diffs)
    e_mae_atom = e_mae / natoms
    e_max = np.max(e_diffs)
    f_mae = np.mean([f.mean() for f in f_diffs])
    f_max = np.max([f.max() for f in f_diffs])
    s_mae = np.mean([s.mean() for s in s_diffs])
    s_max = np.max([s.max() for s in s_diffs])
    print(f"  Energy MAE: {e_mae: .2e} eV")
    print(f"  Energy MAE/atom: {e_mae_atom: .2e} eV/atom")
    print(f"  Energy max diff: {e_max: .2e} eV")
    print(f"  Force MAE: {f_mae: .2e} eV/A")
    print(f"  Force max diff: {f_max: .2e} eV/A")
    print(f"  Stress MAE: {s_mae: .2e} GPa")
    print(f"  Stress max diff: {s_max: .2e} GPa")
    print(f"{'=' * 60}")

    # 4. Optionally make parity plots
    if plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        ax = axes[0]
        ax.plot(lammps_energies, python_energies, "o")
        lims = [
            min(lammps_energies + python_energies) - 0.01,
            max(lammps_energies + python_energies) + 0.01,
        ]
        ax.plot(lims, lims, "k--", alpha=0.5)
        ax.set_xlabel("LAMMPS energy (eV)")
        ax.set_ylabel("Python energy (eV)")
        ax.set_title(f"Energy (MAE={np.mean(e_diffs): .2e} eV)")

        ax = axes[1]
        lf = np.concatenate([f.flatten() for f in lammps_forces])
        pf = np.concatenate([f.flatten() for f in python_forces])
        ax.plot(lf, pf, ".", alpha=0.5, markersize=3)
        lims = [min(lf.min(), pf.min()), max(lf.max(), pf.max())]
        ax.plot(lims, lims, "k--", alpha=0.5)
        ax.set_xlabel("LAMMPS forces (eV/A)")
        ax.set_ylabel("Python forces (eV/A)")
        ax.set_title(f"Forces (MAE={np.mean([f.mean() for f in f_diffs]): .2e} eV/A)")

        ax = axes[2]
        ls = np.concatenate([s.flatten() for s in lammps_stresses])
        ps = np.concatenate([s.flatten() for s in python_stresses])
        ax.plot(ls, ps, "o", alpha=0.7, markersize=4)
        lims = [min(ls.min(), ps.min()), max(ls.max(), ps.max())]
        ax.plot(lims, lims, "k--", alpha=0.5)
        ax.set_xlabel("LAMMPS stress (GPa)")
        ax.set_ylabel("Python stress (GPa)")
        ax.set_title(f"Stress (MAE={np.mean([s.mean() for s in s_diffs]): .2e} GPa)")

        plt.tight_layout()
        out_png = out / "parity.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved parity plot to {out_png}")


if __name__ == "__main__":
    app()
