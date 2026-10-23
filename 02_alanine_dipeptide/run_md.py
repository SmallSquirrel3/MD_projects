"""Alanine dipeptide (Ace-Ala-Nme): the "hydrogen atom" of protein conformational dynamics.

Runs Langevin dynamics with the AMBER ff14SB force field in vacuum and in GBn2 implicit
water, then draws Ramachandran (phi/psi) maps and free-energy surfaces F = -kT ln P.

Run:  python run_md.py                 (~30 min on a laptop CPU, 10 ns per solvent model)
      python run_md.py --ns 2 --quick  (smoke test)
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import openmm as mm
from MDAnalysis.analysis.dihedrals import Ramachandran
from openmm import app, unit

HERE = Path(__file__).parent
KT_KCAL = 0.0019872 * 300  # kT at 300 K in kcal/mol


def simulate(solvent, ns, out, seed):
    pdb = app.PDBFile(str(HERE / "data" / "alanine-dipeptide.pdb"))
    files = ["amber14-all.xml"] + (["implicit/gbn2.xml"] if solvent == "gbn2" else [])
    ff = app.ForceField(*files)
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)
    integrator.setRandomNumberSeed(seed)
    # 22 atoms are too few to split between CPU threads: one thread is ~3x faster here
    platform = mm.Platform.getPlatformByName("CPU")
    sim = app.Simulation(pdb.topology, system, integrator, platform, {"Threads": "1"})
    sim.context.setPositions(pdb.positions)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, seed)
    sim.step(5000)  # 10 ps of equilibration, not saved

    n_steps = int(ns * 1e6 / 2)
    dcd = out / f"ala2_{solvent}.dcd"
    sim.reporters.append(app.DCDReporter(str(dcd), 500))  # one frame per ps
    sim.reporters.append(app.StateDataReporter(str(out / f"ala2_{solvent}.log"), 5000, step=True,
                                               potentialEnergy=True, temperature=True, speed=True))
    sim.step(n_steps)
    with open(out / "ala2.pdb", "w") as fh:
        app.PDBFile.writeFile(sim.topology, sim.context.getState(getPositions=True).getPositions(), fh)
    return dcd


def free_energy_2d(phi, psi, bins=60):
    h, xe, ye = np.histogram2d(phi, psi, bins=bins, range=[[-180, 180], [-180, 180]], density=True)
    with np.errstate(divide="ignore"):
        f = -KT_KCAL * np.log(h)
    return f - np.nanmin(f[np.isfinite(f)]), xe, ye


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ns", type=float, default=10.0, help="production length per solvent model, ns")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="results")
    p.add_argument("--seed", type=int, default=2024)
    args = p.parse_args()
    if args.quick:
        args.ns = 0.2
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, solvent in zip(axes, ["vacuum", "gbn2"]):
        print(f"Simulating {args.ns} ns in {solvent} ...")
        dcd = simulate(solvent, args.ns, out, args.seed)
        u = mda.Universe(str(out / "ala2.pdb"), str(dcd))
        rama = Ramachandran(u.select_atoms("resname ALA")).run()
        phi, psi = rama.results.angles[:, 0, 0], rama.results.angles[:, 0, 1]
        f, xe, ye = free_energy_2d(phi, psi)
        im = ax.contourf(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), f.T,
                         levels=np.arange(0, 8.5, 0.5), cmap="viridis_r", extend="max")
        ax.set(title=f"{solvent}: {len(phi)} frames", xlabel="phi (deg)", xlim=(-180, 180), ylim=(-180, 180))
        # Population of the left-handed region (phi > 0): rare, so a good sampling check
        print(f"  fraction with phi > 0 (alpha_L / C7ax): {np.mean(phi > 0):.3f}")
    axes[0].set_ylabel("psi (deg)")
    fig.colorbar(im, ax=axes, label="F (kcal/mol)")
    fig.savefig(out / "ramachandran_fes.png", dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out / 'ramachandran_fes.png'}")


if __name__ == "__main__":
    main()
