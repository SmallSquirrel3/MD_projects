"""Well-tempered metadynamics of alanine dipeptide on the (phi, psi) Ramachandran plane.

In plain MD (project 2) the left-handed region (phi > 0) is almost never visited: the barrier
is ~5-8 kcal/mol. Metadynamics keeps dropping Gaussian "sand piles" where the system has been,
fills the valleys, and the system is pushed over the barrier. The negative of the accumulated
bias converges to the free-energy surface F(phi, psi).

Uses OpenMM's built-in Metadynamics class (no PLUMED needed).

Run:  python metad.py              (5 ns, ~10-20 min on a laptop CPU)
      python metad.py --quick      (smoke test)
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
from openmm import app, unit
from openmm.app.metadynamics import BiasVariable, Metadynamics

HERE = Path(__file__).parent
PDB = HERE.parent / "02_alanine_dipeptide" / "data" / "alanine-dipeptide.pdb"
KJ_TO_KCAL = 1 / 4.184


def dihedral_force(topology, names):
    """names: list of (residue name, atom name) for the four atoms."""
    idx = []
    for res_name, atom_name in names:
        idx += [a.index for a in topology.atoms() if a.residue.name == res_name and a.name == atom_name]
    force = mm.CustomTorsionForce("theta")
    force.addTorsion(*idx)
    return force


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ns", type=float, default=5.0)
    p.add_argument("--bias-factor", type=float, default=6.0, help="well-tempered gamma = (T + dT) / T")
    p.add_argument("--height", type=float, default=1.0, help="initial Gaussian height, kJ/mol")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    if args.quick:
        args.ns = 0.1
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    pdb = app.PDBFile(str(PDB))
    ff = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    phi = dihedral_force(pdb.topology, [("ACE", "C"), ("ALA", "N"), ("ALA", "CA"), ("ALA", "C")])
    psi = dihedral_force(pdb.topology, [("ALA", "N"), ("ALA", "CA"), ("ALA", "C"), ("NME", "N")])
    width = 0.35  # radians (~20 degrees): about the width of a basin's fluctuations
    cvs = [BiasVariable(phi, -np.pi, np.pi, width, periodic=True, gridWidth=72),
           BiasVariable(psi, -np.pi, np.pi, width, periodic=True, gridWidth=72)]
    meta = Metadynamics(system, cvs, 300 * unit.kelvin, args.bias_factor, args.height * unit.kilojoule_per_mole,
                        frequency=500)  # one Gaussian per ps

    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)
    integrator.setRandomNumberSeed(11)
    sim = app.Simulation(pdb.topology, system, integrator, mm.Platform.getPlatformByName("CPU"), {"Threads": "1"})
    sim.context.setPositions(pdb.positions)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, 11)

    n_chunks = 20
    steps_per_chunk = int(args.ns * 5e5 / n_chunks)
    cv_trace, delta_f, t_ns = [], [], []
    for chunk in range(n_chunks):
        for _ in range(steps_per_chunk // 500):
            meta.step(sim, 500)
            cv_trace.append(meta.getCollectiveVariables(sim))
        fes = meta.getFreeEnergy().value_in_unit(unit.kilojoule_per_mole) * KJ_TO_KCAL  # shape (psi, phi)
        grid = np.linspace(-180, 180, fes.shape[0])
        right = fes[:, grid < 0].min()  # beta / alpha_R basins
        left = fes[:, grid > 0].min()  # alpha_L / C7ax basins
        delta_f.append(left - right)
        t_ns.append((chunk + 1) * steps_per_chunk * 2e-6)
        print(f"{t_ns[-1]:6.2f} ns  F(phi>0) - F(phi<0) = {delta_f[-1]:5.2f} kcal/mol")

    fes -= fes.min()
    cv_trace = np.degrees(np.array(cv_trace))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    im = ax[0].contourf(grid, grid, fes, levels=np.arange(0, 12.5, 0.5), cmap="viridis_r", extend="max")
    fig.colorbar(im, ax=ax[0], label="F (kcal/mol)")
    ax[0].set(xlabel="phi (deg)", ylabel="psi (deg)", title="Free-energy surface from the bias")
    ax[1].plot(np.arange(len(cv_trace)) / 1000, cv_trace[:, 0], ".", ms=1)
    ax[1].set(xlabel="time (ns)", ylabel="phi (deg)", title="phi crosses the barrier again and again")
    ax[2].plot(t_ns, delta_f, marker="o")
    ax[2].set(xlabel="time (ns)", ylabel="Delta F (kcal/mol)", title="Convergence: F(phi > 0) - F(phi < 0)")
    fig.tight_layout()
    fig.savefig(out / "metad_fes.png", dpi=150)
    np.save(out / "fes_kcal.npy", fes)
    crossings = np.sum(np.diff(np.sign(cv_trace[:, 0])) != 0)
    print(f"phi switched between the phi<0 and phi>0 halves {crossings} times. Figure: {out / 'metad_fes.png'}")


if __name__ == "__main__":
    main()
