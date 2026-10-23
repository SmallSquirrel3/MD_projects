"""Does the force field get the torsion right? A relaxed QM torsion scan vs OpenFF Sage.

Torsions are the softest and least transferable part of a force field, and the first thing to
check for a new ligand. Here: the C(methyl)-O-C(ring)-C(ring) torsion of anisole (methoxybenzene),
a fragment found in many drugs. QM says the methoxy group prefers to lie in the ring plane.

  QM: B3LYP-D3(BJ)/def2-SVP in PySCF, constrained optimisations with geomeTRIC
  MM: OpenFF Sage 2.2 in OpenMM, same dihedral restrained, everything else relaxed

Run:  python torsion_scan.py                       (7 points, ~20-40 min on a laptop)
      python torsion_scan.py --smiles "CC(=O)Nc1ccccc1" --smarts "[C:1](=O)[N:2][c:3][c:4]"   (acetanilide)
      python torsion_scan.py --quick                (3 points, HF/STO-3G, smoke test)
"""
import argparse
import logging
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
from openff.toolkit import Molecule
from openmm import app, unit
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from pyscf import dft, gto, scf
from pyscf.geomopt.geometric_solver import optimize

HARTREE_TO_KCAL = 627.509
logging.getLogger("geometric").setLevel(logging.ERROR)  # geomeTRIC is chatty


def set_dihedral(xyz, idx, target_deg, moving):
    """Rotate the `moving` atoms about the j-k bond so that dihedral i-j-k-l equals the target."""
    i, j, k, l = idx
    current = dihedral(xyz, idx)
    angle = np.radians(target_deg - current)
    axis = (xyz[k] - xyz[j]) / np.linalg.norm(xyz[k] - xyz[j])
    kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx
    new = xyz.copy()
    new[moving] = (xyz[moving] - xyz[j]) @ rot.T + xyz[j]
    return new


def dihedral(xyz, idx):
    p0, p1, p2, p3 = xyz[list(idx)]
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 /= np.linalg.norm(b1)
    v, w = b0 - np.dot(b0, b1) * b1, b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def side_atoms(mol, j, k):
    """Atoms on the j side of the j-k bond (these are the ones we rotate)."""
    neighbours = {a.molecule_atom_index: [n.molecule_atom_index for n in a.bonded_atoms] for a in mol.atoms}
    seen, stack = {j}, [j]
    while stack:
        for n in neighbours[stack.pop()]:
            if n not in seen and n != k:
                seen.add(n)
                stack.append(n)
    return sorted(seen)


def qm_constrained_opt(symbols, xyz, idx, angle, method, basis):
    mol = gto.M(atom=list(zip(symbols, xyz)), basis=basis, verbose=0)
    if method == "hf":
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol)
        mf.xc = "b3lyp"
        mf.disp = "d3bj"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        # geomeTRIC uses 1-based atom indices
        fh.write(f"$freeze\ndihedral {' '.join(str(i + 1) for i in idx)}\n")
        constraints = fh.name
    mol_eq = optimize(mf, constraints=constraints, maxsteps=100)
    mf.reset(mol_eq)
    energy = mf.kernel()
    return energy * HARTREE_TO_KCAL, mol_eq.atom_coords(unit="Angstrom")


def mm_restrained_energy(simulation, restraint, xyz_angstrom, angle):
    simulation.context.setPositions(xyz_angstrom * 0.1)
    simulation.context.setParameter("theta0", np.radians(angle))
    mm.LocalEnergyMinimizer.minimize(simulation.context, 1e-3, 0)
    # report the force-field energy without the restraint (force group 1)
    state = simulation.context.getState(getEnergy=True, groups={0})
    return state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smiles", default="COc1ccccc1")
    p.add_argument("--smarts", default="[CH3:1][O:2][c:3][c:4]", help="the 4 mapped atoms define the torsion")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--stop", type=float, default=180.0)
    p.add_argument("--step", type=float, default=30.0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    method, basis = ("hf", "sto-3g") if args.quick else ("b3lyp-d3bj", "def2-svp")
    if args.quick:
        args.step = 90.0
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    mol = Molecule.from_smiles(args.smiles)
    mol.generate_conformers(n_conformers=1)
    idx = mol.chemical_environment_matches(args.smarts)[0]
    moving = side_atoms(mol, idx[1], idx[2])
    symbols = [a.symbol for a in mol.atoms]
    xyz0 = mol.conformers[0].m_as("angstrom")

    # MM setup: OpenFF parameters + a stiff restraint on the scanned dihedral in its own force group
    try:
        mol.assign_partial_charges("openff-gnn-am1bcc-1.0.0.pt")
    except Exception:
        mol.assign_partial_charges("am1bcc")
    ff = app.ForceField()
    ff.registerTemplateGenerator(SMIRNOFFTemplateGenerator(molecules=[mol], forcefield="openff-2.2.0").generator)
    top = mol.to_topology().to_openmm()
    system = ff.createSystem(top, nonbondedMethod=app.NoCutoff)
    restraint = mm.CustomTorsionForce("0.5*k*min(dtheta, 2*pi-dtheta)^2; dtheta = abs(theta-theta0); pi = 3.1415926535")
    restraint.addGlobalParameter("k", 2000.0)  # kJ/mol/rad^2
    restraint.addGlobalParameter("theta0", 0.0)
    restraint.addTorsion(*idx)
    restraint.setForceGroup(1)
    system.addForce(restraint)
    sim = app.Simulation(top, system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))

    angles = np.arange(args.start, args.stop + 1e-6, args.step)
    qm, mm_e, achieved = [], [], []
    for angle in angles:
        start = set_dihedral(xyz0, idx, angle, moving)
        e_qm, xyz_opt = qm_constrained_opt(symbols, start, idx, angle, method, basis)
        e_mm = mm_restrained_energy(sim, restraint, xyz_opt, angle)
        qm.append(e_qm)
        mm_e.append(e_mm)
        achieved.append(dihedral(xyz_opt, idx))
        print(f"dihedral {angle:6.1f} deg: QM {e_qm:.3f}, MM {e_mm:.3f} kcal/mol", flush=True)

    qm = np.array(qm) - min(qm)
    mm_e = np.array(mm_e) - min(mm_e)
    rmse = np.sqrt(np.mean((qm - mm_e) ** 2))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(angles, qm, "o-", label=f"QM {method}/{basis}")
    ax.plot(angles, mm_e, "s--", label="MM OpenFF Sage 2.2")
    ax.set(xlabel="dihedral (deg)", ylabel="relative energy (kcal/mol)",
           title=f"{args.smiles}: RMSE {rmse:.2f} kcal/mol")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "torsion_profile.png", dpi=150)
    np.savetxt(out / "torsion_profile.csv", np.c_[angles, qm, mm_e], delimiter=",",
               header="dihedral_deg,qm_kcal,mm_kcal", comments="")
    print(f"QM barrier {qm.max():.2f}, MM barrier {mm_e.max():.2f}, RMSE {rmse:.2f} kcal/mol. "
          f"Figure: {out / 'torsion_profile.png'}")


if __name__ == "__main__":
    main()
