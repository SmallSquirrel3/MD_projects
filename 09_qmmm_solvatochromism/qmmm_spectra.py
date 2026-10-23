"""QM/MM electrostatic embedding: why acetone absorbs bluer in water than in the gas phase.

The n -> pi* band of acetone shifts to higher energy in water by about +0.2 eV (~1500 cm^-1):
water hydrogen-bonds to the carbonyl oxygen and stabilises the lone pair (n) orbital that the
electron leaves. This is the textbook test of QM/MM, and the same recipe (MD for sampling,
QM with the MM point charges in the Hamiltonian for the chemistry) is how enzyme reactions,
chromophores in proteins (GFP, rhodopsin) and pKa shifts in active sites are studied.

  1. classical MD of acetone in TIP3P water (OpenFF Sage + OpenMM)
  2. for N snapshots: TDDFT (CAM-B3LYP) on acetone, with and without the surrounding water
     as point charges (PySCF qmmm.mm_charge)
  3. the solvent shift is averaged over snapshots: one snapshot is an anecdote, not a spectrum

Run:  python qmmm_spectra.py             (20 snapshots, ~15-30 min on a laptop)
      python qmmm_spectra.py --quick     (smoke test)
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
from openff.toolkit import Molecule
from openmm import app, unit
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from pyscf import dft, gto, qmmm, tdscf

HARTREE_TO_EV = 27.211386
DEBYE_PER_AU = 2.541746


def run_md(mol, n_snapshots, interval_ps, quick):
    ff = app.ForceField("amber/tip3p_standard.xml")
    ff.registerTemplateGenerator(SMIRNOFFTemplateGenerator(molecules=[mol], forcefield="openff-2.2.0").generator)
    modeller = app.Modeller(mol.to_topology().to_openmm(), mol.conformers[0].to_openmm())
    modeller.addSolvent(ff, padding=1.2 * unit.nanometer)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * unit.nanometer,
                             constraints=app.HBonds)
    system.addForce(mm.MonteCarloBarostat(1 * unit.bar, 298 * unit.kelvin))
    nb =next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    charges = np.array([nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
                        for i in range(system.getNumParticles())])
    integrator = mm.LangevinMiddleIntegrator(298 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)
    sim = app.Simulation(modeller.topology, system, integrator)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(298 * unit.kelvin, 5)
    sim.step(1000 if quick else 50_000)  # 100 ps equilibration
    frames = []
    for _ in range(n_snapshots):
        sim.step(int(interval_ps * 500))
        state = sim.context.getState(getPositions=True)
        box = np.diag(state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom))
        frames.append((state.getPositions(asNumpy=True).value_in_unit(unit.angstrom), box))
    waters = [[a.index for a in r.atoms()] for r in modeller.topology.residues() if r.name == "HOH"]
    return frames, charges, waters


def embedding_charges(xyz, box, solute_idx, waters, charges, cutoff):
    """Whole water molecules within `cutoff` of the solute centre, unwrapped by minimum image."""
    centre = xyz[solute_idx].mean(axis=0)
    coords, q = [], []
    for w in waters:
        shift = box * np.round((xyz[w[0]] - centre) / box)
        pos = xyz[w] - shift
        if np.linalg.norm(pos[0] - centre) < cutoff:
            coords.extend(pos)
            q.extend(charges[w])
    return np.array(coords), np.array(q)


def excitations(symbols, xyz, basis, xc, mm_coords=None, mm_charges=None):
    mol = gto.M(atom=list(zip(symbols, xyz)), basis=basis, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = xc
    if mm_coords is not None:
        mf = qmmm.mm_charge(mf, mm_coords, mm_charges, unit="Angstrom")  # charges enter the QM Hamiltonian
    mf.kernel()
    td = tdscf.TDA(mf)
    td.nstates = 4
    td.kernel()
    # n -> pi* is the lowest, nearly dark state: pick the lowest excitation
    e_ev = np.min(td.e) * HARTREE_TO_EV
    dip = np.linalg.norm(mf.dip_moment(unit="Debye", verbose=0))
    return e_ev, dip


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smiles", default="CC(C)=O")
    p.add_argument("--snapshots", type=int, default=20)
    p.add_argument("--interval-ps", type=float, default=10.0, help="time between snapshots")
    p.add_argument("--cutoff", type=float, default=12.0, help="radius of embedded waters, A")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    basis, xc = "6-31+G*", "camb3lyp"
    if args.quick:
        args.snapshots, args.interval_ps, basis, xc = 2, 1.0, "sto-3g", "b3lyp"
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    mol = Molecule.from_smiles(args.smiles)
    mol.generate_conformers(n_conformers=1)
    try:
        mol.assign_partial_charges("openff-gnn-am1bcc-1.0.0.pt")
    except Exception:
        mol.assign_partial_charges("am1bcc")
    symbols = [a.symbol for a in mol.atoms]
    solute = list(range(mol.n_atoms))

    print(f"MD of {args.smiles} in water, {args.snapshots} snapshots every {args.interval_ps} ps ...")
    frames, charges, waters = run_md(mol, args.snapshots, args.interval_ps, args.quick)

    gas, solv, dip_gas, dip_solv = [], [], [], []
    for n, (xyz, box) in enumerate(frames):
        mm_xyz, mm_q = embedding_charges(xyz, box, solute, waters, charges, args.cutoff)
        e_g, d_g = excitations(symbols, xyz[solute], basis, xc)
        e_s, d_s = excitations(symbols, xyz[solute], basis, xc, mm_xyz, mm_q)
        gas.append(e_g), solv.append(e_s), dip_gas.append(d_g), dip_solv.append(d_s)
        print(f"snapshot {n + 1:2d}: {len(mm_q) // 3} waters, n->pi* gas {e_g:.3f} eV, "
              f"embedded {e_s:.3f} eV, shift {e_s - e_g:+.3f} eV", flush=True)

    shift = np.array(solv) - np.array(gas)
    sem = shift.std(ddof=1) / np.sqrt(len(shift)) if len(shift) > 1 else float("nan")
    report = (f"{xc}/{basis}, {len(shift)} snapshots\n"
              f"mean n->pi* (same geometries): gas {np.mean(gas):.3f} eV, in water {np.mean(solv):.3f} eV\n"
              f"solvatochromic shift {shift.mean():+.3f} +/- {sem:.3f} eV "
              f"({shift.mean() * 8065.5:+.0f} cm^-1); experiment for acetone: about +0.2 eV (blue shift)\n"
              f"dipole moment: {np.mean(dip_gas):.2f} D in gas -> {np.mean(dip_solv):.2f} D polarised by water\n")
    (out / "result.txt").write_text(report)
    print(report)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(gas, bins=10, alpha=0.6, label="gas (same geometry)")
    ax[0].hist(solv, bins=10, alpha=0.6, label="in water (QM/MM)")
    ax[0].set(xlabel="n -> pi* excitation energy (eV)", ylabel="snapshots", title="Spectral distribution")
    ax[0].legend()
    ax[1].plot(range(1, len(shift) + 1), np.cumsum(shift) / np.arange(1, len(shift) + 1), marker="o")
    ax[1].axhline(0.2, ls="--", c="grey", label="experiment ~ +0.2 eV")
    ax[1].set(xlabel="number of snapshots", ylabel="running mean shift (eV)", title="Convergence of the shift")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "solvatochromism.png", dpi=150)


if __name__ == "__main__":
    main()
