"""Alchemical hydration free energy of a small molecule (default: ethanol, experiment -5.0 kcal/mol).

This is the same machinery as FEP for protein-ligand binding, on a system small enough for a laptop.
Thermodynamic cycle:

    solute in water  --(switch off charges, then van der Waals)-->  "ghost" solute in water
    solute in vacuum --(switch off charges)----------------------->  "ghost" solute in vacuum

    dG_hydration = dG_vacuum_leg - dG_water_leg

Each leg runs a ladder of lambda windows; every sample is evaluated at all lambdas and the
data are combined with MBAR (pymbar).

Run:  python hfe.py                         (ethanol, ~30-60 min on a laptop CPU)
      python hfe.py --smiles "c1ccccc1O"    (phenol, experiment -6.6 kcal/mol)
      python hfe.py --quick                 (smoke test, the number is meaningless)
"""
import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openmm as mm
from openff.toolkit import Molecule
from openmm import app, unit
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openmmtools import alchemy
from pymbar import MBAR

KT_KCAL = 0.0019872041 * 298.15
TEMPERATURE = 298.15 * unit.kelvin
# (lambda_electrostatics, lambda_sterics): charges first, then soft-core Lennard-Jones
WATER_SCHEDULE = [(1.0, 1.0), (0.75, 1.0), (0.5, 1.0), (0.25, 1.0), (0.0, 1.0),
                  (0.0, 0.9), (0.0, 0.8), (0.0, 0.7), (0.0, 0.6), (0.0, 0.5),
                  (0.0, 0.4), (0.0, 0.3), (0.0, 0.2), (0.0, 0.1), (0.0, 0.0)]
VACUUM_SCHEDULE = [(1.0, 1.0), (0.75, 1.0), (0.5, 1.0), (0.25, 1.0), (0.0, 1.0)]


def make_molecule(smiles):
    mol = Molecule.from_smiles(smiles)
    mol.generate_conformers(n_conformers=1)
    try:
        mol.assign_partial_charges("openff-gnn-am1bcc-1.0.0.pt")
    except Exception:
        mol.assign_partial_charges("am1bcc")
    return mol


def build(mol, solvated):
    ff = app.ForceField("amber/tip3p_standard.xml")
    ff.registerTemplateGenerator(SMIRNOFFTemplateGenerator(molecules=[mol], forcefield="openff-2.2.0").generator)
    top = mol.to_topology().to_openmm()
    modeller = app.Modeller(top, mol.conformers[0].to_openmm())
    if solvated:
        modeller.addSolvent(ff, padding=1.0 * unit.nanometer)
        system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * unit.nanometer,
                                 constraints=app.HBonds)
    else:
        system = ff.createSystem(modeller.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    return modeller, system


def equilibrate_box(modeller, system, steps):
    """Short NPT run to get the right water density; the alchemical runs are then NVT."""
    system = mm.XmlSerializer.clone(system)
    system.addForce(mm.MonteCarloBarostat(1 * unit.bar, TEMPERATURE))
    sim = app.Simulation(modeller.topology, system,
                         mm.LangevinMiddleIntegrator(TEMPERATURE, 1 / unit.picosecond, 2 * unit.femtoseconds))
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy()
    sim.step(steps)
    state = sim.context.getState(getPositions=True)
    return state.getPositions(), state.getPeriodicBoxVectors()


def run_leg(name, mol, solvated, schedule, n_samples, interval, eq_steps):
    modeller, system = build(mol, solvated)
    positions, box = modeller.positions, None
    if solvated:
        positions, box = equilibrate_box(modeller, system, eq_steps * 5)
        system.setDefaultPeriodicBoxVectors(*box)
    region = alchemy.AlchemicalRegion(alchemical_atoms=list(range(mol.n_atoms)))
    factory = alchemy.AbsoluteAlchemicalFactory(alchemical_pme_treatment="exact")
    alch_system = factory.create_alchemical_system(system, region)
    alch_state = alchemy.AlchemicalState.from_system(alch_system)

    integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, 1 / unit.picosecond, 2 * unit.femtoseconds)
    platform = mm.Platform.getPlatformByName("CPU")
    props = {} if solvated else {"Threads": "1"}
    context = mm.Context(alch_system, integrator, platform, props)
    context.setPositions(positions)
    if box is not None:
        context.setPeriodicBoxVectors(*box)
    mm.LocalEnergyMinimizer.minimize(context)

    beta = 1 / (KT_KCAL * 4.184)  # 1 / (kT in kJ/mol)
    n_states = len(schedule)
    u_kln = np.zeros((n_states, n_states, n_samples))
    t0 = time.time()
    for k, (elec, steric) in enumerate(schedule):
        alch_state.lambda_electrostatics, alch_state.lambda_sterics = elec, steric
        alch_state.apply_to_context(context)
        integrator.step(eq_steps)
        for n in range(n_samples):
            integrator.step(interval)
            for l, (e2, s2) in enumerate(schedule):  # evaluate this frame at every lambda
                alch_state.lambda_electrostatics, alch_state.lambda_sterics = e2, s2
                alch_state.apply_to_context(context)
                energy = context.getState(getEnergy=True).getPotentialEnergy()
                u_kln[k, l, n] = beta * energy.value_in_unit(unit.kilojoule_per_mole)
            alch_state.lambda_electrostatics, alch_state.lambda_sterics = elec, steric
            alch_state.apply_to_context(context)
        print(f"  {name}: window {k + 1}/{n_states} (elec={elec:.2f}, vdw={steric:.2f}) "
              f"done, {time.time() - t0:.0f} s elapsed", flush=True)

    mbar = MBAR(u_kln, np.full(n_states, n_samples))
    result = mbar.compute_free_energy_differences()
    df = result["Delta_f"][0] * KT_KCAL  # kcal/mol, relative to the fully interacting state
    ddf = result["dDelta_f"][0] * KT_KCAL
    # Overlap between neighbouring windows: below ~0.03 means the ladder needs more rungs
    overlap = np.diag(mbar.compute_overlap()["matrix"], k=1)
    return df, ddf, overlap


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smiles", default="CCO")
    p.add_argument("--samples", type=int, default=150, help="samples per lambda window")
    p.add_argument("--interval", type=int, default=250, help="MD steps between samples (0.5 ps)")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    eq_steps = 5000
    if args.quick:
        args.samples, args.interval, eq_steps = 10, 50, 200
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    mol = make_molecule(args.smiles)
    print(f"Molecule {args.smiles}: {mol.n_atoms} atoms, net charge {mol.total_charge}")
    dg_vac, err_vac, _ = run_leg("vacuum", mol, False, VACUUM_SCHEDULE, args.samples, args.interval, eq_steps)
    dg_wat, err_wat, overlap = run_leg("water", mol, True, WATER_SCHEDULE, args.samples, args.interval, eq_steps)

    dg_hyd = dg_vac[-1] - dg_wat[-1]
    err = np.hypot(err_vac[-1], err_wat[-1])
    report = (f"dG(vacuum leg) = {dg_vac[-1]:+.2f} +/- {err_vac[-1]:.2f} kcal/mol\n"
              f"dG(water leg)  = {dg_wat[-1]:+.2f} +/- {err_wat[-1]:.2f} kcal/mol\n"
              f"Hydration free energy of {args.smiles}: {dg_hyd:+.2f} +/- {err:.2f} kcal/mol "
              f"(MBAR statistical error only)\n"
              f"Smallest neighbour overlap in water: {overlap.min():.3f}\n")
    (out / "hfe_result.txt").write_text(report)
    print(report)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].errorbar(range(len(WATER_SCHEDULE)), dg_wat, yerr=err_wat, marker="o", label="water")
    ax[0].errorbar(range(len(VACUUM_SCHEDULE)), dg_vac, yerr=err_vac, marker="s", label="vacuum")
    ax[0].axvline(4, ls="--", c="grey", lw=0.8)
    ax[0].text(4.2, ax[0].get_ylim()[0], "charges off | vdW switching off", va="bottom")
    ax[0].set(xlabel="lambda window", ylabel="F relative to start (kcal/mol)", title="Free energy along the ladder")
    ax[0].legend()
    ax[1].bar(range(len(overlap)), overlap)
    ax[1].axhline(0.03, ls="--", c="red", lw=0.8)
    ax[1].set(xlabel="window pair (k, k+1)", ylabel="MBAR overlap", title="Phase-space overlap between neighbours")
    fig.tight_layout()
    fig.savefig(out / "hfe.png", dpi=150)


if __name__ == "__main__":
    main()
