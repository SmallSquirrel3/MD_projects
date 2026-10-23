"""Full explicit-solvent MD pipeline for a small protein: the villin headpiece subdomain (PDB 1VII).

Pipeline (the same eight steps as in any real project):
  1. get structure  2. fix it (PDBFixer)  3. protonate at pH 7  4. force field (AMBER ff14SB + TIP3P-FB)
  5. solvate + 0.15 M NaCl  6. minimise  7. NVT then NPT equilibration with backbone restraints
  8. unrestrained NPT production

Run:  python simulate.py                       (1 ns production, ~30-60 min on a laptop CPU)
      python simulate.py --ns 0.02 --quick     (smoke test)
      python simulate.py --pdb-id 1L2Y         (any other small protein, downloaded from RCSB)
"""
import argparse
import sys
from pathlib import Path

import openmm as mm
from openmm import app, unit
from pdbfixer import PDBFixer

HERE = Path(__file__).parent


def prepare(pdb_id, ph):
    """Steps 1-3: download (or use the bundled copy), add missing atoms and hydrogens."""
    local = HERE / "data" / f"{pdb_id.lower()}.pdb"
    if local.exists():
        fixer = PDBFixer(filename=str(local))
    else:
        fixer = PDBFixer(pdbid=pdb_id)
    fixer.removeHeterogens(keepWater=False)  # drop crystallisation additives and waters
    fixer.findMissingResidues()
    fixer.missingResidues = {}  # do not model missing loops (fine for small, complete proteins)
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)  # simple pH-based protonation; check His states with PROPKA in real work
    return fixer.topology, fixer.positions


def add_backbone_restraints(system, topology, positions, k_kj):
    """Harmonic restraints that keep the protein in place while water relaxes around it."""
    force = mm.CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    force.addGlobalParameter("k", k_kj)
    for name in ("x0", "y0", "z0"):
        force.addPerParticleParameter(name)
    for atom in topology.atoms():
        if atom.name in ("CA", "C", "N"):
            force.addParticle(atom.index, positions[atom.index].value_in_unit(unit.nanometer))
    system.addForce(force)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdb-id", default="1VII")
    p.add_argument("--ns", type=float, default=1.0, help="production length, ns")
    p.add_argument("--ph", type=float, default=7.0)
    p.add_argument("--padding", type=float, default=1.0, help="water padding around the protein, nm")
    p.add_argument("--platform", default=None, help="CPU, CUDA, OpenCL (default: fastest available)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    eq_steps = 500 if args.quick else 50_000  # 100 ps each for NVT and NPT

    topology, positions = prepare(args.pdb_id, args.ph)
    ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    modeller = app.Modeller(topology, positions)
    modeller.addSolvent(ff, padding=args.padding * unit.nanometer, ionicStrength=0.15 * unit.molar,
                        boxShape="dodecahedron")  # dodecahedron: ~30% less water than a cube
    n_res = sum(1 for r in modeller.topology.residues() if r.name not in ("HOH", "NA", "CL"))
    print(f"{args.pdb_id}: {n_res} residues, {modeller.topology.getNumAtoms()} atoms after solvation")
    with open(out / "solvated.pdb", "w") as fh:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, fh)

    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
                             constraints=app.HBonds)
    add_backbone_restraints(system, modeller.topology, modeller.positions, 1000.0)
    barostat = mm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin)
    barostat.setFrequency(0)  # switched off during NVT
    system.addForce(barostat)

    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)
    integrator.setRandomNumberSeed(args.seed)
    platform = mm.Platform.getPlatformByName(args.platform) if args.platform else None
    sim = app.Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)
    print("Platform:", sim.context.getPlatform().getName())

    print("Step 6: energy minimisation")
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()
    sim.minimizeEnergy(maxIterations=0 if not args.quick else 200)
    e1 = sim.context.getState(getEnergy=True).getPotentialEnergy()
    print(f"  potential energy {e0.value_in_unit(unit.kilojoule_per_mole):.0f} -> "
          f"{e1.value_in_unit(unit.kilojoule_per_mole):.0f} kJ/mol")

    log = app.StateDataReporter(str(out / "equilibration.log"), 1000, step=True, time=True, potentialEnergy=True,
                                temperature=True, density=True, speed=True)
    sim.reporters.append(log)
    print("Step 7a: NVT heating with backbone restraints")
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, args.seed)
    sim.step(eq_steps)
    print("Step 7b: NPT with restraints released gradually")
    barostat.setFrequency(25)
    sim.context.reinitialize(preserveState=True)
    for k in (1000.0, 300.0, 100.0, 30.0, 0.0):
        sim.context.setParameter("k", k)
        sim.step(eq_steps // 5)

    print(f"Step 8: production, {args.ns} ns")
    sim.reporters = [
        app.DCDReporter(str(out / "production.dcd"), 5000),  # a frame every 10 ps
        app.StateDataReporter(str(out / "production.log"), 5000, step=True, time=True, potentialEnergy=True,
                              temperature=True, density=True, speed=True),
        app.StateDataReporter(sys.stdout, 50_000, step=True, time=True, temperature=True, speed=True,
                              remainingTime=True, totalSteps=int(args.ns * 5e5)),
        app.CheckpointReporter(str(out / "production.chk"), 50_000),
    ]
    sim.step(int(args.ns * 5e5))
    state = sim.context.getState(getPositions=True, enforcePeriodicBox=True)
    with open(out / "final.pdb", "w") as fh:
        app.PDBFile.writeFile(sim.topology, state.getPositions(), fh)
    print(f"Done. Trajectory: {out / 'production.dcd'}; analyse with: python analyze.py --out {out}")


if __name__ == "__main__":
    main()
