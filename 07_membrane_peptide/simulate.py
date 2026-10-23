"""A transmembrane peptide (WALP23) in a POPC bilayer.

WALP peptides are model transmembrane helices: a Leu-Ala core flanked by Trp "anchors" that like
to sit at the water-lipid interface. A classic question is hydrophobic matching: how much does the
helix tilt to fit its hydrophobic length to the membrane thickness?

Steps: build an ideal alpha helix from sequence -> orient it along z -> insert into POPC
(OpenMM Modeller.addMembrane, AMBER Lipid17 via amber14-all) -> NPT with a membrane barostat.

Run:  python simulate.py              (1 ns, ~30k atoms: 3-5 h on a laptop CPU, minutes on a GPU)
      python simulate.py --quick      (smoke test)
"""
import argparse
import io
from pathlib import Path

import numpy as np
import openmm as mm
import PeptideBuilder
from Bio.PDB import PDBIO
from openmm import app, unit
from pdbfixer import PDBFixer

SEQUENCE = "GWWLALALALALALALALALWWA"  # WALP23


def build_helix(sequence):
    """Ideal alpha helix (phi = -57, psi = -47) with its axis rotated onto z."""
    structure = PeptideBuilder.make_structure(sequence, [-57.0] * (len(sequence) - 1), [-47.0] * (len(sequence) - 1))
    handle = io.StringIO()
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(handle)
    fixer = PDBFixer(pdbfile=io.StringIO(handle.getvalue()))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    xyz = np.array(fixer.positions.value_in_unit(unit.nanometer))
    ca = np.array([i for i, a in enumerate(fixer.topology.atoms()) if a.name == "CA"])
    centre = xyz[ca].mean(axis=0)
    axis = np.linalg.svd(xyz[ca] - centre)[2][0]  # first principal component = helix axis
    # Rodrigues rotation taking `axis` onto z
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(axis, z), np.dot(axis, z)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rot = np.eye(3) + vx + vx @ vx / (1 + c)
    xyz = (xyz - centre) @ rot.T
    return fixer.topology, xyz * unit.nanometer


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ns", type=float, default=1.0)
    p.add_argument("--padding", type=float, default=1.0, help="lipid padding around the peptide, nm")
    p.add_argument("--platform", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    if args.quick:
        args.ns = 0.02
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    topology, positions = build_helix(SEQUENCE)
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")  # amber14-all includes Lipid17
    modeller = app.Modeller(topology, positions)
    print("Inserting into POPC (packing lipids takes a minute or two) ...")
    modeller.addMembrane(ff, lipidType="POPC", membraneCenterZ=0 * unit.nanometer,
                         minimumPadding=args.padding * unit.nanometer, ionicStrength=0.15 * unit.molar)
    n_lipids = sum(1 for r in modeller.topology.residues() if r.name == "POP")
    print(f"{modeller.topology.getNumAtoms()} atoms, {n_lipids} POPC residues")
    with open(out / "system.pdb", "w") as fh:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, fh)

    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
                             constraints=app.HBonds, hydrogenMass=1.5 * unit.amu)
    # Membrane barostat: area (xy) and thickness (z) fluctuate independently, zero surface tension
    system.addForce(mm.MonteCarloMembraneBarostat(1 * unit.bar, 0 * unit.bar * unit.nanometer, 303 * unit.kelvin,
                                                  mm.MonteCarloMembraneBarostat.XYIsotropic,
                                                  mm.MonteCarloMembraneBarostat.ZFree))
    # Keep the peptide backbone in place while lipids and water settle
    restraint = mm.CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", 1000.0)
    for name in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(name)
    for atom in modeller.topology.atoms():
        if atom.residue.chain.index == 0 and atom.name in ("CA", "C", "N"):
            restraint.addParticle(atom.index, modeller.positions[atom.index].value_in_unit(unit.nanometer))
    system.addForce(restraint)

    integrator = mm.LangevinMiddleIntegrator(303 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtoseconds)
    platform = mm.Platform.getPlatformByName(args.platform) if args.platform else None
    sim = app.Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)
    print("Platform:", sim.context.getPlatform().getName())
    sim.minimizeEnergy(maxIterations=500 if args.quick else 0)
    sim.context.setVelocitiesToTemperature(303 * unit.kelvin, 3)
    eq_steps = 200 if args.quick else 50_000  # 200 ps: lipids need longer than proteins
    sim.reporters.append(app.StateDataReporter(str(out / "equilibration.log"), 2500, step=True, time=True,
                                               potentialEnergy=True, temperature=True, volume=True, speed=True))
    for k in (1000.0, 300.0, 100.0, 30.0, 0.0):
        sim.context.setParameter("k", k)
        sim.step(eq_steps // 5)

    n_steps = int(args.ns * 2.5e5)
    sim.reporters = [
        app.DCDReporter(str(out / "production.dcd"), 2500),  # 10 ps
        app.StateDataReporter(str(out / "production.log"), 2500, step=True, time=True, temperature=True,
                              volume=True, speed=True),
    ]
    sim.step(n_steps)
    print(f"Done. Analyse with: python analyze.py --out {out}")


if __name__ == "__main__":
    main()
