"""Protein-ligand MD: T4 lysozyme L99A with p-xylene bound in its engineered hydrophobic cavity.

T4L L99A is the classic benchmark for binding free energies: a small rigid pocket, dozens of
measured ligands. Here we ask the everyday drug-discovery question: is the bound pose stable?

  * protein: AMBER ff14SB; ligand: OpenFF Sage 2.2 with AM1-BCC-like charges from OpenFF NAGL
  * solvent: explicit TIP3P + 0.15 M NaCl (default) or GBn2 implicit water
    (on a CPU, OpenMM's GBn2 is actually slower than PME for a protein this size)
  * 4 fs time step thanks to hydrogen mass repartitioning

Run:  python simulate.py                    (1 ns, ~25k atoms: 2-4 h on a laptop CPU, minutes on a GPU)
      python simulate.py --ns 10 --platform CUDA   (what you would run on Google Colab)
      python simulate.py --quick            (smoke test)
"""
import argparse
import time
from pathlib import Path

import openmm as mm
from openff.toolkit import Molecule
from openmm import app, unit
from openmmforcefields.generators import SystemGenerator
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = Path(__file__).parent
LIGAND_RESNAME, LIGAND_SMILES = "TMP", "Cc1ccc(C)cc1"  # p-xylene


def load_ligand(pdb_path):
    """PDB files carry no bond orders, so we take them from a SMILES template."""
    lines = [l for l in open(pdb_path) if l.startswith(("ATOM", "HETATM")) and l[17:20] == LIGAND_RESNAME]
    rd = Chem.MolFromPDBBlock("".join(lines), removeHs=False, proximityBonding=True)
    rd = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(LIGAND_SMILES), rd)
    Chem.SanitizeMol(rd)
    ligand = Molecule.from_rdkit(rd, allow_undefined_stereo=True)
    try:  # graph-network AM1-BCC charges: no AmberTools needed, works on any OS
        ligand.assign_partial_charges("openff-gnn-am1bcc-1.0.0.pt")
    except Exception:
        ligand.assign_partial_charges("am1bcc")  # needs AmberTools (sqm) in PATH
    ligand.name = LIGAND_RESNAME
    return ligand, rd.GetConformer().GetPositions() * 0.1  # Angstrom -> nm


def build(args):
    pdb_path = HERE / "data" / "t4l_l99a_pxylene.pdb"
    pdb = app.PDBFile(str(pdb_path))
    ligand, lig_xyz = load_ligand(pdb_path)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.delete([r for r in modeller.topology.residues() if r.name == LIGAND_RESNAME])
    lig_top = ligand.to_topology().to_openmm()
    for residue in lig_top.residues():
        residue.name = LIGAND_RESNAME
    modeller.add(lig_top, lig_xyz * unit.nanometer)

    explicit = args.solvent == "explicit"
    generator = SystemGenerator(
        forcefields=["amber/ff14SB.xml", "amber/tip3p_standard.xml" if explicit else "implicit/gbn2.xml"],
        small_molecule_forcefield="openff-2.2.0",
        molecules=[ligand],
        forcefield_kwargs={"constraints": app.HBonds, "hydrogenMass": 1.5 * unit.amu},  # HMR -> 4 fs step
        periodic_forcefield_kwargs={"nonbondedMethod": app.PME, "nonbondedCutoff": 1.0 * unit.nanometer},
        nonperiodic_forcefield_kwargs={"nonbondedMethod": app.CutoffNonPeriodic, "nonbondedCutoff": 2.0 * unit.nanometer},
        barostat=mm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin) if explicit else None,
        cache=str(HERE / "data" / "ligand_params.json"),
    )
    if explicit:
        modeller.addSolvent(generator.forcefield, padding=1.0 * unit.nanometer, ionicStrength=0.15 * unit.molar,
                            boxShape="dodecahedron")
    system = generator.create_system(modeller.topology)
    return modeller, system


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--solvent", choices=["implicit", "explicit"], default="explicit")
    p.add_argument("--ns", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--platform", default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    if args.quick:
        args.ns = 0.01
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    t0 = time.time()

    modeller, system = build(args)
    print(f"System: {modeller.topology.getNumAtoms()} atoms ({args.solvent} solvent), built in {time.time() - t0:.0f} s")

    # Restrain heavy atoms of protein and ligand during equilibration
    restraint = mm.CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    restraint.addGlobalParameter("k", 500.0)
    for name in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(name)
    solute = [a for a in modeller.topology.atoms() if a.residue.name not in ("HOH", "NA", "CL")]
    for atom in solute:
        if atom.element.symbol != "H":
            restraint.addParticle(atom.index, modeller.positions[atom.index].value_in_unit(unit.nanometer))
    system.addForce(restraint)

    dt = 4 * unit.femtoseconds
    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, dt)
    integrator.setRandomNumberSeed(args.seed)
    platform = mm.Platform.getPlatformByName(args.platform) if args.platform else None
    sim = app.Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)
    print("Platform:", sim.context.getPlatform().getName())
    sim.minimizeEnergy(maxIterations=200 if args.quick else 0)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin, args.seed)
    eq_steps = 250 if args.quick else 25_000  # 100 ps
    for k in (500.0, 100.0, 10.0, 0.0):
        sim.context.setParameter("k", k)
        sim.step(eq_steps // 4)

    # Save only protein + ligand: water would make the trajectory 10x bigger
    subset = [a.index for a in solute]
    sub_top = app.Modeller(modeller.topology, modeller.positions)
    sub_top.delete([r for r in sub_top.topology.residues() if r.name in ("HOH", "NA", "CL")])
    state = sim.context.getState(getPositions=True)
    with open(out / "complex.pdb", "w") as fh:
        app.PDBFile.writeFile(sub_top.topology, state.getPositions(asNumpy=True)[subset], fh)

    n_steps = int(args.ns * 1e6 / 4)
    sim.reporters.append(app.DCDReporter(str(out / "complex.dcd"), 2500, atomSubset=subset))  # every 10 ps
    sim.reporters.append(app.StateDataReporter(str(out / "production.log"), 2500, step=True, time=True,
                                               potentialEnergy=True, temperature=True, speed=True))
    print(f"Production: {args.ns} ns with a 4 fs step (hydrogen mass repartitioning)")
    sim.step(n_steps)
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Analyse with: python analyze.py --out {out}")


if __name__ == "__main__":
    main()
