# Molecular dynamics and QM/MM for biochemistry: a laptop-sized portfolio

Nine self-contained projects that go from writing an MD integrator by hand to QM/MM spectroscopy of a
molecule in water. Every project answers a real biophysical question, runs on a laptop CPU, has a
`--quick` smoke-test mode, and produces publication-style figures.

[![smoke-tests](../../actions/workflows/smoke-tests.yml/badge.svg)](../../actions/workflows/smoke-tests.yml)

| # | Project | Question | Methods | Laptop time |
|---|---------|----------|---------|-------------|
| 01 | [Lennard-Jones fluid from scratch](01_lennard_jones_from_scratch) | Why does every MD code use velocity Verlet? | NumPy MD, PBC, NVE, g(r) | 1-2 min |
| 02 | [Alanine dipeptide](02_alanine_dipeptide) | Which backbone conformations does a peptide visit, and how does water change that? | OpenMM, AMBER ff14SB, GBn2, Ramachandran free energy | ~30 min |
| 03 | [Protein in water](03_protein_in_water) | Is the villin headpiece stable, and where is it flexible? | Full pipeline: PDBFixer, TIP3P-FB, PME, NVT/NPT, RMSD/RMSF/DSSP | 1-2 h |
| 04 | [Protein-ligand complex](04_protein_ligand) | Is a docked/crystal ligand pose stable? (T4 lysozyme L99A + p-xylene) | OpenFF Sage ligand parameters, NAGL charges, HMR 4 fs, contact analysis | 2-4 h |
| 05 | [Metadynamics](05_metadynamics) | How do we sample states that plain MD never reaches? | Well-tempered metadynamics on phi/psi | 10-20 min |
| 06 | [Hydration free energy](06_hydration_free_energy) | How much does ethanol "like" water? (the core of FEP) | Alchemical decoupling, soft-core, MBAR | 30-60 min |
| 07 | [Membrane peptide](07_membrane_peptide) | How does a transmembrane helix sit in a lipid bilayer? | WALP23 in POPC, Lipid17, membrane barostat | 3-5 h |
| 08 | [QM torsion scan](08_qm_torsion_scan) | Does the force field get a ligand torsion right? | DFT (B3LYP-D3/def2-SVP, PySCF) vs OpenFF | 20-40 min |
| 09 | [QM/MM solvatochromism](09_qmmm_solvatochromism) | Why does acetone absorb bluer in water? | MD sampling + TDDFT with electrostatic embedding | 15-30 min |

Times are for a 4-core laptop CPU. Projects 03, 04 and 07 run 50-100x faster on a GPU
(`--platform CUDA`, e.g. free Google Colab).

## Quick start

```bash
conda env create -f environment.yml        # or: mamba env create -f environment.yml
conda activate md-portfolio
./run_smoke_tests.sh                       # every project in --quick mode, ~20-30 min
cd 02_alanine_dipeptide && python run_md.py
```

Each folder has its own README with the scientific background, what to look at in the results,
and ideas for extensions.

## Learning path

```
01 physics of MD  ->  02 first biomolecule  ->  03 full protein pipeline  ->  04 drug-like ligand
                                     \-> 05 enhanced sampling -> 06 free energies
                                                      07 membranes
08 quantum chemistry for force fields  ->  09 QM/MM
```

## Skills demonstrated

- MD engines and setup: OpenMM, PDBFixer, AMBER/CHARMM-family force fields, OpenFF for small molecules
- Statistical mechanics: ensembles, thermostats/barostats, free energy from histograms, MBAR
- Enhanced sampling (metadynamics) and alchemical free-energy calculations
- Membrane systems, lipid force fields, membrane-specific barostats
- Quantum chemistry (DFT, TDDFT, constrained optimisations) and QM/MM electrostatic embedding
- Trajectory analysis with MDAnalysis; reproducible environments, CLI scripts, CI smoke tests

## License

MIT. Input structures: alanine dipeptide and T4 lysozyme L99A/p-xylene from the OpenMM test suite;
villin headpiece (PDB 1VII) from the RCSB PDB.
