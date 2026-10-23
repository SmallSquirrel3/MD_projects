# 06. Alchemical hydration free energy (the engine of FEP)

**Question.** What is the free energy of transferring ethanol from gas to water? Experiment: -5.0 kcal/mol.

**Method.** We never move the molecule out of the water. Instead we make it vanish:
1. Water leg: switch off the solute's charges (5 windows), then its van der Waals interactions with
   soft-core potentials (10 windows).
2. Vacuum leg: switch off the charges in vacuum (the intramolecular part of the electrostatics).
3. dG_hyd = dG_vacuum - dG_water. Free energy is a state function, so the unphysical path is allowed.

Each frame is evaluated at every lambda and all data are combined with MBAR (pymbar).
Alchemical systems are built by `openmmtools.alchemy`; the ligand uses OpenFF Sage 2.2.

```bash
python hfe.py                          # ethanol, 30-60 min
python hfe.py --smiles "c1ccccc1O"     # phenol, experiment -6.6
python hfe.py --quick                  # smoke test (numbers meaningless)
```

**What to look at** (`results/hfe.png`, `results/hfe_result.txt`)
- The free-energy profile along the lambda ladder for both legs.
- Overlap between neighbouring windows: below ~0.03 the estimate is unreliable, add windows there.
- OpenFF Sage typically gives hydration free energies within ~1 kcal/mol of experiment for small
  neutral molecules; the statistical error here is a few tenths of a kcal/mol.

**Interview point.** The same thermodynamic cycle with the ligand in the binding site instead of in
water gives absolute binding free energies; morphing ligand A into B gives relative binding free
energies (RBFE), which is what FEP+ and OpenFE do in drug discovery.

**Extensions.** Run a series (methanol, ethanol, propanol) and compare with experiment (FreeSolv
database); test the effect of the number of windows on the error; compute the TI estimate and compare.
