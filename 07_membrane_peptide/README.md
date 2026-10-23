# 07. A transmembrane helix in a lipid bilayer

**Question.** How does the model transmembrane peptide WALP23 (GWW(LA)8LWWA) sit in a POPC bilayer?
WALP peptides probe hydrophobic matching: the Trp residues anchor at the water-lipid interface and
the helix tilts if its hydrophobic length does not match the membrane thickness.

**Methods.**
- An ideal alpha helix is built from sequence (PeptideBuilder), protonated with PDBFixer and
  rotated so its axis lies along z.
- `Modeller.addMembrane` packs POPC around it (AMBER Lipid17 parameters, part of amber14-all),
  adds TIP3P water and 0.15 M NaCl.
- `MonteCarloMembraneBarostat`: area and thickness fluctuate independently at zero surface tension.
- Backbone restraints released over 200 ps (lipids relax slower than proteins), 4 fs with HMR, 303 K.

```bash
python simulate.py                         # 1 ns, ~30k atoms: 3-5 h on a CPU
python simulate.py --ns 50 --platform CUDA # a meaningful run on a GPU
python analyze.py
```

**Analysis** (`results/membrane_analysis.png`)
- Area per lipid (POPC experiment: 0.64-0.68 nm^2) and phosphate-phosphate thickness (3.8-4.0 nm).
- Helix tilt with respect to the membrane normal (a modest tilt; published simulations of WALP23 in PC bilayers report roughly 5-30 deg depending on lipid and force field).
- Distance of the Trp indole N-H from the phosphate plane: anchoring at the interface.

**Honest caveat.** Lipid properties converge over tens of nanoseconds. A 1 ns laptop run shows the
workflow and the trends; numbers for a paper need 100+ ns. Say this in the README of your own runs too.

**Extensions.** Replace POPC with DMPC (thinner) or POPE and compare the tilt; build the membrane
with CHARMM-GUI and CHARMM36; coarse-grain with Martini 3 and run microseconds.
