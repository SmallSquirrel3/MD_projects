# 08. QM torsion scan vs force field

**Question.** Does the general small-molecule force field (OpenFF Sage 2.2) reproduce the quantum
mechanical energy profile of a key ligand torsion? Example: the methoxy torsion of anisole,
C(H3)-O-C(ar)-C(ar), a fragment of many drugs.

**Methods.**
- QM: relaxed scan, B3LYP-D3(BJ)/def2-SVP in PySCF; at each angle the dihedral is frozen and all
  other coordinates are optimised with geomeTRIC.
- MM: the same dihedral is held by a stiff harmonic restraint in OpenMM and everything else is
  minimised; the restraint energy is excluded from the reported MM energy.
- Both profiles are shifted to their minima and compared (RMSE, barrier height).

```bash
python torsion_scan.py         # 7 points from 0 to 180 deg, 20-40 min
python torsion_scan.py --smiles "CC(=O)Nc1ccccc1" --smarts "[C:1](=O)[N:2][c:3][c:4]"
python torsion_scan.py --quick # HF/STO-3G smoke test
```

**What to look at** (`results/torsion_profile.png`, `.csv`)
- QM: the planar conformation (0 deg) is the minimum; the perpendicular one is a few kcal/mol higher.
- An RMSE above ~1 kcal/mol or a misplaced minimum means the torsion should be refitted
  (e.g. with OpenFF BespokeFit or ForceBalance) before running FEP on this ligand.

**Interview point.** Torsions are the least transferable force-field terms, and bad torsions are a
leading cause of FEP outliers. Checking them against QM is routine in industrial FEP workflows.

**Extensions.** Use a larger basis (def2-TZVP) single points on the optimised geometries;
compare with GFN2-xTB; scan a torsion of a real drug from your field.
