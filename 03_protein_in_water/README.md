# 03. A small protein in explicit water: the full pipeline

**Question.** Is the villin headpiece subdomain (HP35, PDB 1VII, 36 residues, a famous fast folder)
stable in the AMBER ff14SB force field, and which parts are flexible?

**Pipeline** (the same eight steps as in any real project)
1. Structure: bundled `data/1vii.pdb` (NMR model 1) or any PDB ID via `--pdb-id`.
2. Repair with PDBFixer (missing atoms, non-standard residues).
3. Protonation at pH 7 (check His states with PROPKA in real work).
4. Force field: AMBER ff14SB + TIP3P-FB water.
5. Truncated dodecahedron box, 1 nm padding, 0.15 M NaCl.
6. Energy minimisation.
7. 100 ps NVT with backbone restraints, then 100 ps NPT with restraints released gradually.
8. Unrestrained NPT production with PME, 2 fs, Langevin thermostat, Monte Carlo barostat.

```bash
python simulate.py             # 1 ns, ~6k atoms: 1-2 h on a laptop CPU, a few minutes on a GPU
python simulate.py --ns 50 --platform CUDA   # a meaningful run on Colab
python analyze.py              # RMSD, RMSF, Rg, DSSP, backbone H-bonds, with block-averaged errors
```

**What to look at** (`results/analysis.png`, `results/summary.txt`)
- C-alpha RMSD reaches a plateau around 1.5-3 A; a steady climb means unfolding or drift.
- RMSF peaks at the termini and the inter-helix loops.
- DSSP keeps the three helices; helix fraction ~0.55-0.6.
- Errors come from block averaging: consecutive frames are correlated.

**Extensions.** Run three replicas with different seeds; mutate a core Phe to Ala and compare
stability; run at 360 K and watch the first helix fray.
