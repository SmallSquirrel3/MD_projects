# 04. Protein-ligand complex: is the pose stable?

**System.** T4 lysozyme L99A with p-xylene in its engineered hydrophobic cavity. This is the classic
benchmark for binding free energy methods: a small, buried, apolar pocket with dozens of measured ligands.

**Methods.**
- Protein: AMBER ff14SB; ligand: OpenFF Sage 2.2, partial charges from OpenFF NAGL
  (a graph neural network trained to reproduce AM1-BCC, no AmberTools needed).
- Bond orders are not stored in PDB files, so the ligand is rebuilt from a SMILES template with RDKit.
- Explicit TIP3P + 0.15 M NaCl, PME; hydrogen mass repartitioning allows a 4 fs time step.
- Heavy-atom restraints released over 100 ps; only protein + ligand are written to the trajectory.

```bash
python simulate.py                          # 1 ns, ~25k atoms: 2-4 h on a CPU
python simulate.py --ns 20 --platform CUDA  # Colab GPU
python analyze.py
```

**Analysis** (`results/ligand_analysis.png`)
- Ligand RMSD after superposing the protein backbone (no refit of the ligand): < 2 A means the pose holds.
- Distance between ligand and pocket centres.
- Contact occupancy per pocket residue. Expected cavity lining: Leu84, Val87, Tyr88, Ala99, Val103,
  Val111, Leu118, Leu121, Phe153.

**Interview point.** Pose stability in a few ns is a filter, not proof of binding: an unbinding event
for a nanomolar ligand takes seconds. Affinities need free-energy methods (project 06 shows the machinery).

**Extensions.** Swap in benzene or toluene from a SMILES; dock a new ligand with Vina and test its pose;
compute an MM/GBSA estimate over the trajectory; use ProLIF for interaction fingerprints.
