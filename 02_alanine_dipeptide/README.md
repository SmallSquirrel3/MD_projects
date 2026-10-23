# 02. Alanine dipeptide: the "hydrogen atom" of protein dynamics

**Question.** Which backbone conformations (phi/psi) does a single residue visit, and how does water change them?

**Methods.** OpenMM, AMBER ff14SB, Langevin dynamics at 300 K, 2 fs with H-bond constraints.
Two runs of 10 ns: vacuum and GBn2 implicit water. Free energy from the histogram,
F(phi, psi) = -kT ln P(phi, psi), analysed with MDAnalysis.

```bash
python run_md.py              # ~30 min on a CPU
python run_md.py --quick      # smoke test
```

**What to look at** (`results/ramachandran_fes.png`)
- Vacuum: the intramolecular H-bonded C7eq state (phi ~ -80, psi ~ +70) dominates.
- Water: the extended beta/PPII region and the right-handed alpha helix region (psi ~ -40) appear,
  because water competes for the H-bond. This is why gas-phase intuition fails for proteins.
- The left-handed region (phi > 0) is essentially never visited: its barrier is too high for plain MD.
  Project 05 fixes this.

**Note.** For 22 atoms a single CPU thread is fastest: parallel overhead dominates (set in the script).

**Extensions.** Compare force fields (amber99sb vs ff14SB vs charmm36); run in explicit TIP3P water.
