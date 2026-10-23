# 05. Well-tempered metadynamics: crossing barriers on purpose

**Question.** In project 02 the left-handed region (phi > 0) was never visited. What is its free
energy, and how do we sample states that plain MD cannot reach?

**Method.** Well-tempered metadynamics on the two backbone dihedrals (phi, psi) of alanine dipeptide
in GBn2 implicit water, using OpenMM's built-in `Metadynamics` class. A Gaussian of 1 kJ/mol and
width 0.35 rad is added every ps; the bias factor 6 makes the hills shrink as the surface fills, so the
bias converges instead of overfilling.

```bash
python metad.py            # 5 ns, 10-20 min on a CPU
python metad.py --quick
```

**What to look at** (`results/metad_fes.png`)
- The free-energy surface now includes the alpha_L / C7ax basin at phi ~ +60.
- phi jumps between the two halves of the map many times (in plain MD: zero times).
- Convergence panel: the difference F(phi > 0) - F(phi < 0) should flatten. A 1 ns test run gave
  ~1-3 kcal/mol fluctuating; 5 ns is much flatter. If it keeps drifting, run longer.

**Interview point.** Metadynamics only accelerates the chosen collective variables. If a slow motion
is missing from the CVs (here: nothing serious; in proteins: often), the map looks smooth but is wrong.
Signs: hysteresis and a free-energy difference that never settles.

**Extensions.** Run the same in vacuum; bias only phi and check that psi is still sampled;
reweight to get distributions of unbiased observables; reproduce with PLUMED.
