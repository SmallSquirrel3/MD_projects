# 09. QM/MM with electrostatic embedding: solvatochromism of acetone

**Question.** Why does the n -> pi* absorption of acetone shift by about +0.2 eV (blue shift) from
gas to water? This is the textbook test of QM/MM and a stepping stone to enzymes and photoreceptors.

**Methods.**
1. Classical MD of acetone in TIP3P water (OpenFF Sage 2.2 + OpenMM), 100 ps equilibration,
   then a snapshot every 10 ps.
2. For every snapshot: TDDFT (Tamm-Dancoff, CAM-B3LYP/6-31+G*) on acetone twice, alone and with all
   water molecules within 12 A included as point charges in the QM Hamiltonian
   (`pyscf.qmmm.mm_charge`, electrostatic embedding). Same geometry, so the difference is purely the
   effect of the environment.
3. Average the shift over snapshots and report the standard error.

```bash
python qmmm_spectra.py          # 20 snapshots, 15-30 min
python qmmm_spectra.py --quick  # smoke test
```

**What to look at** (`results/solvatochromism.png`, `results/result.txt`)
- A blue shift of the n -> pi* excitation (positive shift), of the right order of magnitude (~0.1-0.3 eV).
- A broad distribution: every snapshot has a different hydrogen-bond pattern, which is the origin of
  inhomogeneous broadening of spectra.
- The dipole moment grows from ~3 D in gas (experiment: 2.9 D; the --quick STO-3G run gives less) to a larger value in water: the electrons polarise.

**Limitations worth stating.** Point charges cannot polarise back (no mutual polarisation), there is
no Pauli repulsion from the solvent, and fixed TIP3P charges are already "pre-polarised". Polarisable
embedding or a larger QM region (first-shell waters in QM) improve the agreement.

**Bridge to enzymes.** Replace "acetone in water" with "substrate in an active site", and the excitation
energy with a reaction energy along a coordinate: that is QM/MM enzymology (with DFTB or DFT
and umbrella sampling for the free-energy barrier).

**Extensions.** Put the 2-3 waters H-bonded to the carbonyl into the QM region; compare with an
implicit solvent model (PCM); do the same for a chromophore in a protein pocket.
