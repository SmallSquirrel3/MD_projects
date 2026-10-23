#!/usr/bin/env bash
# Runs every project in --quick mode (about 20-30 min in total on a laptop). Used by CI.
set -euo pipefail
cd "$(dirname "$0")"
OUT=$(mktemp -d)
echo "Writing smoke-test results to $OUT"
(cd 01_lennard_jones_from_scratch && python lj_md.py --quick --out "$OUT/01")
(cd 02_alanine_dipeptide && python run_md.py --quick --out "$OUT/02")
(cd 03_protein_in_water && python simulate.py --quick --ns 0.02 --out "$OUT/03" && python analyze.py --out "$OUT/03")
(cd 04_protein_ligand && python simulate.py --quick --out "$OUT/04" && python analyze.py --out "$OUT/04")
(cd 05_metadynamics && python metad.py --quick --out "$OUT/05")
(cd 06_hydration_free_energy && python hfe.py --quick --out "$OUT/06")
(cd 07_membrane_peptide && python simulate.py --quick --out "$OUT/07" && python analyze.py --out "$OUT/07")
(cd 08_qm_torsion_scan && python torsion_scan.py --quick --out "$OUT/08")
(cd 09_qmmm_solvatochromism && python qmmm_spectra.py --quick --out "$OUT/09")
echo "All smoke tests passed."
