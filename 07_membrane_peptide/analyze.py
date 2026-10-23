"""Membrane analysis: area per lipid, bilayer thickness, peptide tilt and Trp anchoring.

Reference values for POPC at 303 K: area per lipid ~0.64-0.68 nm^2, P-P thickness ~3.8-4.0 nm.
They converge only after tens of ns, so short laptop runs show the trend, not the final number.

Run after simulate.py:  python analyze.py --out results
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from MDAnalysis import transformations as trans


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results")
    args = p.parse_args()
    out = Path(args.out)

    u = mda.Universe(str(out / "system.pdb"), str(out / "production.dcd"))
    # The bilayer is built around z = 0, so periodic wrapping puts one leaflet at the bottom of the box
    # and the other at the top. Put the peptide (= membrane centre) in the middle of the box and rewrap.
    u.trajectory.add_transformations(trans.center_in_box(u.select_atoms("protein and name CA"), wrap=False),
                                     trans.wrap(u.atoms, compound="residues"))
    phosphates = u.select_atoms("resname POP and name P")
    n_lipids = len(phosphates)
    ca = u.select_atoms("protein and name CA")
    trp_ring = u.select_atoms("protein and resname TRP and name NE1")

    rows = []
    for ts in u.trajectory:
        mid = phosphates.positions[:, 2].mean()
        upper = phosphates.positions[:, 2] > mid
        apl = ts.dimensions[0] * ts.dimensions[1] / (n_lipids / 2) / 100  # A^2 -> nm^2
        thickness = (phosphates.positions[upper, 2].mean() - phosphates.positions[~upper, 2].mean()) / 10
        centred = ca.positions - ca.positions.mean(axis=0)
        axis = np.linalg.svd(centred)[2][0]
        tilt = np.degrees(np.arccos(abs(axis[2])))
        # How deep do the Trp indole N-H groups sit, relative to the nearest phosphate plane?
        plane_up, plane_low = phosphates.positions[upper, 2].mean(), phosphates.positions[~upper, 2].mean()
        trp_depth = np.mean([min(abs(z - plane_up), abs(z - plane_low)) for z in trp_ring.positions[:, 2]]) / 10
        rows.append((ts.time / 1000, apl, thickness, tilt, trp_depth))
    t, apl, thick, tilt, trp = np.array(rows).T

    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for a, y, label, ref in zip(ax, (apl, thick, tilt, trp),
                                ("area per lipid (nm$^2$)", "P-P thickness (nm)", "helix tilt (deg)",
                                 "Trp NE1 distance to P plane (nm)"),
                                ((0.64, 0.68), (3.8, 4.0), None, None)):
        a.plot(t, y, marker="o", ms=3)
        if ref:
            a.axhspan(*ref, color="green", alpha=0.15, label="experiment")
            a.legend()
        a.set(xlabel="time (ns)", ylabel=label)
    fig.tight_layout()
    fig.savefig(out / "membrane_analysis.png", dpi=150)

    summary = (f"{n_lipids} POPC, {len(t)} frames\n"
               f"area per lipid {apl.mean():.3f} nm^2 (exp. 0.64-0.68)\n"
               f"P-P thickness {thick.mean():.2f} nm (exp. 3.8-4.0)\n"
               f"WALP23 tilt {tilt.mean():.1f} deg; Trp NE1 within {trp.mean():.2f} nm of the phosphate plane\n")
    (out / "summary.txt").write_text(summary)
    print(summary + f"Figure: {out / 'membrane_analysis.png'}")


if __name__ == "__main__":
    main()
