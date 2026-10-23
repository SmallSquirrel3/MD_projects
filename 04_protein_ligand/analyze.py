"""Is the ligand pose stable? Ligand RMSD in the protein frame, distance to the pocket,
and which pocket residues the ligand touches (contact occupancy).

Run after simulate.py:  python analyze.py --out results
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import align, rms
from MDAnalysis.lib.distances import distance_array

LIGAND = "resname TMP"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results")
    p.add_argument("--contact-cutoff", type=float, default=4.5, help="heavy-atom contact distance, A")
    args = p.parse_args()
    out = Path(args.out)

    u = mda.Universe(str(out / "complex.pdb"), str(out / "complex.dcd"))
    ref = mda.Universe(str(out / "complex.pdb"))
    # Superimpose on the protein backbone only: ligand motion is then measured relative to the pocket
    align.AlignTraj(u, ref, select="protein and name CA", in_memory=True).run()
    lig = u.select_atoms(f"{LIGAND} and not element H")
    ref_lig = ref.select_atoms(f"{LIGAND} and not element H")
    pocket = u.select_atoms(f"protein and not element H and around 5.0 ({LIGAND})")  # from frame 0
    pocket_res = sorted(set(pocket.resids))
    print(f"Pocket residues (within 5 A at start): {[int(r) for r in pocket_res]}")

    time_ns, lig_rmsd, com_dist, contacts = [], [], [], []
    pocket_heavy = u.select_atoms("protein and not element H and resid " + " ".join(map(str, pocket_res)))
    for ts in u.trajectory:
        time_ns.append(ts.time / 1000)
        # no refit of the ligand itself: this is the "did it move in the pocket" RMSD
        lig_rmsd.append(rms.rmsd(lig.positions, ref_lig.positions, superposition=False))
        com_dist.append(np.linalg.norm(lig.center_of_mass() - pocket_heavy.center_of_mass()))
        d = distance_array(lig.positions, pocket_heavy.positions)
        touched = set(pocket_heavy[np.any(d < args.contact_cutoff, axis=0)].resids)
        contacts.append([r in touched for r in pocket_res])
    contacts = np.array(contacts)
    occupancy = contacts.mean(axis=0)
    names = {r.resid: f"{r.resname}{r.resid}" for r in u.select_atoms("protein").residues}

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(time_ns, lig_rmsd)
    ax[0].axhline(2.0, ls="--", c="grey", lw=0.8)
    ax[0].set(xlabel="time (ns)", ylabel="ligand RMSD in pocket frame (A)", title="Pose stability (< 2 A = stable)")
    ax[1].plot(time_ns, com_dist)
    ax[1].set(xlabel="time (ns)", ylabel="distance (A)", title="Ligand to pocket centre")
    order = np.argsort(occupancy)[::-1]
    ax[2].bar([names[pocket_res[i]] for i in order], occupancy[order])
    ax[2].set(ylabel="contact occupancy", title=f"Pocket contacts (< {args.contact_cutoff} A)")
    ax[2].tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(out / "ligand_analysis.png", dpi=150)

    lig_rmsd = np.array(lig_rmsd)
    verdict = "stable" if np.percentile(lig_rmsd, 90) < 2.0 else "moves in the pocket: inspect visually"
    summary = (f"frames: {len(time_ns)}; ligand RMSD mean {lig_rmsd.mean():.2f} A, 90th percentile "
               f"{np.percentile(lig_rmsd, 90):.2f} A -> pose {verdict}\n"
               "Top contacts: " + ", ".join(f"{names[pocket_res[i]]} {occupancy[i]:.2f}" for i in order[:6]) + "\n")
    (out / "summary.txt").write_text(summary)
    print(summary + f"Figure: {out / 'ligand_analysis.png'}")


if __name__ == "__main__":
    main()
