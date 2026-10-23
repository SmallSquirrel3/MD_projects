"""Standard trajectory analysis with MDAnalysis: RMSD, RMSF, radius of gyration,
secondary structure (DSSP) and backbone hydrogen bonds.

Run after simulate.py:  python analyze.py --out results
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
from matplotlib.colors import ListedColormap
from MDAnalysis.analysis import align, rms
from MDAnalysis.analysis.dssp import DSSP
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from MDAnalysis import transformations as trans


def block_error(x, n_blocks=5):
    """Standard error of the mean from block averaging (frames are correlated!)."""
    blocks = np.array_split(np.asarray(x), n_blocks)
    means = [b.mean() for b in blocks]
    return np.std(means, ddof=1) / np.sqrt(n_blocks)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results")
    args = p.parse_args()
    out = Path(args.out)

    u = mda.Universe(str(out / "solvated.pdb"), str(out / "production.dcd"))
    protein = u.select_atoms("protein")
    protein.guess_bonds()  # PDB files written by OpenMM have no CONECT records for standard residues
    # Make the protein whole across periodic boundaries, centre it, then remove global rotation
    u.trajectory.add_transformations(
        trans.unwrap(protein),
        trans.center_in_box(protein, center="mass"),
        trans.wrap(u.select_atoms("not protein"), compound="residues"),
    )
    ref = mda.Universe(str(out / "solvated.pdb"))
    align.AlignTraj(u, ref, select="protein and name CA", in_memory=True).run()
    time_ns = np.array([ts.time for ts in u.trajectory]) / 1000.0

    rmsd = rms.RMSD(u, ref, select="protein and name CA").run().results.rmsd[:, 2]
    calphas = u.select_atoms("protein and name CA")
    rmsf = rms.RMSF(calphas).run().results.rmsf
    rg = np.array([protein.radius_of_gyration() for _ in u.trajectory])
    dssp = DSSP(u).run().results.dssp  # 'H' helix, 'E' strand, '-' loop
    helix_frac = (dssp == "H").mean(axis=1)

    hb = HydrogenBondAnalysis(u, donors_sel="protein and name N", hydrogens_sel="protein and name H",
                              acceptors_sel="protein and name O", d_a_cutoff=3.5)
    hb.run()
    n_hbonds = hb.count_by_time()

    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].plot(time_ns, rmsd)
    ax[0, 0].set(xlabel="time (ns)", ylabel="C-alpha RMSD (A)", title="Drift from the starting structure")
    ax[0, 1].plot(calphas.resids, rmsf, marker="o", ms=3)
    ax[0, 1].set(xlabel="residue", ylabel="RMSF (A)", title="Per-residue flexibility")
    ax[1, 0].plot(time_ns, rg)
    ax[1, 0].set(xlabel="time (ns)", ylabel="Rg (A)", title="Compactness")
    ss_map = {"H": 2, "E": 1, "-": 0}
    ss = np.vectorize(ss_map.get)(dssp)
    cmap = ListedColormap(["#d9d9d9", "#f28e2b", "#4e79a7"])  # loop, strand, helix
    ax[1, 1].imshow(ss.T, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=2, interpolation="nearest",
                    extent=[time_ns[0], time_ns[-1], calphas.resids[0], calphas.resids[-1]])
    ax[1, 1].set(xlabel="time (ns)", ylabel="residue", title="DSSP: blue helix, orange strand, grey loop")
    fig.tight_layout()
    fig.savefig(out / "analysis.png", dpi=150)

    summary = (
        f"frames: {len(time_ns)}, simulated time: {time_ns[-1]:.2f} ns\n"
        f"C-alpha RMSD: {rmsd.mean():.2f} +/- {block_error(rmsd):.2f} A\n"
        f"Radius of gyration: {rg.mean():.2f} +/- {block_error(rg):.2f} A\n"
        f"Helical fraction (DSSP): {helix_frac.mean():.2f}\n"
        f"Backbone H-bonds per frame: {n_hbonds.mean():.1f}\n"
        f"Most flexible residues: {', '.join(str(r) for r in calphas.resids[np.argsort(rmsf)[-3:]])}\n"
    )
    (out / "summary.txt").write_text(summary)
    print(summary + f"Figure: {out / 'analysis.png'}")


if __name__ == "__main__":
    main()
