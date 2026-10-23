"""Molecular dynamics of a Lennard-Jones fluid (liquid argon), written from scratch in NumPy.

What this script demonstrates:
  * reduced Lennard-Jones units, periodic boundary conditions, minimum image convention
  * velocity Verlet vs. explicit Euler: energy conservation (NVE)
  * temperature from kinetic energy, radial distribution function g(r)

Run:  python lj_md.py            (about 1-2 minutes on a laptop)
      python lj_md.py --quick    (smoke test, a few seconds)
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Argon in reduced units: sigma = 3.405 A, epsilon/kB = 119.8 K, mass = 39.95 u
SIGMA_A, EPS_K = 3.405, 119.8


def fcc_lattice(n_cells, box):
    """Place 4 * n_cells**3 atoms on an FCC lattice (a crystal is a clash-free start)."""
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cells = np.array([[i, j, k] for i in range(n_cells) for j in range(n_cells) for k in range(n_cells)])
    pos = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3)
    return pos * box / n_cells


def forces_and_energy(pos, box, rc):
    """Truncated and shifted LJ potential. O(N^2) and fully vectorised: fine for N < 1000."""
    d = pos[:, None, :] - pos[None, :, :]
    d -= box * np.round(d / box)  # minimum image convention (the Pac-Man trick)
    r2 = np.sum(d * d, axis=-1)
    np.fill_diagonal(r2, np.inf)
    mask = r2 < rc * rc
    inv6 = np.where(mask, 1.0 / r2**3, 0.0)
    shift = 4.0 * (rc**-12 - rc**-6)
    pair_e = np.where(mask, 4.0 * (inv6 * inv6 - inv6) - shift, 0.0)
    # F_ij = 24 (2 r^-12 - r^-6) / r^2 * d_ij
    fmag = np.where(mask, 24.0 * (2.0 * inv6 * inv6 - inv6) / r2, 0.0)
    f = np.sum(fmag[:, :, None] * d, axis=1)
    return f, 0.5 * pair_e.sum()


def run(pos, vel, box, dt, n_steps, rc, integrator, rdf_every=0, rdf_bins=None):
    f, epot = forces_and_energy(pos, box, rc)
    energies, temps, rdf_hist = [], [], None
    n = len(pos)
    if rdf_every:
        rdf_hist = np.zeros(len(rdf_bins) - 1)
    for step in range(n_steps):
        if integrator == "verlet":
            vel += 0.5 * dt * f
            pos += dt * vel
            pos %= box
            f, epot = forces_and_energy(pos, box, rc)
            vel += 0.5 * dt * f
        else:  # explicit Euler: the "wrong" way, kept for comparison
            pos_new = (pos + dt * vel) % box
            vel += dt * f
            pos = pos_new
            f, epot = forces_and_energy(pos, box, rc)
        ekin = 0.5 * np.sum(vel**2)
        energies.append((ekin, epot, ekin + epot))
        temps.append(2.0 * ekin / (3 * n - 3))
        if rdf_every and step % rdf_every == 0:
            d = pos[:, None, :] - pos[None, :, :]
            d -= box * np.round(d / box)
            r = np.sqrt(np.sum(d * d, axis=-1))[np.triu_indices(n, 1)]
            rdf_hist += np.histogram(r, bins=rdf_bins)[0]
    return pos, vel, np.array(energies), np.array(temps), rdf_hist


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", type=int, default=4, help="FCC cells per edge (N = 4*cells^3)")
    p.add_argument("--density", type=float, default=0.8442, help="reduced density (liquid argon near triple point)")
    p.add_argument("--temperature", type=float, default=0.728, help="reduced temperature T* = kT/epsilon")
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--quick", action="store_true", help="tiny run for testing")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    if args.quick:
        args.cells, args.steps = 3, 300

    rng = np.random.default_rng(42)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    n = 4 * args.cells**3
    box = (n / args.density) ** (1 / 3)
    rc = min(2.5, box / 2)
    print(f"N = {n} atoms, box = {box:.3f} sigma ({box * SIGMA_A / 10:.2f} nm), T* = {args.temperature}")

    pos0 = fcc_lattice(args.cells, box)
    vel0 = rng.normal(0, np.sqrt(args.temperature), size=pos0.shape)
    vel0 -= vel0.mean(axis=0)  # no net drift of the whole box
    # Melt the crystal and equilibrate with crude velocity rescaling (a "poor man's thermostat")
    pos, vel = pos0.copy(), vel0.copy()
    for _ in range(args.steps // 400 or 1):
        pos, vel, _, temps, _ = run(pos, vel, box, args.dt, 200, rc, "verlet")
        vel *= np.sqrt(args.temperature / temps[-50:].mean())

    # Production in NVE: compare integrators from the same starting point
    bins = np.linspace(0.0, box / 2, 151)
    _, _, e_verlet, t_verlet, hist = run(pos.copy(), vel.copy(), box, args.dt, args.steps, rc, "verlet", 10, bins)
    with np.errstate(over="ignore", invalid="ignore"):  # Euler is expected to blow up
        _, _, e_euler, _, _ = run(pos.copy(), vel.copy(), box, args.dt, args.steps, rc, "euler")

    # g(r): histogram normalised by the ideal-gas count in each shell
    shell = 4 / 3 * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)
    n_frames = len(range(0, args.steps, 10))
    g = hist / (n_frames * 0.5 * n * (n / box**3) * shell)
    r_mid = 0.5 * (bins[1:] + bins[:-1])

    time = np.arange(args.steps) * args.dt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    # |E(t) - E(0)| on a log scale: Verlet wobbles around zero, Euler runs away exponentially
    ax[0].semilogy(time, np.abs(e_verlet[:, 2] - e_verlet[0, 2]) / n + 1e-12, label="velocity Verlet")
    ax[0].semilogy(time, np.abs(e_euler[:, 2] - e_euler[0, 2]) / n + 1e-12, label="Euler", alpha=0.8)
    ax[0].set(xlabel="time (reduced units)", ylabel="|E(t) - E(0)| per atom", title="Energy conservation (NVE)")
    ax[0].legend()
    ax[1].plot(time, t_verlet * EPS_K)
    ax[1].set(xlabel="time (reduced units)", ylabel="T (K)", title="Instantaneous temperature")
    ax[2].plot(r_mid * SIGMA_A, g)
    ax[2].axhline(1, color="grey", lw=0.8)
    ax[2].set(xlabel="r (A)", ylabel="g(r)", title="Radial distribution function")
    fig.tight_layout()
    fig.savefig(out / "lj_summary.png", dpi=150)

    drift = lambda e: (e[-1, 2] - e[0, 2]) / n
    print(f"Energy drift per atom: Verlet {drift(e_verlet):+.2e}, Euler {drift(e_euler):+.2e}")
    print(f"Mean T = {t_verlet.mean() * EPS_K:.1f} K; first g(r) peak at {r_mid[np.argmax(g)] * SIGMA_A:.2f} A")
    print(f"Figure saved to {out / 'lj_summary.png'}")


if __name__ == "__main__":
    main()
