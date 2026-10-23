# 01. Lennard-Jones fluid from scratch

**Question.** What actually happens inside an MD engine, and why does everybody use velocity Verlet?

**What the script does.** 256 argon atoms (reduced Lennard-Jones units) start on an FCC lattice,
melt, are brought to 87 K with crude velocity rescaling, and are then propagated in NVE twice from
the same state: once with velocity Verlet, once with explicit Euler. Only NumPy is used:
forces, minimum-image periodic boundaries and the radial distribution function are all written by hand.

```bash
python lj_md.py            # 1-2 min
python lj_md.py --quick    # seconds
```

**Expected result** (`results/lj_summary.png`)
- Verlet: total energy wobbles around a constant (drift ~1e-5 per atom over 4000 steps).
- Euler: energy explodes exponentially (the integrator is not symplectic).
- g(r): first peak at ~3.7 A, typical of a simple liquid; g(r) -> 1 at long range.

**What to say in an interview.** Verlet is time-reversible and symplectic, so it conserves a
"shadow" Hamiltonian; errors stay bounded instead of accumulating. The time step is limited by the
fastest motion in the system.

**Extensions.** Add a Langevin thermostat; compute the diffusion coefficient from the mean-squared
displacement; replace the O(N^2) loop with a cell list and measure the speed-up.
