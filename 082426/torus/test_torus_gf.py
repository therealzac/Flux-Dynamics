"""
Acceptance tests for the torus + lattice-Green's-function engine.
WRITTEN BEFORE THE ENGINE.  Nothing here may be relaxed to make the engine pass.

Three levels, each of which must pass before the next means anything:

  L1  STRUCTURE   the k-space dynamical matrix reproduces the real-space one
  L2  PHYSICS     the known analytic facts about this lattice appear
  L3  VERDICTS    the six configurations whose legality the PBD solver has
                  already established come back with the same verdicts

L3 is the bar that matters.  Ground truth, measured on the PBD engine, fresh
solve from REST, checked in both sweep frames (082426/README.md):

    polar + 1 equatorial          LEGAL    base 6.1e-11 / 7.4e-12
    polar + 2 equatorial same ax  LEGAL    base 9.5e-11 / 9.5e-11
    4 equatorial, no polar        LEGAL    base 2.1e-11
    polar + 2 equatorial diff ax  ILLEGAL  base 2.476e-5 / 1.898e-5
    polar + 3 equatorial          ILLEGAL  base 9.7e-2
    polar + 4 equatorial          ILLEGAL  base 1.5e-1
"""
import numpy as np, itertools, sys

TOL_MATCH = 1e-10          # k-space vs real-space agreement
S = 1/np.sqrt(3)           # integer coords -> world
SHORT = 2/np.sqrt(3)       # <200> rest length
DELTA = SHORT - 1.0        # how far a shortcut must contract

FAILED = []
def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + (("   " + detail) if detail else ""))
    if not ok: FAILED.append(name)

# ----------------------------------------------------------------- L1
def L1_structure(eng):
    print("\nL1  STRUCTURE")
    N = 3
    t = eng.Torus(N)
    Phi_real = t.phi_real()                       # 6N^3 x 6N^3, dense (small N)
    n = Phi_real.shape[0]
    check("real-space Phi is symmetric", np.allclose(Phi_real, Phi_real.T, atol=1e-12))
    check("real-space Phi is PSD",
          np.linalg.eigvalsh(Phi_real).min() > -1e-9,
          f"min eig {np.linalg.eigvalsh(Phi_real).min():.2e}")
    # exactly 3 zero modes: uniform translation. (Any more = extra floppy modes.)
    ev = np.linalg.eigvalsh(Phi_real)
    nz = int((ev < 1e-9).sum())
    check("uniform translation is a zero mode",
          np.allclose(Phi_real @ np.tile([1.,0,0], n//3), 0, atol=1e-10))
    # the k-space blocks must reproduce the full spectrum
    kspec = np.sort(np.concatenate([np.linalg.eigvalsh(t.phi_k(k)) for k in t.kpoints()]))
    check("k-space spectrum == real-space spectrum",
          np.allclose(kspec, np.sort(ev), atol=TOL_MATCH),
          f"max dev {np.abs(kspec-np.sort(ev)).max():.2e}")
    print(f"        ({n} dof, {nz} zero modes)")
    return nz

# ----------------------------------------------------------------- L2
def L2_physics(eng):
    print("\nL2  PHYSICS")
    t = eng.Torus(6)
    # C' = 0 : <111> bonds are blind to tetragonal <100> strain at first order.
    # Apply diag(e,-e,0) and measure the change in every base bond length.
    e = 1e-3
    F = np.diag([1+e, 1-e, 1.0])
    worst_base = t.strain_bond_change(F, kind="base")
    worst_short = t.strain_bond_change(F, kind="short")
    check("base <111> bonds blind to Bain strain (2nd order)",
          worst_base < 3*e**2, f"worst {worst_base:.3e}  vs e^2/3 = {e*e/3:.3e}")
    check("shortcut <200> candidates respond at 1st order",
          worst_short > 0.5*e, f"worst {worst_short:.3e}  vs e = {e:.1e}")
    # MARGINAL STABILITY.  Replaced 2026-08-24 with Zac's explicit consent.
    # The previous test looked for a branch of Phi(k) scaling faster than k^2 as
    # k -> 0.  That encoded a wrong hypothesis: measured exponents are 0,0,0,2,2,2
    # in every direction -- three optical, three acoustic, nothing anomalous.
    # C' = 0 does NOT show up as a fractional power-law branch.  It shows up as
    # EXACT ZERO MODES on a measure-zero set, which a radial scan through generic
    # directions never touches.  The correct statement:
    #
    #     dim ker Phi = 6N - 3, supported exactly on
    #     k_a = 0 and k_b = +/- k_c   -- six <110> lines through the origin
    #
    for N in (3, 4, 5, 6, 8):
        tt = eng.Torus(N)
        nz, wrong = tt.kernel_census()
        check(f"N={N}: dim ker Phi == 6N-3",
              nz == 6*N - 3, f"got {nz}, expected {6*N-3}")
        check(f"N={N}: kernel supported exactly on the six <110> lines",
              wrong == 0, f"{wrong} k-points disagree with the predicate")
    # and the lines really are <110>: pairwise 90 deg inside a coordinate plane,
    # 60 deg between planes -- the angular signature measured in 081626/
    ang = eng.Torus(4).line_angles()
    check("the six lines are <110>, at 90 deg within a plane / 60 deg across",
          ang == {60.0, 90.0}, f"angles present: {sorted(ang)}")

# ----------------------------------------------------------------- L3
CONFIGS = {
    "polar + 1 equatorial            ": ("legal",   ["polar", "eqA1"]),
    "polar + 2 equatorial, same axis  ": ("legal",   ["polar", "eqA1", "eqA2"]),
    "4 equatorial, no polar           ": ("legal",   ["eqA1", "eqA2", "eqB1", "eqB2"]),
    "polar + 2 equatorial, diff axes  ": ("illegal", ["polar", "eqA1", "eqB1"]),
    "polar + 3 equatorial             ": ("illegal", ["polar", "eqA1", "eqA2", "eqB1"]),
    "polar + 4 equatorial             ": ("illegal", ["polar", "eqA1", "eqA2", "eqB1", "eqB2"]),
}

def L3_verdicts(eng, N=8):
    print(f"\nL3  VERDICTS   (torus N={N}, {2*N**3} nodes)")
    t = eng.Torus(N)
    cell = t.central_cell()
    rows = []
    for name, (want, slots) in CONFIGS.items():
        res = t.solve(cell.pairs(slots))
        got = "legal" if res.legal else "illegal"
        rows.append((name, want, got, res))
        check(name, got == want,
              f"want {want:7s} got {got:7s}  base {res.base_resid:.3e}  "
              f"feasible {res.feasibility:.3e}")
    print("\n  configuration                      want     got      base resid   feasibility")
    for name, want, got, r in rows:
        print(f"  {name} {want:8s} {got:8s} {r.base_resid:11.3e}  {r.feasibility:11.3e}")
    # the separation must be unambiguous, not a threshold squeak
    L = [r.base_resid for _, w, _, r in rows if w == "legal"]
    I = [r.base_resid for _, w, _, r in rows if w == "illegal"]
    if L and I:
        check("legal and illegal populations separate by >=100x",
              min(I) > 100*max(L), f"max legal {max(L):.2e}  min illegal {min(I):.2e}")

# ----------------------------------------------------------------- driver
if __name__ == "__main__":
    import torus_gf as eng
    L1_structure(eng)
    L2_physics(eng)
    L3_verdicts(eng)
    print("\n" + ("ALL PASS" if not FAILED else f"{len(FAILED)} FAILED: " + ", ".join(FAILED)))
    sys.exit(1 if FAILED else 0)
