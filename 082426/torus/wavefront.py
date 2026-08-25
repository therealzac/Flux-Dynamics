"""
THE WAVEFRONT, DERIVED.

The BFS cone assumes causality travels one <111> hop per tick, which makes the
front a CUBE and c faster by exactly sqrt3 along <111> than along <100>. That is
a modelling choice, not a result.

The real front is the RAY (group-velocity) surface of the lattice's own dynamical
matrix: for each wavevector k, omega(k) comes from Phi(k), and energy travels at
v_g = grad_k omega. The front at time t is {v_g(k) t}. Its outer envelope is how
far a disturbance has actually got, in every direction.
"""
import numpy as np, math, itertools
from torus_gf import Torus
np.set_printoptions(precision=4, suppress=True)
t = Torus(8)                      # only phi_k is used; N sets nothing here

def omega(k):
    """acoustic + optical branch frequencies at wavevector k"""
    w = np.linalg.eigvalsh(t.phi_k(np.asarray(k, float)))
    w = np.clip(w, 0, None)
    return np.sqrt(w)

def group_velocity(k, h=1e-5):
    """v_g = grad_k omega, per branch, by central difference"""
    w0 = omega(k)
    V = np.zeros((6, 3))
    for a in range(3):
        e = np.zeros(3); e[a] = h
        wp, wm = omega(np.asarray(k)+e), omega(np.asarray(k)-e)
        V[:, a] = (wp - wm) / (2*h)
    return w0, V

print("PHASE speed by direction, long-wavelength limit (|k| = 1e-3)\n")
print(f"{'direction':>10} {'branch speeds (omega/|k|)':>44}")
DIRS = {'<100>':(1,0,0), '<110>':(1,1,0), '<111>':(1,1,1), '<210>':(2,1,0), '<211>':(2,1,1)}
kmag = 1e-3
speeds = {}
for nm, d in DIRS.items():
    n = np.array(d, float); n /= np.linalg.norm(n)
    w = omega(n*kmag)/kmag
    speeds[nm] = w
    print(f"{nm:>10}   " + "  ".join(f"{x:8.4f}" for x in w))
print("\n  the three LOWEST branches are acoustic; the front is set by the FASTEST.")
vmax = {nm: w.max() for nm, w in speeds.items()}
base = vmax['<100>']
print(f"\n{'direction':>10} {'fastest':>10} {'relative to <100>':>19}")
for nm in DIRS:
    print(f"{nm:>10} {vmax[nm]:10.4f} {vmax[nm]/base:19.4f}")
print(f"\n  BFS cone would give <111>/<100> = sqrt3 = {math.sqrt(3):.4f}")
print(f"  sqrt(7/3) = {math.sqrt(7/3):.4f}   (Christoffel prediction for C11=C12=C44)")

# ---- does any branch have ZERO speed?  That is the C' = 0 kernel showing up.
print("\nSLOWEST branch by direction -- zero means waves do not propagate at all:")
for nm, d in DIRS.items():
    n = np.array(d, float); n /= np.linalg.norm(n)
    w = omega(n*kmag)/kmag
    ac = np.sort(w)[:3]
    print(f"{nm:>10}   acoustic branches {ac[0]:.6f} {ac[1]:.6f} {ac[2]:.6f}"
          + ("   <-- ZERO MODE" if ac[0] < 1e-6 else ""))
