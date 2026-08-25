import numpy as np, itertools, torus_gf as e
np.set_printoptions(precision=5, suppress=True)
"""
WHY DOES THE EQUATORIAL-PLANE PAIR GO DARK?

A 4-rod oct in the XZ plane has its ring at integer coords
    (0,0,0) (2,0,0) (0,0,2) (2,0,2)      [all even -> sublattice 0]
i.e. cells (0,0,0) (1,0,0) (0,0,1) (1,0,1), with rods
    two along X, at z-cell 0 and z-cell 1
    two along Z, at x-cell 0 and x-cell 1

Kernel modes cost no energy, so the response is carried by them, weighted by
how strongly the source couples: |S z| for each kernel mode z.  Group by which
of the six <110> lines the mode sits on and see which group gets zero.
"""
N = 6
t = e.Torus(N)
Z = t.kernel_basis()
print(f"torus N={N}: {t.nnodes} nodes, dim ker Phi = {Z.shape[1]} (6N-3 = {6*N-3})")

def oct_rows(plane):
    """the four rods of a 4-rod oct whose equator lies in `plane`"""
    ax = {"XZ": (0, 2), "XY": (0, 1), "YZ": (1, 2)}[plane]
    a1, a2 = ax
    o = np.zeros(3, int)
    e1 = np.eye(3, dtype=int)[a1]; e2 = np.eye(3, dtype=int)[a2]
    return [t.sc_row(o,      a1, 0),      # rod along a1, at a2-offset 0
            t.sc_row(e2,     a1, 0),      # rod along a1, at a2-offset 1
            t.sc_row(o,      a2, 0),      # rod along a2, at a1-offset 0
            t.sc_row(e1,     a2, 0)]      # rod along a2, at a1-offset 1

# label every kernel basis vector by which line family it lives on.
# Do it by projecting onto the plane-wave subspaces: a mode on family k_a=0 is
# invariant under translation by one cell along a.
def family_of(z):
    """which coordinate axis does this kernel mode NOT vary along?"""
    zz = z.reshape(-1, 6) if False else z
    out = []
    for a in range(3):
        sh = np.zeros_like(z)
        for n in itertools.product(range(N), repeat=3):
            n = np.array(n)
            m = n.copy(); m[a] = (m[a]+1) % N
            for s in (0,1):
                i = t.dof(n, s); j = t.dof(m, s)
                sh[j:j+3] = z[i:i+3]
        out.append(float(np.abs(sh - z).max()))
    return out

print("\nlabelling the kernel basis by translation invariance...")
lab = []
for c in range(Z.shape[1]):
    d = family_of(Z[:, c])
    lab.append(d)
lab = np.array(lab)
inv = lab < 1e-8          # invariant under a one-cell shift along that axis
print(f"  modes invariant along x / y / z: {inv[:,0].sum()} / {inv[:,1].sum()} / {inv[:,2].sum()}")

print("\n=== coupling |S z| of a 4-rod oct to each kernel mode ===")
for plane in ("XZ", "XY", "YZ"):
    S = np.array(oct_rows(plane))
    C = S @ Z                                # 4 x (6N-3)
    strength = np.linalg.norm(C, axis=0)     # per kernel mode
    print(f"\n  oct equator in {plane}:")
    for a, nm in enumerate("xyz"):
        m = inv[:, a]
        if m.sum() == 0: continue
        print(f"     modes with k_{nm}=0  (the <110> pair lying "
              f"{'IN' if nm not in plane else 'OUT of'} the {plane} plane): "
              f"n={m.sum():3d}  max|Sz| = {strength[m].max():.3e}")
    other = ~(inv[:,0] | inv[:,1] | inv[:,2])
    if other.sum():
        print(f"     modes on no single family: n={other.sum()}  max|Sz| = {strength[other].max():.3e}")
