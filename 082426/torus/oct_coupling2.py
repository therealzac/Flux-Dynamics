import numpy as np, itertools, torus_gf as e
np.set_printoptions(precision=5, suppress=True)
N = 6
t = e.Torus(N)

def kernel_modes():
    """Real-space kernel modes, each LABELLED by the line family it sits on."""
    out = []
    for m in itertools.product(range(N), repeat=3):
        if not t.kernel_predicate(m): continue
        k = 2*np.pi*np.array(m)/N
        w, V = np.linalg.eigh(t.phi_k(k))
        for c in np.where(w < 1e-10)[0]:
            v = V[:, c]
            for part in (np.real, np.imag):
                z = np.zeros(6*N**3)
                for n in itertools.product(range(N), repeat=3):
                    n = np.array(n)
                    ph = np.exp(1j*np.dot(k, n))
                    for s in (0, 1):
                        i = t.dof(n, s)
                        z[i:i+3] = part(v[3*s:3*s+3]*ph)
                if np.linalg.norm(z) > 1e-9:
                    fam = [a for a in range(3) if m[a] % N == 0]
                    out.append({"m": m, "fam": fam, "z": z/np.linalg.norm(z)})
    return out

def oct_rows(plane):
    a1, a2 = {"XZ": (0, 2), "XY": (0, 1), "YZ": (1, 2)}[plane]
    e1 = np.eye(3, dtype=int)[a1]; e2 = np.eye(3, dtype=int)[a2]
    o = np.zeros(3, int)
    return [t.sc_row(o, a1, 0), t.sc_row(e2, a1, 0),
            t.sc_row(o, a2, 0), t.sc_row(e1, a2, 0)]

def single_rod(axis):
    return [t.sc_row(np.zeros(3, int), axis, 0)]

M = kernel_modes()
print(f"torus N={N}: {len(M)} labelled kernel modes\n")
LINE = {0: "k_x=0  -> <011>,<01-1>", 1: "k_y=0  -> <101>,<10-1>", 2: "k_z=0  -> <110>,<1-10>"}

def report(rows, title):
    S = np.array(rows)
    print(title)
    for a in range(3):
        sel = [d for d in M if d["fam"] == [a]]          # exactly one zero comp
        if not sel: continue
        c = np.array([np.linalg.norm(S @ d["z"]) for d in sel])
        print(f"   {LINE[a]:26s} n={len(sel):3d}   max|Sz| = {c.max():.3e}   mean = {c.mean():.3e}")
    print()

for plane in ("XZ", "XY", "YZ"):
    report(oct_rows(plane), f"4-ROD OCT, equator in {plane}   (in-plane pair = "
           + {"XZ":"k_y=0","XY":"k_z=0","YZ":"k_x=0"}[plane] + ")")
for a, nm in enumerate("xyz"):
    report(single_rod(a), f"SINGLE ROD along {nm}")
