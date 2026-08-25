import numpy as np, torus_gf as e, scipy.sparse as sp
from scipy.optimize import least_squares
np.set_printoptions(precision=4, suppress=True)
"""
AFFINE vs RELAXED elastic response.  The Born term C_affine assumes every node
follows the homogeneous strain.  A lattice with a basis does not: the internal
coordinates relax, and the relaxed constant is
    C_relaxed = C_affine - coupling^T Phi^+ coupling
which can be far smaller -- and if Phi has zero modes in the right place, zero.
Test: impose a homogeneous strain, relax all internal coords to minimise the
base-length residual, and report what residual survives.
"""
t=e.Torus(4); nl=e.Nonlinear(t); L=2*t.N*e.S; S0=nl.X0/L
rows,cols=[],[]
for r,(i,j) in enumerate(nl.bonds):
    for a in range(3): rows+=[r,r]; cols+=[3*i+a,3*j+a]
Sp=sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(nl.bonds),nl.ndof))
def lens(A,Sf,pairs):
    out=np.empty(len(pairs))
    for r,(i,j) in enumerate(pairs):
        d=Sf[j]-Sf[i]; d=d-np.round(d); out[r]=np.linalg.norm(A@d)
    return out
def relaxed_residual(E, amp):
    A=L*(np.eye(3)+amp*E)
    affine=float(np.abs(lens(A,S0,nl.bonds)-1).max())
    f=lambda u:lens(A,S0+u.reshape(-1,3),nl.bonds)-1.0
    r=least_squares(f,np.zeros(nl.ndof),jac_sparsity=Sp,method="trf",
                    tr_solver="lsmr",xtol=1e-14,ftol=1e-14,gtol=1e-14,max_nfev=300)
    return affine, float(np.abs(f(r.x)).max())
TESTS={"hydrostatic diag(1,1,1)":np.eye(3)/np.sqrt(3),
       "Bain        diag(1,-1,0)":np.diag([1,-1,0])/np.sqrt(2),
       "Bain        diag(1,1,-2)":np.diag([1,1,-2])/np.sqrt(6),
       "shear       e_xy":np.array([[0,1,0],[1,0,0],[0,0,0]])/np.sqrt(2),
       "shear       e_xz":np.array([[0,0,1],[0,0,0],[1,0,0]])/np.sqrt(2),
       "shear       e_yz":np.array([[0,0,0],[0,0,1],[0,1,0]])/np.sqrt(2)}
amp=0.05
print(f"homogeneous strain amplitude {amp}, N=4\n")
print(f"{'direction':26s} {'affine resid':>13s} {'RELAXED resid':>14s}   verdict")
for k,E in TESTS.items():
    a,rr=relaxed_residual(E,amp)
    v="FREE after relaxation" if rr<1e-9 else "costs"
    print(f"{k:26s} {a:13.3e} {rr:14.3e}   {v}")
