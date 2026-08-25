import numpy as np, math, sys, time
import scipy.sparse as sp
from scipy.optimize import least_squares
from lattice import build
NODE,KEY,BASE=build()
S=1/math.sqrt(3); N=len(NODE)
X0=np.array([[c*S for c in n] for n in NODE])
CAND=[[102,179],[103,180],[136,137],[137,215],[137,144],[137,138],[143,144],[144,222],
      [144,145],[172,179],[173,180],[179,256],[179,186],[179,180],[180,257],[180,187],
      [214,215],[215,222],[215,216],[221,222],[222,223]]
BASE_E=[tuple(e) for e in BASE]
def rods(m): return [tuple(CAND[b]) for b in range(len(CAND)) if m>>b & 1]
def solve(m, w=1e3, maxit=600):
    """
    INDEPENDENT of the engine: Levenberg-Marquardt on the exact length residuals,
    sparse analytic Jacobian.  The engine uses Gauss-Seidel projection; this uses
    a trust-region least-squares step.  Different algorithm, different failure
    modes -- so agreement is meaningful and disagreement localises the problem.
    """
    sc=rods(m); E=BASE_E+list(sc); sw=math.sqrt(w)
    ne=len(E); nd=3*N
    rows=[];cols=[]
    for r,(i,j) in enumerate(E):
        for a in range(3): rows+= [r,r]; cols+= [3*i+a,3*j+a]
    Sp=sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(ne,nd))
    def lens(X):
        d=X[[j for _,j in E]]-X[[i for i,_ in E]]
        return np.sqrt((d*d).sum(1))
    def fun(u):
        X=X0+u.reshape(-1,3); L=lens(X)-1.0
        L[len(BASE_E):]*=sw
        return L
    r=least_squares(fun,np.zeros(nd),jac_sparsity=Sp,method='trf',tr_solver='lsmr',
                    xtol=1e-15,ftol=1e-15,gtol=1e-15,max_nfev=maxit)
    X=X0+r.x.reshape(-1,3); L=lens(X)
    base=float(np.abs(L[:len(BASE_E)]-1).max()); scr=float(np.abs(L[len(BASE_E):]-1).max())
    # non-overlap: closest non-base pair
    from scipy.spatial import cKDTree
    t=cKDTree(X); pairs=t.query_pairs(1.0-1e-9)
    bs=set(BASE_E)|{(j,i) for i,j in BASE_E}|set(sc)|{(j,i) for i,j in sc}
    bad=[p for p in pairs if p not in bs and (p[1],p[0]) not in bs]
    minsep=min((np.linalg.norm(X[a]-X[b]) for a,b in bad), default=1.0)
    return base, scr, minsep, r.nfev
UND=[30,60,120,180,550,585,676,774,778,833,840,900,904,2097,2594,2625,2818,2820,3077,
     3105,3137,3232,3330,3392,4680,5192,8769,8833,10753,10754,11265,65618,65624,65684,
     65712,65744,65816,65936,66113,66180,66192,66240,66306,69792,98436,131236]

print("independent LM solver, 360 nodes, 1168 base + k shortcut constraints\n")
print(f"{'mask':>8} {'base':>11} {'sc':>11} {'minSep':>9} {'nfev':>5}  verdict")
legal=0
t0=time.time()
for m in UND:
    b,s,ms,nf=solve(m)
    ok = b<1e-8 and s<1e-8 and ms>1-1e-8
    if ok: legal+=1
    print(f"{m:>8} {b:11.3e} {s:11.3e} {ms:9.6f} {nf:>5}  {'LEGAL' if ok else 'refused'}",flush=True)
print(f"\n{legal}/{len(UND)} legal by the independent solver   ({time.time()-t0:.0f}s)")
