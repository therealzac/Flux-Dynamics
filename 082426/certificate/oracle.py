"""
LM LEGALITY ORACLE.

Answers "does a legal packing exist for this shortcut set?" by CONSTRUCTION,
not by exhausting a sweep budget. Two ideas make it fast:

  * Levenberg-Marquardt on the exact length residuals. Second-order, so it
    converges in ~12 iterations where the engine's Gauss-Seidel projection
    needs tens of thousands of sweeps and often never gets there. LM's damping
    is essential -- the Jacobian is badly singular here (142 self-stresses plus
    the floppy manifold), so undamped Newton would fail.

  * ACTIVE SET for packing. Non-overlap is ~64,000 inequalities, but almost all
    are slack. Solve the lengths, see which pairs actually overlap, enforce only
    those, repeat. The active set converges in a few rounds and stays tiny.

Verdicts:
  LEGAL     a configuration satisfying every length AND every non-overlap
            constraint was constructed. This is a WITNESS -- it is a proof.
  INFEASIBLE the active set stopped growing while violations persist: the
            solver has a stationary point that still overlaps. Strong evidence,
            not a proof.
"""
import numpy as np, math, time
import scipy.sparse as sp
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from lattice import build

NODE,KEY,BASE=build(); S=1/math.sqrt(3); N=len(NODE)
X0=np.array([[c*S for c in n] for n in NODE])
BASE_E=[tuple(e) for e in BASE]
_BS=set(BASE_E)|{(j,i) for i,j in BASE_E}
TOL=1e-7

def legal(sc, maxrounds=15, w=1e3, verbose=False):
    sc=[tuple(sorted(p)) for p in sc]
    scset=set(sc)|{(j,i) for i,j in sc}
    act=[]                                  # active repulsion pairs
    nd=3*N; sw=math.sqrt(w)
    for rnd in range(maxrounds):
        E=BASE_E+list(sc); ne=len(E)
        AP=np.array(act,dtype=int).reshape(-1,2)
        I=np.array([i for i,_ in E]); J=np.array([j for _,j in E])
        rows=[];cols=[]
        for r,(i,j) in enumerate(E):
            for a in range(3): rows+=[r,r]; cols+=[3*i+a,3*j+a]
        for r,(i,j) in enumerate(AP):
            for a in range(3): rows+=[ne+r,ne+r]; cols+=[3*i+a,3*j+a]
        Sp=sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(ne+len(AP),nd))
        def fun(u):
            X=X0+u.reshape(-1,3)
            d=X[J]-X[I]; L=np.sqrt((d*d).sum(1))-1.0; L[len(BASE_E):]*=sw
            if len(AP):
                # An ACTIVE inequality is TIGHT at the optimum, so enforce d = 1
                # exactly rather than a one-sided penalty. The penalty form left
                # a residual of order 1/weight -- it converged to minSep 0.9999
                # and failed its own control on 44 configurations known legal.
                dr=X[AP[:,1]]-X[AP[:,0]]; Lr=np.sqrt((dr*dr).sum(1))
                return np.concatenate([L,(Lr-1.0)*sw])
            return L
        r=least_squares(fun,np.zeros(nd),jac_sparsity=Sp,method='trf',
                        tr_solver='lsmr',xtol=1e-14,ftol=1e-14,gtol=1e-14,max_nfev=400)
        X=X0+r.x.reshape(-1,3)
        d=X[J]-X[I]; L=np.sqrt((d*d).sum(1))
        base=float(np.abs(L[:len(BASE_E)]-1).max()); scr=float(np.abs(L[len(BASE_E):]-1).max()) if len(sc) else 0.0
        t=cKDTree(X); pr=t.query_pairs(1.0-TOL)
        viol=[p for p in pr if p not in _BS and p not in scset and (p[1],p[0]) not in scset]
        if verbose: print(f"   round {rnd}: base {base:.2e} sc {scr:.2e} active {len(act)} viol {len(viol)}")
        md=float(np.abs(r.x.reshape(-1,3)).max())
        if not viol and base<1e-8 and scr<1e-8:
            # RECONSTRUCTION GUARD. At rest there are no non-bonded contacts; a
            # genuine solution of a k-shortcut configuration adds exactly k. A
            # large active set means the solver densified into a different
            # packing -- constraints satisfied, structure destroyed. That is the
            # failure that invalidated chained engine solves, and it invalidates
            # this the same way.
            recon = len(act) > 4*len(sc) + 4 or md > 0.45
            return dict(verdict='RECONSTRUCTED' if recon else 'LEGAL',
                        base=base, sc=scr, rounds=rnd+1, active=len(act), maxdisp=md)
        new=[p for p in viol if p not in set(map(tuple,act))]
        # release active pairs that have separated on their own
        if len(AP):
            keep=[]
            for k,(i,j) in enumerate(AP):
                if np.linalg.norm(X[j]-X[i])<1.0+1e-6: keep.append((int(i),int(j)))
            act=keep
        if not new:
            ms=min(np.linalg.norm(X[a]-X[b]) for a,b in viol) if viol else 1.0
            return dict(verdict='INFEASIBLE', base=base, sc=scr, rounds=rnd+1,
                        active=len(act), minSep=float(ms), stuckViolations=len(viol))
        act+=new
    return dict(verdict='UNDECIDED', rounds=maxrounds, active=len(act))

if __name__=="__main__":
    CAND=[[102,179],[103,180],[136,137],[137,215],[137,144],[137,138],[143,144],[144,222],
          [144,145],[172,179],[173,180],[179,256],[179,186],[179,180],[180,257],[180,187],
          [214,215],[215,222],[215,216],[221,222],[222,223]]
    rods=lambda m:[tuple(CAND[b]) for b in range(len(CAND)) if m>>b&1]
    # GROUND TRUTH from the engine (fresh solve from REST, both frames)
    GT=[("empty",           0,                        'LEGAL'),
        ("1 rod",           1<<13,                    'LEGAL'),
        ("one tet",         (1<<13)|(1<<7),           'LEGAL'),
        ("two tets edge",   (1<<13)|(1<<7)|(1<<3),    'LEGAL'),
        ("face-share",      (1<<13)|(1<<7)|(1<<17),   'ILLEGAL'),
        ("4 eq no polar",   (1<<7)|(1<<3)|(1<<17)|(1<<4),'ILLEGAL?')]
    print("validating the oracle against engine ground truth\n")
    for name,m,want in GT:
        t0=time.time(); r=legal(rods(m))
        print(f"  {name:16s} want {want:9s} got {r['verdict']:11s} "
              f"rounds {r['rounds']:2d} active {r['active']:3d}  {time.time()-t0:5.2f}s")
