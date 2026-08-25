"""
DOES RULE 2 EXPLAIN THE VETO?

Zac's scheme:
  R1  every <111> neighbour at exactly 1
  R2  every <100> neighbour (integer offset 2 along a cube axis) in [1, sqrt2]
      -- 1 = fired shortcut, sqrt2 = the octahedron case, 2/sqrt3 = rest
  R3  nothing else constrains

The engine enforces R1, the active shortcuts at 1, and >=1 on every non-bonded
pair. It has NO UPPER BOUND on <100> pairs. So R2's ceiling is a constraint the
simulation never applied. Test: in a solution satisfying only the engine's
constraints, does any <100> pair exceed sqrt2?
"""
import numpy as np, math, collections
import scipy.sparse as sp
from scipy.optimize import least_squares
from lattice import build
NODE,KEY,BASE=build(); S=1/math.sqrt(3); N=len(NODE)
X0=np.array([[c*S for c in n] for n in NODE])
BASE_E=[tuple(e) for e in BASE]
CAND=[[102,179],[103,180],[136,137],[137,215],[137,144],[137,138],[143,144],[144,222],
      [144,145],[172,179],[173,180],[179,256],[179,186],[179,180],[180,257],[180,187],
      [214,215],[215,222],[215,216],[221,222],[222,223]]
rods=lambda m:[tuple(sorted(CAND[b])) for b in range(len(CAND)) if m>>b&1]
# every <100> neighbour pair in the lattice (integer offset 2 along a cube axis)
SC_ALL=[]
for i,c in enumerate(NODE):
    for a in range(3):
        d=list(c); d[a]+=2
        j=KEY.get(tuple(d))
        if j is not None: SC_ALL.append((min(i,j),max(i,j)))
SC_ALL=sorted(set(SC_ALL))
print(f"lattice: {N} nodes, {len(BASE_E)} <111> bonds, {len(SC_ALL)} <100> neighbour pairs\n")
def solve_engine_rules(sc, w=1e3, maxit=400):
    """R1 + active shortcuts at 1.  (No <100> ceiling -- the engine has none.)"""
    E=BASE_E+list(sc); ne=len(E); nd=3*N; sw=math.sqrt(w)
    I=np.array([i for i,_ in E]); J=np.array([j for _,j in E])
    rows=[];cols=[]
    for r,(i,j) in enumerate(E):
        for a in range(3): rows+=[r,r]; cols+=[3*i+a,3*j+a]
    Sp=sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(ne,nd))
    def fun(u):
        X=X0+u.reshape(-1,3); d=X[J]-X[I]
        L=np.sqrt((d*d).sum(1))-1.0; L[len(BASE_E):]*=sw; return L
    r=least_squares(fun,np.zeros(nd),jac_sparsity=Sp,method='trf',tr_solver='lsmr',
                    xtol=1e-14,ftol=1e-14,gtol=1e-14,max_nfev=maxit)
    return X0+r.x.reshape(-1,3)
def r2_report(X, sc):
    scset=set(sc)
    A=np.array([p for p in SC_ALL if p not in scset])
    d=X[A[:,1]]-X[A[:,0]]; L=np.sqrt((d*d).sum(1))
    return float(L.min()), float(L.max()), int((L<1-1e-9).sum()), int((L>math.sqrt(2)+1e-9).sum())
UND=[30,60,120,180,550,585,676,774,778,833,840,900,904,2097,2594,2625,2818,2820,3077,
     3105,3137,3232,3330,3392,4680,5192,8769,8833,10753,10754,11265,65618,65624,65684,
     65712,65744,65816,65936,66113,66180,66192,66240,66306,69792,98436,131236]
LEG=[92,688,771,788,802,912,1608,2086,2136,2224,2817,2824,2832,3168,3840,4704,7170,8707,
     8770,8961,8962,10274,65606,65610,65612,65634,65681,65682,65688,65704,65730,65732,
     65760,65809,65858,65864,65888,66752,73794,73800,81988,156,792,834]
print(f"{'set':>14}  {'minR2':>8} {'maxR2':>8}  {'below 1':>9} {'above sqrt2':>12}")
for name,lst in (("PROVEN LEGAL",LEG),("UNDECIDED",UND)):
    mn=[];mx=[];lo=[];hi=[]
    for m in lst:
        sc=rods(m); X=solve_engine_rules(sc)
        a,b,c,d=r2_report(X,sc); mn.append(a);mx.append(b);lo.append(c);hi.append(d)
    print(f"{name:>14}  {min(mn):8.4f} {max(mx):8.4f}  "
          f"{sum(1 for x in lo if x):>4}/{len(lst)} cfg {sum(1 for x in hi if x):>5}/{len(lst)} cfg")
    print(f"{'':>14}  violating pairs: below-1 mean {np.mean(lo):.1f}, above-sqrt2 mean {np.mean(hi):.1f}")
