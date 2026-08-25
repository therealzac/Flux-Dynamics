import numpy as np, math, time
import scipy.sparse as sp
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from lattice import build
NODE,KEY,BASE=build(); S=1/math.sqrt(3); N=len(NODE)
X0=np.array([[c*S for c in n] for n in NODE])
CAND=[[102,179],[103,180],[136,137],[137,215],[137,144],[137,138],[143,144],[144,222],
      [144,145],[172,179],[173,180],[179,256],[179,186],[179,180],[180,257],[180,187],
      [214,215],[215,222],[215,216],[221,222],[222,223]]
BASE_E=[tuple(e) for e in BASE]
def rods(m): return [tuple(CAND[b]) for b in range(len(CAND)) if m>>b & 1]
def solve(m,w=1e3,maxit=600):
    sc=rods(m); E=BASE_E+list(sc); sw=math.sqrt(w); ne=len(E); nd=3*N
    rows=[];cols=[]
    for r,(i,j) in enumerate(E):
        for a in range(3): rows+=[r,r]; cols+=[3*i+a,3*j+a]
    Sp=sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(ne,nd))
    I=np.array([i for i,_ in E]); J=np.array([j for _,j in E])
    def fun(u):
        X=X0+u.reshape(-1,3); d=X[J]-X[I]
        L=np.sqrt((d*d).sum(1))-1.0; L[len(BASE_E):]*=sw; return L
    r=least_squares(fun,np.zeros(nd),jac_sparsity=Sp,method='trf',tr_solver='lsmr',
                    xtol=1e-15,ftol=1e-15,gtol=1e-15,max_nfev=maxit)
    X=X0+r.x.reshape(-1,3); d=X[J]-X[I]; L=np.sqrt((d*d).sum(1))
    base=float(np.abs(L[:len(BASE_E)]-1).max()); scr=float(np.abs(L[len(BASE_E):]-1).max())
    t=cKDTree(X); pr=t.query_pairs(1.0-1e-9)
    bs=set(BASE_E)|{(j,i) for i,j in BASE_E}|set(sc)|{(j,i) for i,j in sc}
    bad=[p for p in pr if p not in bs and (p[1],p[0]) not in bs]
    ms=min((np.linalg.norm(X[a]-X[b]) for a,b in bad),default=1.0)
    return base,scr,ms,len(bad)
UND=[30,60,120,180,550,585,676,774,778,833,840,900,904,2097,2594,2625,2818,2820,3077,
     3105,3137,3232,3330,3392,4680,5192,8769,8833,10753,10754,11265,65618,65624,65684,
     65712,65744,65816,65936,66113,66180,66192,66240,66306,69792,98436,131236]
LEG=[92,688,771,788,802,912,1608,2086,2136,2224,2817,2824,2832,3168,3840,4704,7170,8707,
     8770,8961,8962,10274,65606,65610,65612,65634,65681,65682,65688,65704,65730,65732,
     65760,65809,65858,65864,65888,66752,73794,73800,81988,156,792,834]
print("LENGTH-ONLY solve (no repulsion).  Does minSep<1 separate the populations?\n")
for name,lst in (("UNDECIDED",UND),("PROVEN LEGAL",LEG)):
    res=[solve(m) for m in lst]
    ms=[r[2] for r in res]; ov=[r[3] for r in res]
    good=sum(1 for x in ms if x>1-1e-8)
    print(f"{name:14s} n={len(res):3d}  lengths always solved (base max {max(r[0] for r in res):.1e})")
    print(f"               minSep>=1 in {good}/{len(res)}   minSep range {min(ms):.4f}..{max(ms):.4f}")
    print(f"               overlapping pairs: min {min(ov)}  max {max(ov)}  mean {sum(ov)/len(ov):.1f}\n")
