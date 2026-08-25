import numpy as np, math, itertools, collections
from lattice import build
np.set_printoptions(precision=4,suppress=True)
NODE,KEY,BASE=build()
S=1/math.sqrt(3); N=len(NODE)
P=np.array([[c*S for c in n] for n in NODE])
CAND=[[102,179],[103,180],[136,137],[137,215],[137,144],[137,138],[143,144],[144,222],
      [144,145],[172,179],[173,180],[179,256],[179,186],[179,180],[180,257],[180,187],
      [214,215],[215,222],[215,216],[221,222],[222,223]]
UND=[30,60,120,180,550,585,676,774,778,833,840,900,904,2097,2594,2625,2818,2820,3077,
     3105,3137,3232,3330,3392,4680,5192,8769,8833,10753,10754,11265,65618,65624,65684,
     65712,65744,65816,65936,66113,66180,66192,66240,66306,69792,98436,131236]
LEG=[92,688,771,788,802,912,1608,2086,2136,2224,2817,2824,2832,3168,3840,4704,7170,8707,
     8770,8961,8962,10274,65606,65610,65612,65634,65681,65682,65688,65704,65730,65732,
     65760,65809,65858,65864,65888,66752,73794,73800,81988,156,792,834]
DELTA=1.0-2/math.sqrt(3)          # every shortcut contracts by this
def rigidity(edges):
    R=np.zeros((len(edges),3*N))
    for r,(i,j) in enumerate(edges):
        d=P[j]-P[i]; d=d/np.linalg.norm(d)
        R[r,3*i:3*i+3]=-d; R[r,3*j:3*j+3]=d
    return R
def rods(m): return [tuple(CAND[b]) for b in range(len(CAND)) if m>>b & 1]
BASE_E=[tuple(e) for e in BASE]
def test(m, tol=1e-9):
    sc=rods(m); E=BASE_E+sc
    R=rigidity(E)
    # self-stresses = left null space of R
    U,s,Vt=np.linalg.svd(R, full_matrices=True)
    ns=(s<tol).sum() + (R.shape[0]-len(s))
    W=U[:, R.shape[0]-ns:] if ns>0 else np.zeros((R.shape[0],0))
    # dL: 0 on base, DELTA on shortcuts
    dL=np.zeros(len(E)); dL[len(BASE_E):]=DELTA
    # first-order feasibility <=> W^T dL = 0
    v=W.T@dL if W.shape[1] else np.zeros(0)
    return dict(selfStresses=int(W.shape[1]),
                worstViolation=float(np.abs(v).max()) if v.size else 0.0,
                feasible1st=bool(v.size==0 or np.abs(v).max()<1e-8))
print("first-order feasibility via SELF-STRESS:  need  omega . dL = 0  for every self-stress\n")
for name,lst in (("UNDECIDED",UND),("PROVEN LEGAL",LEG)):
    res=[test(m) for m in lst]
    inf=[r for r in res if not r['feasible1st']]
    ss=collections.Counter(r['selfStresses'] for r in res)
    print(f"{name:14s} n={len(res):3d}   first-order INFEASIBLE: {len(inf):3d}"
          f"   selfStressCounts={dict(sorted(ss.items()))}")
    if inf:
        w=[r['worstViolation'] for r in inf]
        print(f"                 violation range {min(w):.3e} .. {max(w):.3e}")
