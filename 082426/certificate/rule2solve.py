"""
IS ZAC'S SCHEME THE PHYSICS?
Solve R1 (all <111> = 1) + active shortcuts (=1) + R2 ([1,sqrt2] on every other
<100> neighbour), and nothing else.  If a solution EXISTS for the legal configs
and not for the undecided ones, the three rules ARE the veto -- local, finite,
and with no reference to packing at all.
"""
import numpy as np, math, time, collections
import scipy.sparse as sp
from scipy.optimize import least_squares
from lattice import build
NODE,KEY,BASE=build(); S=1/math.sqrt(3); N=len(NODE)
X0=np.array([[c*S for c in n] for n in NODE]); BASE_E=[tuple(e) for e in BASE]
CAND=[[102,179],[103,180],[136,137],[137,215],[137,144],[137,138],[143,144],[144,222],
      [144,145],[172,179],[173,180],[179,256],[179,186],[179,180],[180,257],[180,187],
      [214,215],[215,222],[215,216],[221,222],[222,223]]
rods=lambda m:[tuple(sorted(CAND[b])) for b in range(len(CAND)) if m>>b&1]
SC_ALL=[]
for i,c in enumerate(NODE):
    for a in range(3):
        d=list(c); d[a]+=2; j=KEY.get(tuple(d))
        if j is not None: SC_ALL.append((min(i,j),max(i,j)))
SC_ALL=sorted(set(SC_ALL)); R2HI=math.sqrt(2.0)
def solve(sc,w=1e3,maxit=600):
    scset=set(sc)
    R2=np.array([p for p in SC_ALL if p not in scset])
    E=BASE_E+list(sc); ne=len(E); nd=3*N; sw=math.sqrt(w); nb=len(BASE_E)
    I=np.array([i for i,_ in E]); J=np.array([j for _,j in E])
    rows=[];cols=[]
    for r,(i,j) in enumerate(E):
        for a in range(3): rows+=[r,r]; cols+=[3*i+a,3*j+a]
    for r,(i,j) in enumerate(R2):
        for a in range(3): rows+=[ne+r,ne+r]; cols+=[3*i+a,3*j+a]
    Sp=sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(ne+len(R2),nd))
    def fun(u):
        X=X0+u.reshape(-1,3)
        d=X[J]-X[I]; L=np.sqrt((d*d).sum(1))-1.0; L[nb:]*=sw
        dr=X[R2[:,1]]-X[R2[:,0]]; Lr=np.sqrt((dr*dr).sum(1))
        pen=(np.maximum(0.0,1.0-Lr)+np.maximum(0.0,Lr-R2HI))*sw
        return np.concatenate([L,pen])
    r=least_squares(fun,np.zeros(nd),jac_sparsity=Sp,method='trf',tr_solver='lsmr',
                    xtol=1e-14,ftol=1e-14,gtol=1e-14,max_nfev=maxit)
    X=X0+r.x.reshape(-1,3)
    d=X[J]-X[I]; L=np.sqrt((d*d).sum(1))
    base=float(np.abs(L[:nb]-1).max()); scr=float(np.abs(L[nb:]-1).max()) if len(sc) else 0.0
    dr=X[R2[:,1]]-X[R2[:,0]]; Lr=np.sqrt((dr*dr).sum(1))
    return base,scr,float(Lr.min()),float(Lr.max()),int((Lr<1-1e-7).sum()),int((Lr>R2HI+1e-7).sum())
UND=[30,60,120,180,550,585,676,774,778,833,840,900,904,2097,2594,2625,2818,2820,3077,
     3105,3137,3232,3330,3392,4680,5192,8769,8833,10753,10754,11265,65618,65624,65684,
     65712,65744,65816,65936,66113,66180,66192,66240,66306,69792,98436,131236]
LEG=[92,688,771,788,802,912,1608,2086,2136,2224,2817,2824,2832,3168,3840,4704,7170,8707,
     8770,8961,8962,10274,65606,65610,65612,65634,65681,65682,65688,65704,65730,65732,
     65760,65809,65858,65864,65888,66752,73794,73800,81988,156,792,834]
TOL=1e-6
print("R1 + shortcuts + R2 range enforced together\n")
print(f"{'set':>14} {'n':>4}  {'satisfies all':>14}   {'minR2':>7} {'maxR2':>7}")
for name,lst in (("PROVEN LEGAL",LEG),("UNDECIDED",UND)):
    t0=time.time(); ok=0; mns=[];mxs=[]
    for m in lst:
        b,s,mn,mx,lo,hi=solve(rods(m))
        good = b<TOL and s<TOL and lo==0 and hi==0
        ok+=good; mns.append(mn); mxs.append(mx)
    print(f"{name:>14} {len(lst):>4}  {ok:>6}/{len(lst):<7}   {min(mns):7.4f} {max(mxs):7.4f}   ({time.time()-t0:.0f}s)")
