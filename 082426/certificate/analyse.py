import itertools, math, collections
from lattice import build
NODE,KEY,BASE=build()
CAND=[[102,179],[103,180],[136,137],[137,215],[137,144],[137,138],[143,144],[144,222],
      [144,145],[172,179],[173,180],[179,256],[179,186],[179,180],[180,257],[180,187],
      [214,215],[215,222],[215,216],[221,222],[222,223]]
UNDECIDED=[30,60,120,180,550,585,676,774,778,833,840,900,904,2097,2594,2625,2818,2820,
           3077,3105,3137,3232,3330,3392,4680,5192,8769,8833,10753,10754,11265,65618,
           65624,65684,65712,65744,65816,65936,66113,66180,66192,66240,66306,69792,
           98436,131236]
LEGAL=[92,688,771,788,802,912,1608,2086,2136,2224,2817,2824,2832,3168,3840,4704,7170,
       8707,8770,8961,8962,10274,65606,65610,65612,65634,65681,65682,65688,65704,65730,
       65732,65760,65809,65858,65864,65888,66752,73794,73800,81988,156,792,834]
def rods(m): return [CAND[b] for b in range(len(CAND)) if m>>b & 1]
def ax(r):
    a,b=NODE[r[0]],NODE[r[1]]
    return [k for k in range(3) if a[k]!=b[k]][0]
def other(r,n): return r[1] if r[0]==n else r[0]
def feats(m):
    R=rods(m); deg=collections.Counter()
    for r in R: deg[r[0]]+=1; deg[r[1]]+=1
    mx=max(deg.values()); best=None
    for h,c in deg.items():
        if c!=mx: continue
        at=[r for r in R if h in r]
        axes=sorted(ax(r) for r in at)
        coll=sum(1 for a,b in itertools.combinations(at,2)
                 if ax(a)==ax(b) and NODE[other(a,h)][ax(a)]!=NODE[other(b,h)][ax(b)])
        perp=sum(1 for a,b in itertools.combinations(at,2) if ax(a)!=ax(b))
        cand=(mx,len(set(axes)),coll,perp)
        if best is None or cand>best: best=cand
    return dict(maxDeg=best[0],distinctAxAtHub=best[1],collinearAtHub=best[2],perpAtHub=best[3])
# sanity: rods of mask 30 must match what the engine reported
chk=[[NODE[i],NODE[j]] for i,j in rods(30)]
want=[[(-2,0,0),(0,0,0)],[(-1,-1,-3),(-1,-1,-1)],[(-1,-1,-1),(1,-1,-1)],[(-1,-1,-1),(-1,1,-1)]]
assert [[tuple(a),tuple(b)] for a,b in chk]==want, chk
print("index reconstruction verified against the engine\n")
U=[feats(m) for m in UNDECIDED]; L=[feats(m) for m in LEGAL]
def tab(rows,k): return dict(sorted(collections.Counter(r[k] for r in rows).items()))
print(f"undecided n={len(U)}   proven-legal n={len(L)}\n")
for k in ('maxDeg','distinctAxAtHub','collinearAtHub','perpAtHub'):
    print(f"{k}\n   undecided {tab(U,k)}\n   legal     {tab(L,k)}")
print("\njoint (maxDeg, perpAtHub):")
cu=collections.Counter((r['maxDeg'],r['perpAtHub']) for r in U)
cl=collections.Counter((r['maxDeg'],r['perpAtHub']) for r in L)
for k in sorted(set(cu)|set(cl)):
    print(f"   {str(k):>8}   undecided {cu.get(k,0):>3}   legal {cl.get(k,0):>3}")
