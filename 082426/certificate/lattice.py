"""
Rebuild the 360-node ball exactly as buildLattice() does, so node INDICES match
the engine's. Order is the lexicographic (x,y,z) scan, keeping shared-parity
triples inside the radius.
"""
import numpy as np, math
S=0.577350269189626
LCENTER=[0,0,-1]; R=4.05
def build():
    Ri=math.ceil(R/S)+2
    c0=LCENTER; NODE=[]; KEY={}
    for x in range(math.floor(c0[0])-Ri, math.ceil(c0[0])+Ri+1):
     for y in range(math.floor(c0[1])-Ri, math.ceil(c0[1])+Ri+1):
      for z in range(math.floor(c0[2])-Ri, math.ceil(c0[2])+Ri+1):
        if not (x%2==y%2==z%2): continue
        if math.hypot(x-c0[0],y-c0[1],z-c0[2])*S > R: continue
        KEY[(x,y,z)]=len(NODE); NODE.append((x,y,z))
    BASE=[]
    dirs=[(1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1)]
    off=[]
    for d in dirs: off+= [d, (-d[0],-d[1],-d[2])]
    for i,c in enumerate(NODE):
        for d in off:
            j=KEY.get((c[0]+d[0],c[1]+d[1],c[2]+d[2]))
            if j is not None and i<j: BASE.append((i,j))
    return NODE,KEY,BASE
if __name__=="__main__":
    NODE,KEY,BASE=build()
    print("nodes",len(NODE),"(want 360)   baseEdges",len(BASE),"(want 1168)")
    print("NODE[0]  ",NODE[0],   "(want (-6,-2,-4))")
    print("NODE[179]",NODE[179], "(want (0,0,-2))")
    print("NODE[180]",NODE[180], "(want (0,0,0))")
    print("NODE[-1] ",NODE[-1],  "(want (6,2,2))")
