"""
THE RAY SURFACE.  Energy travels at v_g = grad_k omega, which in an anisotropic
medium is NOT parallel to k. The disturbance front is the outer envelope of
{v_g(k)}, so the reach in a spatial direction m is the support function
    f(m) = max over k of  v_g(k) . m
That f is what a causal cone should use, in place of a hop count.
"""
import numpy as np, math, json
from torus_gf import Torus
t = Torus(8)
def omega(k):
    w = np.linalg.eigvalsh(t.phi_k(np.asarray(k,float)))
    return np.sqrt(np.clip(w,0,None))
def vg_samples(nk=42, kmag=1e-3, h=1e-5):
    """group velocities of the three ACOUSTIC branches over many k directions"""
    V=[]
    ga=math.pi*(3-math.sqrt(5))
    for i in range(nk*nk):
        z=1-2*(i+0.5)/(nk*nk); r=math.sqrt(max(0,1-z*z)); th=ga*i
        n=np.array([r*math.cos(th), r*math.sin(th), z])
        k=n*kmag
        w0=omega(k)
        G=np.zeros((6,3))
        for a in range(3):
            e=np.zeros(3); e[a]=h
            G[:,a]=(omega(k+e)-omega(k-e))/(2*h)
        order=np.argsort(w0)
        for b in order[:3]:                      # acoustic only
            V.append(G[b])
    return np.array(V)
V=vg_samples()
print(f"sampled {len(V)} acoustic group velocities\n")
def support(m):
    m=np.asarray(m,float); m/=np.linalg.norm(m)
    return float((V@m).max())
DIRS={'<100>':(1,0,0),'<110>':(1,1,0),'<111>':(1,1,1),'<210>':(2,1,0),'<211>':(2,1,1),'<221>':(2,2,1)}
base=support((1,0,0))
print(f"{'direction':>10} {'reach f(m)':>12} {'rel <100>':>11}   BFS cube rel")
CUBE={'<100>':1.0,'<110>':math.sqrt(2),'<111>':math.sqrt(3),'<210>':math.sqrt(5)/2*2/2,'<211>':None,'<221>':None}
for nm,d in DIRS.items():
    f=support(d); c=CUBE.get(nm)
    print(f"{nm:>10} {f:12.5f} {f/base:11.4f}   "+(f"{c:.4f}" if c else "   –"))
mx=max(support(d) for d in DIRS.values()); mn=min(support(d) for d in DIRS.values())
print(f"\nanisotropy of the RAY surface  max/min = {mx/mn:.4f}")
print(f"anisotropy of the BFS cube               = {math.sqrt(3):.4f}")
# export a direction->reach table for the engine
tab=[]
ga=math.pi*(3-math.sqrt(5))
NS=256
for i in range(NS):
    z=1-2*(i+0.5)/NS; r=math.sqrt(max(0,1-z*z)); th=ga*i
    n=[r*math.cos(th), r*math.sin(th), z]
    tab.append([round(n[0],6),round(n[1],6),round(n[2],6),round(support(n)/base,6)])
json.dump(tab, open('raysurface.json','w'))
print(f"\nwrote raysurface.json: {NS} directions, reach normalised to <100> = 1")
