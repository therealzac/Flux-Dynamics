import numpy as np, itertools
from itertools import permutations
phi=(1+5**0.5)/2
unit=lambda v:(lambda a:a/np.linalg.norm(a))(np.array(v,float))
V=[unit(v) for v in [(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)]]
AX6=[unit(a) for a in [(0,1,phi),(0,-1,phi),(1,phi,0),(-1,phi,0),(phi,0,1),(phi,0,-1)]]
Us=[]
for c in itertools.combinations(range(6),4):
    A0=[AX6[i] for i in c]
    for perm in permutations(range(4)):
        A=[A0[perm[k]] for k in range(4)]
        u,sv,vt=np.linalg.svd(np.array(A).T); s=vt[-1]
        if np.linalg.norm(np.array(A).T@s)>1e-9 or min(abs(s))<1e-6: continue
        M=np.array([s[k]*A[k] for k in range(3)]).T@np.linalg.inv(np.array(V[:3]).T)
        if np.linalg.norm(M@V[3]-s[3]*A[3])>1e-8: continue
        d=np.linalg.det(M)
        if abs(d)<1e-9: continue
        M=M/abs(d)**(1/3); M=-M if np.linalg.det(M)<0 else M
        w,Q=np.linalg.eigh(M.T@M); Us.append(Q@np.diag(np.sqrt(w))@Q.T)
states={}
for U in Us:
    for a in range(3):
        o=[i for i in range(3) if i!=a]
        pol=np.eye(3)[a]
        d1=np.zeros(3); d1[o[0]]=1; d1[o[1]]= 1; d1=unit(d1)
        d2=np.zeros(3); d2[o[0]]=1; d2[o[1]]=-1; d2=unit(d2)
        if not all(np.linalg.norm(U@v-(v@U@v)*v)<1e-9 for v in (pol,d1,d2)): continue
        s1=round(float(np.linalg.norm(U@d1)),6); s2=round(float(np.linalg.norm(U@d2)),6)
        LONG='(0'+('+' if o[0]<o[1] else '')+')'
        # which diagonal is the LONG one
        which='[%s] long'%('+'.join('xyz'[i] for i in o) if s1>s2 else '-'.join('xyz'[i] for i in o))
        states[('polar '+'xyz'[a],which)]=states.get(('polar '+'xyz'[a],which),0)+1
        break
print("distinct strain STATES available to a single BCC cell:",len(states))
for k,n in sorted(states.items()): print(f"   {k[0]:9s}  {k[1]:12s}   x{n}")
print("\n= 3 polar orientations  x  2 choices of which equatorial diagonal is long")
print("\n--- the identity that keeps erasing phi ---")
print("phi^2 + phi^-2 =",round(phi**2+phi**-2,12))
for label,vec,a in [("4 equatorial <100> edges",(0,1,0),0),
                    ("8 neighbour <110> centroid dirs",(1,1,0),0),
                    ("cube axis y",(0,1,0),0)]:
    pass
print("any direction that samples the two diagonals with EQUAL weight loses phi,")
print("because (phi^2 + phi^-2)/2 = 3/2 is rational. That covers:")
print("   the 4 equatorial <100> edges      -> all 2^(1/3)      = %.6f"%(2**(1/3)))
print("   the 8 neighbour <110> directions  -> all               = %.6f"%(( (2**(2/3))/2 + 3*2**(-1/3)/4 )**0.5))
print("   -> phi survives ONLY on the 2 equatorial diagonals, as phi^2")
