import numpy as np, torus_gf as e, time, json, sys
N=int(sys.argv[1]) if len(sys.argv)>1 else 4
t=e.Torus(N); nl=e.Nonlinear(t); P=e.cell_pairs(t)
CFG=[("polar + 1 equatorial          ",["polar","eqA1"],"legal"),
     ("polar + 2 equatorial same axis",["polar","eqA1","eqA2"],"legal"),
     ("4 equatorial, no polar        ",["eqA1","eqA2","eqB1","eqB2"],"legal"),
     ("polar + 2 equatorial diff axes",["polar","eqA1","eqB1"],"illegal"),
     ("polar + 3 equatorial          ",["polar","eqA1","eqA2","eqB1"],"illegal"),
     ("polar + 4 equatorial          ",["polar","eqA1","eqA2","eqB1","eqB2"],"illegal")]
print(f"torus N={N}  {t.nnodes} nodes  {6*N**3} dof")
print(f"{'configuration':32s} {'want':8s} {'got':8s} {'base resid':>12s} {'sc resid':>11s} nfev   s")
bad=0
for name,slots,want in CFG:
    t0=time.time()
    r=nl.solve([P[s] for s in slots])
    got="legal" if r["legal"] else "illegal"
    if got!=want: bad+=1
    print(f"{name} {want:8s} {got:8s} {r['base']:12.3e} {r['sc']:11.3e} {r['nfev']:4d} {time.time()-t0:5.1f}"
          + ("   <-- MISMATCH" if got!=want else ""), flush=True)
print("\nmismatches:",bad,"/",len(CFG))
