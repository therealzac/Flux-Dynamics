import numpy as np, torus_gf as e, time, sys, json
CFG=[("polar+1eq        ",["polar","eqA1"],"legal"),
     ("polar+2eq sameax ",["polar","eqA1","eqA2"],"legal"),
     ("4eq no polar     ",["eqA1","eqA2","eqB1","eqB2"],"legal"),
     ("polar+2eq diffax ",["polar","eqA1","eqB1"],"illegal"),
     ("polar+3eq        ",["polar","eqA1","eqA2","eqB1"],"illegal"),
     ("polar+4eq        ",["polar","eqA1","eqA2","eqB1","eqB2"],"illegal")]
out={}
for N in [int(x) for x in sys.argv[1:]] or [4,5,6,7,8]:
    t=e.Torus(N); nl=e.Sparse(t); P=e.cell_pairs(t)
    print(f"\n=== N={N}  {t.nnodes} nodes  {6*N**3} dof  (FIXED BOX) ===", flush=True)
    for name,slots,want in CFG:
        t0=time.time(); r=nl.solve([P[s] for s in slots])
        out.setdefault(name,{})[N]=r["base"]
        print(f"  {name} want {want:8s} base {r['base']:.4e}  sc {r['sc']:.2e}"
              f"  maxdisp {r['maxdisp']:.4f}  {time.time()-t0:5.1f}s", flush=True)
    json.dump(out, open("scan_N.json","w"), indent=1)
print("\n=== base residual vs N ===")
Ns=sorted({n for v in out.values() for n in v})
print("config              "+"".join(f"{n:>12d}" for n in Ns))
for name,v in out.items():
    print(f"{name}  "+"".join(f"{v.get(n,float('nan')):12.3e}" for n in Ns))
