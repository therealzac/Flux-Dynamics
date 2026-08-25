import numpy as np, torus_gf as e, time, sys, json
CFG=[("polar+1eq        ",["polar","eqA1"],"legal"),
     ("polar+2eq sameax ",["polar","eqA1","eqA2"],"legal"),
     ("4eq no polar     ",["eqA1","eqA2","eqB1","eqB2"],"legal"),
     ("polar+2eq diffax ",["polar","eqA1","eqB1"],"illegal"),
     ("polar+3eq        ",["polar","eqA1","eqA2","eqB1"],"illegal"),
     ("polar+4eq        ",["polar","eqA1","eqA2","eqB1","eqB2"],"illegal")]
out={}
for N in [int(x) for x in sys.argv[1:]]:
    t=e.Torus(N); nl=e.SparseVarCell(t); P=e.cell_pairs(t)
    print(f"\n=== N={N}  {t.nnodes} nodes  defect concentration 1/{t.nnodes}  (FREE BOX) ===",flush=True)
    for name,slots,want in CFG:
        t0=time.time(); r=nl.solve([P[s] for s in slots])
        out.setdefault(name,{})[N]={"base":r["base"],"box":r["boxstrain"],"want":want}
        print(f"  {name} want {want:8s} base {r['base']:.4e}  boxstrain {r['boxstrain']:.4e}"
              f"  sc {r['sc']:.1e}  {time.time()-t0:6.1f}s",flush=True)
        json.dump(out,open("scan_free.json","w"),indent=1)
