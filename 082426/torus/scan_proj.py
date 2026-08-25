import numpy as np, torus_gf as e, time, sys, json
CFG=[("polar+1eq        ",["polar","eqA1"],"legal"),
     ("polar+2eq sameax ",["polar","eqA1","eqA2"],"legal"),
     ("4eq no polar     ",["eqA1","eqA2","eqB1","eqB2"],"legal"),
     ("polar+2eq diffax ",["polar","eqA1","eqB1"],"illegal"),
     ("polar+3eq        ",["polar","eqA1","eqA2","eqB1"],"illegal"),
     ("polar+4eq        ",["polar","eqA1","eqA2","eqB1","eqB2"],"illegal")]
out={}
for N in [int(x) for x in sys.argv[1:]]:
    t=e.Torus(N); nl=e.ProjectedCell(t); P=e.cell_pairs(t)
    print(f"\n=== N={N}  {t.nnodes} nodes  (BAIN CHANNEL PROJECTED OUT) ===",flush=True)
    bad=0
    for name,slots,want in CFG:
        t0=time.time(); r=nl.solve([P[s] for s in slots])
        got="legal" if r["legal"] else "illegal"
        if got!=want: bad+=1
        out.setdefault(name,{})[N]=r["base"]
        print(f"  {name} want {want:8s} got {got:8s} base {r['base']:.4e}"
              f"  strain {r['strain']:.3e}  bain {r['bain']:.1e}  {time.time()-t0:5.1f}s"
              +("  <-- MISMATCH" if got!=want else ""),flush=True)
        json.dump(out,open("scan_proj.json","w"),indent=1)
    print(f"  mismatches: {bad}/6",flush=True)
