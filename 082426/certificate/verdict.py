import time, collections
from oracle import legal
CAND=[[102,179],[103,180],[136,137],[137,215],[137,144],[137,138],[143,144],[144,222],
      [144,145],[172,179],[173,180],[179,256],[179,186],[179,180],[180,257],[180,187],
      [214,215],[215,222],[215,216],[221,222],[222,223]]
rods=lambda m:[tuple(CAND[b]) for b in range(len(CAND)) if m>>b&1]
UND=[30,60,120,180,550,585,676,774,778,833,840,900,904,2097,2594,2625,2818,2820,3077,
     3105,3137,3232,3330,3392,4680,5192,8769,8833,10753,10754,11265,65618,65624,65684,
     65712,65744,65816,65936,66113,66180,66192,66240,66306,69792,98436,131236]
LEG=[92,688,771,788,802,912,1608,2086,2136,2224,2817,2824,2832,3168,3840,4704,7170,8707,
     8770,8961,8962,10274,65606,65610,65612,65634,65681,65682,65688,65704,65730,65732,
     65760,65809,65858,65864,65888,66752,73794,73800,81988,156,792,834]
for name,lst,expect in (("PROVEN LEGAL (control)",LEG,'LEGAL'),("UNDECIDED",UND,'?')):
    t0=time.time(); c=collections.Counter(); mins=[]
    for m in lst:
        r=legal(rods(m)); c[r['verdict']]+=1
        if r['verdict']=='INFEASIBLE': mins.append(r['minSep'])
    dt=time.time()-t0
    print(f"{name:26s} n={len(lst):3d}  {dict(c)}   {dt:.0f}s  ({dt/len(lst):.2f}s each)")
    if mins: print(f"     INFEASIBLE minSep range {min(mins):.4f} .. {max(mins):.4f}")
    if expect=='LEGAL':
        print(f"     CONTROL: all {len(lst)} are PROVEN legal by the engine -- oracle must agree")
    print()
