# Traversal sense maps: polarizations, the rhombic dodecahedron, and why A/B are special

**2026-08-06.** Working session, Zac + Claude. All numbers below are measured on the
L2 lattice (339 nodes, 1392 tets, 618 octahedra) unless marked as arithmetic.

---

## 1. What started it

The nucleon CA needs 90° turns. A proton is two 90° and one 60° per three turns; a
neutron is two 60° and one 90°. Stepping a proton through by hand, it stalled on
move 4: the block owed a 90°, the only 90° on the board was a base hop back toward
the start node, and **chirality alone forbade it**.

Measured at that position: legs 1, 1, 1.4895 — the far pair at √2 — so **96.3°**, a
right angle at the relaxed board's tolerance. Direction `-1,1,1`. Not a shuttle
(prev was elsewhere), balance counter at zero. The only bar was that `-1,1,1` is not
in chirality set A.

## 2. An octahedron has three great squares, not one

- **the equator** — 4 shortcuts, closed ring, defines the flux mode/generation
- **two apex squares** — each uses both apexes and an opposite pair of ring
  vertices, joined entirely by base edges

Measured on the canonical oct (ring `0,31,36,3`, apexes `13,14`), all twelve angles
exactly 90°:

```
equator        (2,0,0) (0,2,0) (-2,0,0) (0,-2,0)          shortcuts
AC square      (-1,-1,1) (1,1,1) (1,1,-1) (-1,-1,-1)      apex 13 -> 0 -> 14 -> 36
BD square      (1,-1,1) (-1,1,1) (-1,1,-1) (1,-1,-1)      apex 13 -> 31 -> 14 -> 3
```

Chirality governs base directions only, so it never touches the equator. The apex
squares are where it bites.

## 3. Winding cannot be encoded as a set of directions

Each apex square is **antipodally closed**: the AC square's directions are
±(1,1,1) and ±(1,1,-1). Going round backwards uses *the same four directions* as
going round forwards, in reverse order.

Therefore **any direction set that permits one winding permits both**, and no subset
of the eight base directions can express a winding sense. Closing a square as a
cycle needs both senses of both its axes, which contradicts "one sense per axis"
outright.

So closing the square is not the goal. **Sustaining 90° motion is.**

## 4. The sixteen sense maps: fourteen polarizations, two apolar

A sense map fixes one sense per base axis — 2⁴ = 16 in principle. The ones that
*cohere* are those expressible as `sign(d · n)` for some vector n: a **polarization**.

Four central planes cut the sphere into 14 regions, so **14 of the 16 are
polarizations and 2 are not**. The two exceptions are forced: if Σd = 0 then
Σ(d·n) = 0, so no n can make all four positive.

**Measured, exhaustively over all 16: 14 hemispherical, 2 not — and the 2 are exactly
the zero-sum sets A and B.**

Six of the fourteen polarize along a cardinal axis, one antipodal pair per axis:

```
+X  1,1,1   1,1,-1   1,-1,1   1,-1,-1        -X  (negated)
+Y  1,1,1   1,1,-1  -1,1,-1  -1,1,1          -Y  (negated)
+Z  1,1,1  -1,-1,1   1,-1,1  -1,1,1          -Z  (negated)
```

The other eight are oblique.

### Zac's labelling is the -Z map

For the XZ oct at centre (1,0,7) — ring `131,200,201,132` at y=0, apexes
`162,169` along ±Y:

```
a+ = (-1, 1,-1)   a- = ( 1,-1, 1)      axis ±(1,-1, 1)
b+ = ( 1, 1,-1)   b- = (-1,-1, 1)      axis ±(1, 1,-1)
c+ = (-1,-1,-1)   c- = ( 1, 1, 1)      axis ±(1, 1, 1)
d+ = ( 1,-1,-1)   d- = (-1, 1, 1)      axis ±(1,-1,-1)
```

Every "+" has dz = -1; every "-" has dz = +1. The plus set is a single half-space —
the **-Z** polarization, with **+Z** as its partner. That is the "north points the
same as south, east the same as west" rule, stated as a linear functional.

## 5. The zonotope reading (credit: Zac's research note)

Standard zonotope fact: for Z = Σ[−dᵢ, dᵢ], the **vertices of Z biject with the topes
of the associated hyperplane arrangement**, via ε ↦ Σ εᵢ dᵢ. Two Scale Bridges Part I
already has KRD = the zonotope of the four FCC base directions.

So the 14 regions and the 14 rhombic-dodecahedron vertices are **not two facts that
agree — they are one fact counted twice.**

Verified by direct computation over all 16 sign patterns on the zero-sum frame A:

| sign pattern | count | image | RD vertex |
|---|---|---|---|
| one-vs-three | **8** | ⟨111⟩, \|v\| = 2√3 | 3-valent (acute) |
| two-two | **6** | ⟨100⟩, \|v\| = 4 | 4-valent (obtuse) |
| all-same | **2** | origin | **not a vertex** |

Ratio obtuse/acute = **1.154701 = 2/√3 exactly**, the correct RD acute-to-obtuse
vertex ratio. The two all-same patterns are `++++` = **A** and `----` = **B**.

Three consequences:

1. **The polarization *is* the momentum.** Σ εᵢ dᵢ is literally the net displacement
   of taking one step in each allowed sense. "Choose a coherent handedness" and
   "choose a corner of the node's own Voronoi cell" are the same act, and the vertex
   you land on is the drift it produces.
2. **A and B land on the centre.** They are not vertices; they are the origin.
   "Momentum-free" stops being a nice reading and becomes the literal geometric
   statement — the two apolar maps are the ones whose vector sum is the lattice node
   itself.
3. **The 8+6 is the bridge-lattice 8+6** — eight ⟨111⟩ body diagonals, six ⟨100⟩
   cube axes, the |M| = 8 / next-shortest = 6 structure posed on 3 Aug.

**Caution, and it matters.** The bijection means this classification carries no
information the zonotope did not already have. It is a restatement, not
corroboration — elegant, and it confirms the objects were the right ones, but the 14
must **not** be logged as independent support for the RD's centrality. It is the same
fact wearing a second hat. Where it earns its keep is the dynamical reading: that a
sense map *has a vertex*, and that exactly two have none.

## 6. The two chiralities, L and R

Colour the four base axes and take "toward" to mean a positive component along
cardinal up:

```
a red  (1,1,1)      b green  (1,1,-1)
c blue (1,-1,1)     d yellow (1,-1,-1)
```

```
L :  red toward, blue away, green toward, yellow away
R :  red toward, blue away, green away,   yellow toward
```

Flipping all four is not a third option — it is the same object seen upside-down,
the antipodal vertex.

**Why these two.** The apex squares are AC (red/blue) and BD (green/yellow). A
square can only be circulated if one of its axes points toward and the other away —
rise on one, fall on the other. Do that for **both** squares and the map's vector
sum lands on a **4-valent ⟨100⟩ obtuse vertex**: L sums to (4,0,0), R to (0,0,4),
both magnitude 4. Let a square's two axes agree instead and the sum lands on a
**3-valent ⟨111⟩ acute vertex**, (2,±2,2), magnitude 2√3 — and those cannot
circulate.

So **chirality is the relative orientation of the two apex squares**, same or
opposite. It has nothing to do with which generation the particle is in.

## 7. Electron test

The electron now draws L or R at spawn and holds it for life. Audited over
**741 ticks / 12 runs**, chirality drawn 7×L / 5×R:

| check | result |
|---|---|
| chirality violations | **0** |
| shuttling (a-b-a) | **0** |
| generation breaks | **0** |
| ticks without a closed tet | **0** |
| banned (Y) rods installed | **0** |
| vacuum refusals / broken tets | **0 / 0** |
| turns | **741 / 741 at 60°** |
| loops closed | **12 / 12** |
| **loop steps skipped** | **0** |
| collapsed to bare rail | **0** |
| mean angle to line | **2.48°**, best 0.42° |

The loop is replayed verbatim — no step skipped, nothing re-derived mid-run.
2.48° is the best this CA has recorded; it ran at ~5° under A/B.

Separately, L and R are **exactly degenerate**, as they must be: over 12 shared
lines both closed 12/12 with identical loop lengths line-for-line
(`30, 30, 42, 34, 14, 42, 42, 22, 16, 12, 30, 34`), mean angles 3.76° and 4.24°.
Same lines, same loops, mirrored windings.

### Two bugs found getting there — both mine, both in the hop table

Zac's prediction was that X and Z must be degenerate for an XZ electron (Y being
the banned axis) and that L/R should not affect traversal statistics. Both were
true of the physics and false of the code. The asymmetry was a symptom to fix,
not a finding to explain.

1. **The approach walk was capped at one base hop.** A and B always reach a rod
   endpoint in one, so the cap was invisible; a polarization often needs a detour
   to arrive from a legal quarter. Same shape as gating `tetWalk` on tet
   membership. Fixing it took A/B from 24 to 40 table entries — they had been
   running under the cap too.
2. **The add order was hardcoded X-rod-first.** `m.b` is listed [X rod, Z rod] and
   the plan always added `b[0]` first. Which rod goes first is a free choice of
   route, not a property of the move, and hardcoding it broke X↔Z equivariance —
   costing L exactly its four ±X steers while R lost none. Trying both orders
   fixed it.

Entry counts before → after, and the equivariance check `L|U|d ≡ R|D|swapXZ(d)`
now passes with **zero mismatches**:

| map | before | after | empty moves |
|---|---|---|---|
| A, B | 24 | **68** | 0 |
| L | 18 | **40** | 0 |
| R | 22 | **40** | 0 |
| X±, Z± | 18 / 22 | **40** | 0 |
| Y± | 8 | 12 | 2 (the rail opposing the polarization) |

**Superseded.** An earlier sweep on this page reported polarizations as
"reliably fragile" — A/B closing 12/12 while L/R closed 4/6 with collapses, and
±Y failing 5/6. Those numbers were produced by the two bugs above and should not
be cited. The mechanism offered for them (fewer table entries → fewer closable
multisets) was a plausible story asserted without test; the entry counts were
real but their cause was the capped walk, not the polarization.

## 8. Corrections logged

Two claims made during this session and then measured false. Recorded so they are not
repeated:

- **"There is a different chirality partition per generation, with XY and XZ sharing
  one and YZ differing."** False. That asymmetry was an artifact of the oct
  enumeration fixing both apex-square windings from an arbitrary `apex[]`/`ring[]`
  ordering, plus reading a pattern from n=1 per mode. Properly sampled: each
  generation has exactly **two** partitions, identical across every oct of that mode,
  and the separating invariants permute correctly — XY uses `dy·dz` and `dx·dz`, XZ
  uses `dy·dz` and `dx·dy`, YZ uses `dx·dz` and `dx·dy`. Each mode uses the two
  products involving its dormant axis and excludes the product of its own two axes.
- **"The polarization sign matters — Z+ works and Z− collapses."** False, or rather
  per-line accident. Over six lines neither sign is reliably better; the single-line
  test happened to pick one where Z+ closed and Z− did not.
- **"Start class (U vs D) predicts collapse."** False. Measured over 48 runs:
  Z+ collapsed 0/3 starting U and 1/2 starting D; Z− collapsed 1/2 starting U and
  0/2 starting D. No correlation.

## 9. Open

- **Which pair is the nucleon's?** Zac's ±Z polarizes along an *active* axis; the ±Y
  pair polarizes along the apex/dormant axis. In zonotope terms both are 4-valent
  ⟨100⟩ vertices of the same cell, so the choice is between two corners of the same
  rhombic dodecahedron — not between different kinds of object.
- **Why do polarizations lose entry states?** The counts (24 / 22 / 18 / 8) are
  measured but not explained. The dormant-axis case has a clean argument (all base
  edges of a tet share a dy sign); ±X losing exactly 4 steers does not yet.
- **Does the proton's move-4 stall clear** under a polarization rather than A/B? Not
  yet tested — that was the original motivation and remains open.
