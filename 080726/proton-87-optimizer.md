# proton 87 — the optimizer build

**2026-08-07.** Handoff note. State as of commit `28975c4`.

## Where things stand

`proton 87` (`window._XZPROTON`, loop id `proton-87`, in `sim/_tetline-scratch.js`)
walks a closed traversal of the stella octangula — an octahedron held permanently
with one tetrahedron actualized at a time. The walk is **generated once at spawn
and replayed verbatim**; nothing reacts at runtime. Every lap is byte-identical
from lap 1 onward (lap 0 differs only because it starts on the bare oct).

Two generators exist. `genWalk` is the old greedy one, kept as a fallback.
`genWalkExact` is the constructive one and is what runs.

| | dwell spread | ratio90 | lap |
|---|---|---|---|
| greedy, best | 8.5% relative | 0.656 | 133 |
| constructive, L=10 | **0** | 0.174 | 88 |
| constructive, L=16 | **0** | 0.44 | 136 |

Equal face time is **solved and structural**: eight segments of identical length,
one per face, and a segment runs from entering its apex to entering the next, so
equal segment length *is* equal dwell. Zero variance at every length.

The turn ratio is **not** solved: 0.44 against 2/3. That is now a search-quality
problem inside a correct structure, not a constraint conflict — which is the
whole point of having built it this way. The constraint can no longer be traded
away to flatter the objective.

## The structure the optimizer exploits

Given a face order, **segments are independent**. A segment is fully determined
by its entry node, and affects the next segment through exactly two things:

1. the node it exits on
2. which oct edges it consumed

That makes this an exact dynamic program rather than a search.

```
state       (segment index 0..7, entry node, oct-edge bitmask)
size        8 x 7 x 2^12  ~= 229k
transition  enumerate valid paths for segment i
            -> (exit node, edges gained, 90-count)
objective   maximise total 90s, require mask == all 12 at the end
```

Per segment, enumerate paths once and keep only the Pareto-dominant
`(exit, mask-gained) -> max 90s`. That collapses an enormous path count into a
small table. The DP over segment choices is then **exact**, not sampled.

The only remaining search is the face order: `8!/8 = 5040` cyclic orders, halved
to **2520** by reflection. Small enough to enumerate outright.

So the whole thing is: 2520 face orders x an exact DP each, giving the true
optimum ratio for a given segment length.

## Build order

1. **Per-segment path enumeration + Pareto table.** Independently testable: for a
   fixed apex, entry node and length, enumerate all legal segment paths and
   confirm the table dominates what the current code samples.
2. **The DP** across 8 segments with the coverage mask.
3. **Face-order enumeration** on top.
4. Compare against the committed constructive result (0.44 at L=16) — the
   optimizer must beat it or the build is wrong.

## Constraints a segment must satisfy

- moves only among the six oct vertices and its **own** apex (touching another
  apex hands the face over early)
- spends all three of its apex edges — needs two apex visits, since degree 3 and
  entering/leaving consume two ends
- ends adjacent to the next apex with a legal turn into it
- all turns 60 / 90 / 0; 120 refused
- across the eight segments, all 12 oct edges covered, and the assembled lap must
  cover all 36 edges of the compound and close legally

## Traps worth remembering

- **The freshness bonus swamped everything.** In the greedy generator it was 100
  against turn and dwell terms of at most 20, so coverage decided nearly every
  step and five rounds of weight-tuning were only breaking ties underneath it.
  Check magnitudes before tuning.
- **Budget the segment DFS.** Removing the early exit for a best-of search hung
  generation outright at L=18. Current settings: stop at 60 candidates once one
  is found, 400k node visits.
- **`0` means no change of heading**, chord 2 — reported as an interior angle of
  180 in earlier notes. Same thing, opposite convention.
- **`solids.tets` counts implied solids.** The engine's detector is geometric, so
  it reports many more tets than the xon builds; relaxation closes shortcut slots
  behind it. Don't read it as "how many tets we made".
- The lattice re-centres between sessions, so node coordinates move. The compound
  is found fresh each spawn.
