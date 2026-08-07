# The traversal problem — formal statement

**2026-08-07.** Zac's statement, written down so solver and physics agree.

## The object

The augmented stella octangula: an inner octahedron with a regular tetrahedron
stacked on each of its 8 faces. 36 unit-length edges. Crossing one costs 1 tick.

**Equator** — 4 of the octahedron's vertices, forming a unit square. Its
directions are X and Y.

**Poles** — the other 2 octahedron vertices, off the equator.

**Tet apexes** — the 8 stacked tetrahedra, one per octahedron face.

At each pole there are X and Y edges intersecting at exactly 90 degrees.
Each pole carries 4 of them, 8 in all — one per tetrahedron.

## Activation

Traversing a **pole-connected X or Y edge activates the tetrahedron carrying
that edge**. A tet stays active until a different one is activated, i.e. until
another pole X or Y edge is traversed. Exactly one tet is active at a time.

Activation is a property of the walk alone. The walk actualizes what it
traverses, so nothing need be assumed about the surrounding lattice.

A tet apex has three edges: one to its pole (activating) and two to equator
vertices (not). So a walk may pass through an apex without activating its tet.

## Turn classes

Set by the chord between the two neighbours used, in edge units:

| chord | turn | |
|---|---|---|
| 2     | 0   | same direction two ticks running |
| 1     | 60  | |
| sqrt2 | 90  | |
| sqrt3 | 120 | **refused** — nothing above 90 |

## Requirements

Design a closed edge-wise traversal of the 36 edges such that:

1. **Even tet time.** All 8 tetrahedra are active for the same number of ticks.
2. **Nucleon identity.** 2/3 of steps are 90 and 1/3 are 60 — and any window of
   N moves must average 2/3, so it holds at every scale, not just globally.
   0-turns count toward neither total.

Coverage of all 36 edges, and evenness of edge use, break ties.

## Consequences worth stating

**Even tet time is structural if activations are evenly spaced.** A tet is
active from its activating edge until the next one, so 8 activations spaced L
apart give every tet exactly L ticks, exactly. No fitness term needed.

**Poles are where the 90s live.** Entering a pole on one activating edge and
leaving on another is a 90 turn — 48 of the 72 ordered pairs, 0 for the other
24, never 60.

**A turn at a tet apex is always 60.** Its three edges run to the three corners
of one equilateral face, so its neighbours are mutually unit-distance and the
chord is 1 whichever pair is used. Unavoidable when an apex is visited — but an
apex need not be visited to route the walk, only to activate its tet.

---

## Colour

Each actualized tetrahedron carries a colour: **red, green or blue**.

Each octahedron vertex carries one of six: **red, green, blue** or their antis
**cyan (anti-red), magenta (anti-green), yellow (anti-blue)**.

When an actualized tet is given a colour, the pole it touches takes that same
colour and the opposite pole takes the anti. The four equator vertices then
take the remaining four colours.

So antipodal vertices are always a colour/anti pair, and a colouring is a
bijection from the octahedron's three axes to the three pairs
`(R,C) (G,M) (B,Y)`. The pole axis is forced — its pair is the one whose
primary is the tet's colour, oriented so the touching pole is the primary. The
two equator axes take the remaining two pairs: 2 ways to allocate them and 2
orientations each. **3 tet colours x 8 arrangements = 24 colourings per
activation.**

### Requirements

4. **Tet colour balance.** Over any window of N ticks, an actualized tet sees
   red, green and blue in equal measure.
5. **Six-fold vertex balance.** Over any window of N ticks, each octahedron
   vertex spends equal time as red, green, blue, cyan, magenta and yellow.

### Consequences

Colour is constant for the whole of an activation, so with 8 equal activations
per lap the colouring is a choice of one of 24 options per activation and
nothing finer. Each tet is activated once per lap, so requirement 4 needs a
period of at least **3 laps** — a tet cannot see three colours equally in one.

Each pole is touched by 4 of the 8 tets. When a tet on the far pole is active a
pole takes an ANTI colour, so requirement 5 pushes the tet order to alternate
between poles: that is what lets a pole spend equal time primary and anti.

Over K laps there are 8K activations and each vertex must take each of 6
colours equally, so 8K must divide by 6 — **K divisible by 3**, with K=3 the
smallest candidate: 24 activations, each vertex each colour 4 times.
