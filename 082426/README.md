# 2026-08-24 — Rule-based lattice legality, and the KRD↔BRD strain

Two threads running in parallel.

**Thread A — can a rule replace the solver?** Testing whether a purely local,
integer rule on shortcut configurations reproduces the vacuum's legality
verdicts. If it does, geometric vetoes become O(1) per shortcut and 100k-node
lattices open up. See the sufficiency notes below.

**Thread B — the KRD↔BRD strain.** Whether the crystallographic↔icosahedral
strain of the two-scale-bridges paper can source the orthorhombic field that
lifts the Snoek degeneracy.

---

## Index

| doc | what |
|---|---|
| [observation_krd_brd_strain_1plus1plus1.md](observation_krd_brd_strain_1plus1plus1.md) | The KRD→BRD strain has a unique spectrum (2^(−1/6)/φ, 2^(1/3), 2^(−1/6)φ), is diagonal in the BCC bipyramid's own frame, and splits it 1+1+1. Six states per cell. |
| [krd_brd_strain.py](krd_brd_strain.py) | Reproduction script for the above. |
| [torus/observation_floppy_modes.md](torus/observation_floppy_modes.md) | The ⟨111⟩ network has exactly **6N−3** zero modes on an N³ torus, supported on **six ⟨110⟩ lines** in k-space. Independently confirms the ⟨110⟩ displacement planes measured in `081626/` — all four measured normals are kernel lines, every angle matches to <1.1°. |
| [observation_rule_is_sound_not_complete.md](observation_rule_is_sound_not_complete.md) | **Exhaustive, 578 classes.** The local rule never accepts an illegal configuration (sound) but refuses to reject 21% of illegal 4-rod ones (not complete). Also: the veto range is **finite** — pull an illegal pattern apart and it becomes legal — which retracts the argument that no finite local rule can exist. |
| [torus/observation_generation_swap_confirms_kernel.md](torus/observation_generation_swap_confirms_kernel.md) | **Falsification test, passed.** Predicted from ker Φ that switching XZ→XY generation sends ⟨110⟩/⟨1−10⟩ dark and lights ⟨101⟩/⟨10−1⟩. It did, on all six lines, 0.00–0.73° off ideal. **The displacement planes ARE the kernel of Φ.** |
| [torus/observation_why_the_inplane_pair_is_dark.md](torus/observation_why_the_inplane_pair_is_dark.md) | **Derived.** An oct centre lies ON a ⟨110⟩ lattice plane exactly for the pair in its own equatorial plane (parity argument, verified all three generations), so a ±0.5 slab catches one plane there and two elsewhere. Includes a **correction**: the in-plane family is excited — most strongly, in fact — it just has no slab signature. "Dark" ≠ "unexcited". |
| [torus/observation_floppy_manifold.md](torus/observation_floppy_manifold.md) | The k=0 half: an exact **2-sphere of finite deformations** A = R·diag(a,b,c), a²+b²+c²=3, preserving every ⟨111⟩ bond. Explains why a torus formulation cannot answer legality, and gives the physical statement: **a shortcut costs nothing if the whole universe performs it.** |

---

## Method notes

* **Never trust a single solve.** The engine's own comment warns that "a false
  'illegal' is invisible — it looks exactly like a real violation." This turned
  out to cut *both* ways and cost a retracted result (below). Every legality
  verdict must be chained: re-solve with `{here:true}` from the previous state,
  repeatedly, until either the residual crosses TOL or it stops moving.
* **Discriminate by response to chaining, not by residual size.** Measured on
  the central cell, the two populations separate cleanly:
  * false negatives sit at ~2.5e-5 and **cross TOL in one extra round**
  * true violations sit at ~0.1 and **do not move in 25 rounds** (one drifted
    *up*, 0.09656 → 0.09933)
* **U is invariant under post-rotation.** For strain work, `M' = RM` gives
  `M'ᵀM' = MᵀM`, so enumerating target orientations is unnecessary — enumerate
  generator subsets and pairings only.

---

## Two retractions, in order

### 1. "32/32 perfect agreement" on the central cell — RETRACTED
Built on single-solve verdicts. The engine's own comment warns that "a false
'illegal' is invisible." True, but the fix has to be applied carefully — see
below for how getting that wrong produced a second, worse error.

### 2. "Face-sharing tetrahedra are legal" — RETRACTED
**This was wrong.** It rested on a single chained `{here:true}` continuation
crossing TOL, which is *not* a test of whether a configuration has a legal
packing. A chained solve resumes from an already badly-strained state (max node
displacement 0.57, over half an edge length) and asks a different question.

The correct test is a **fresh solve from REST**, and it must be run in both
sweep orders, because the engine canonicalises node indices and the two frames
are different members of the solution set:

| configuration | canonical frame | raw/identity frame | mean node displacement |
|---|---|---|---|
| polar + 1 equatorial (one tet) | 6.1e-11 ✓ | 7.4e-12 ✓ | 0.047 |
| polar + 2 equatorial, **same axis** (edge-sharing) | 9.5e-11 ✓ | 9.5e-11 ✓ | 0.073 |
| polar + 2 equatorial, **different axes** (face-sharing) | 2.476e-5 ✗ | **1.898e-5 ✗** | **0.251** |
| polar + 3 equatorial | 0.09656 ✗ | — | — |
| polar + 4 equatorial | 0.1501 ✗ | — | — |

Face-sharing refuses in **both** frames, from REST, after the full 66,093-sweep
budget, and demands 3.4× the mean lattice deformation of the legal edge-sharing
case. The geometry it strains toward is the correct one — apex-to-apex measures
1.632981 against the exact 2√2/√3 = 1.632993 for two face-sharing unit tets — so
the solver understood the request. The vacuum simply will not close it.

**So the rule stands exactly as originally stated, both clauses:**

> For every active shortcut, at most **2** of the 4 equatorial shortcuts of its
> implied cell may be active, and if 2, they must lie on the **same ⟨100⟩ axis**.

### What the chained solve was actually doing — it reconstructed the packing

Contact-topology diff on the chained run, counting non-base pairs at touching
distance (< 1.02) before and after:

| | at REST | after chaining |
|---|---|---|
| non-base contacts | **0** | **355** |
| contacts gained | — | **355** |
| contacts lost | — | 0 |
| max node displacement | — | 0.575 (over half an edge) |
| mean node displacement | — | 0.251 |

Zero at rest is correct: in the BCC void lattice the base ⟨111⟩ edges sit at 1
and the next distance up is 2/√3 = 1.1547, so nothing else is touching. After
the chained solve, **355 new contacts exist that the lattice does not have.**

The residuals look perfect — base 1.004e-7, sc 2.95e-8, minSep exactly 1, and
the two tets are regular to six figures with apex-to-apex at 1.632993, the exact
value. Every number passes. But the lattice has **densified into a different
packing**. That is a phase change, not an excitation, and it is precisely the
Kepler-density failure mode: the constraint set was satisfied by rebuilding the
structure rather than by deforming it.

This is why a chained continuation can never establish legality. Satisfying
"base edges unit, shortcuts unit, nothing overlapping" does not pin the
structure — many packings satisfy it. Only a fresh solve from REST asks whether
*this lattice* can host the configuration.

### The structural reason, which should have been the prior
An actualised tet is a local FCC close-packing. The FCC void graph is bipartite
tet/oct: every triangular face borders exactly one tetrahedron and one
octahedron, never two of a kind, and the tet sector is chirally bipartite on top
of that. Two tets sharing a face requires two same-type voids adjacent, which
that graph forbids. `082026/BCC Bipyramid.html` reaches the same veto from the
geometry side (2×70.529° + 2×109.471° = 360° around an edge — the two tets sit
opposite each other, separated by the octahedra).

### Method rule this establishes
**Never promote a chained continuation to a legality verdict.** Chaining is for
diagnosing whether a *refusal* is a plateau or a slow tail. A configuration is
legal iff a **fresh solve from REST converges**, checked in both frames. The two
questions are not the same and conflating them cost a retracted result.

## Open threads

* **Re-run the multi-cell sufficiency sweep with chaining.** The 3,937-config
  sweep over the 21 shortcut candidates within radius 1.0 was built on
  single-solve verdicts and inherits the same flaw. Both apparent counter-
  examples (v1, v2 — three shortcuts in mutually non-overlapping cells) are
  still unresolved; v1 decays as t^−0.78 (r² = 0.99992) and had reached 4.86e-6
  after 40 chained rounds without crossing.
* **Is the corrected rule (count only, no axis condition) sufficient?** Untested
  at multi-cell scope.
* **Why does the tetrahedral-octahedral-honeycomb argument fail here?** In that
  honeycomb every face borders one tet and one oct, never two of a kind. The
  lattice apparently does not enforce it.
* Thread-B threads are listed in the observation doc.
