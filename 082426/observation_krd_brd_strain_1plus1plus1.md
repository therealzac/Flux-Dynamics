# KRD → BRD: the strain is 1+1+1 in the bipyramid's own frame

**2026-08-24** · exact linear algebra, exhaustive over all realisations
· reproduction: [`krd_brd_strain.py`](krd_brd_strain.py)

---

## Statement

Let **M** be any linear map carrying the four generators of the Kepler rhombic
dodecahedron onto four generators of a Bilinski rhombic dodecahedron. Then:

1. The pure-strain part of **M** has a **unique** spectrum, independent of every
   choice made in constructing it.
2. That strain is **diagonal in the principal frame of a BCC bipyramid** — the
   polar ⟨100⟩ axis together with the two equatorial diagonals ⟨110⟩.
3. On that frame it splits **1 + 1 + 1**: three distinct stretches, no degeneracy.

All three are exact and were checked over the complete set of 360 realisations.

---

## Setup

| | generators | pairwise cos | face |
|---|---|---|---|
| **KRD** | the four ⟨111⟩ unit vectors (tetrahedral 3-fold axes, T ⊂ W(A₃)) | 1/3 | √2-rhombus |
| **BRD** | four of the six icosahedral 5-fold axes (I ≅ A₅ ⊂ S₅ = W(A₄)) | 1/√5 | golden rhombus |

`M` is fixed by `M vₖ = sₖ aₖ`. Because the KRD generators satisfy Σ vₖ = 0, the
scale factors are not free: applying **M** gives Σ sₖ aₖ = 0, so **s** is forced
to be the null vector of the 3×4 matrix [a₁ a₂ a₃ a₄]. **M** is then determined
by three of the four conditions and the fourth is a consistency check (it holds
to 1e-8 in every accepted case).

**Enumeration.** All C(6,4) = 15 subsets of the icosahedral axes × all 4! = 24
pairings = **360 valid maps**. Each normalised to det **M** = 1, then polar-
decomposed **M** = **RU** with **U** = √(**M**ᵀ**M**).

> **The enumeration is complete up to rotation of the target.** If **M**′ = **RM**
> then **M**′ᵀ**M**′ = **M**ᵀ**RᵀRM** = **M**ᵀ**M**, so **U** is unchanged by any
> reorientation of the BRD relative to the cube. The spectrum below is therefore
> an invariant of the KRD↔BRD *relation*, not an artefact of how the icosahedron
> was placed.

---

## Result 1 — the spectrum is unique and closed-form

All 360 realisations give the same principal stretches:

| | value | closed form |
|---|---|---|
| λ₁ | 0.550606 | **2^(−1/6) / φ** |
| λ₂ | 1.259921 | **2^(1/3)** |
| λ₃ | 1.441504 | **2^(−1/6) · φ** |

* det = λ₁λ₂λ₃ = 1 exactly (volume-preserving).
* λ₃/λ₁ = **φ²** = 2.618034 exactly.
* The two metallic means of the Coxeter spine appear separately: the **2** of
  A₃ (h = 4) sits on the middle stretch, the **φ** of A₄ (h = 5) on the outer
  pair. λ₂ = 1/(λ₁λ₃) = 2^(1/3) ties them.

---

## Result 2 — the frame is the bipyramid's, and the split is 1+1+1

**360 of 360** realisations are diagonal in some BCC-bipyramid frame
{polar ⟨100⟩, equatorial diagonal ⟨110⟩, equatorial diagonal ⟨110⟩}. The
assignment is universal:

```
polar <100> axis          ->  2^(1/3)          = 1.259921
equatorial diagonal  A    ->  2^(-1/6) * phi   = 1.441504
equatorial diagonal  B    ->  2^(-1/6) / phi   = 0.550606
```

Three distinct stretches on the cell's three principal axes. **This is the
1+1+1.** It is not approximate and it is not a choice of basis made to produce
it — the principal frame of a symmetric tensor is canonical, and it lands on the
bipyramid exactly.

The bipyramid frame is the natural one here for a structural reason: those three
axes are precisely the ⟨100⟩ + ⟨110⟩ + ⟨110⟩ triple that the cell itself
distinguishes (one apex-apex axis, two equatorial diagonals), and no other
mutually orthogonal triple in the cell has that status.

---

## Result 3 — six states per cell

Each realisation labels a cell with an orientation *and* a chirality:

| polar axis | long equatorial diagonal | count |
|---|---|---|
| x | [y+z] | 60 |
| x | [y−z] | 60 |
| y | [x+z] | 60 |
| y | [x−z] | 60 |
| z | [x+y] | 60 |
| z | [x−y] | 60 |

**6 = 3 orientations × 2 chiralities**, evenly populated. The chirality is real
content, not a relabelling: the two equatorial diagonals receive *different*
stretches (φ vs 1/φ), so "which diagonal is long" is a physical distinction the
cell carries.

---

## Result 4 — the same tensor in other bases

The 1+1+1 above is the strain in its own frame. Projected onto other observables
the picture degenerates, and every degeneracy has the same cause:

| observable | result | φ survives? |
|---|---|---|
| 2 equatorial diagonals | ratio **φ²**; square equator → rhombus, cos = **√5/3** | **yes** |
| 5 ⟨100⟩ shortcut segments | **1 + 4**: polar 2^(4/3)/√3, four equatorial all **2^(1/3)** | no |
| 8 face-sharing neighbour ⟨110⟩ directions | 8-fold degenerate at **1.178548** | no |
| 3 cube axes | **1 + 2**, singlet/doublet ratio exactly **2/√3** | no |

**The cause, in one line.** φ² + φ⁻² = 3, since φ² = φ + 1 and φ⁻² = 2 − φ.
Any direction that samples the two equatorial diagonals with *equal weight*
averages λ₃² and λ₁² to 2^(−1/3)·3/2, which is rational — and the golden ratio
drops out. The four equatorial ⟨100⟩ edges do this (they bisect the diagonals),
the eight neighbour ⟨110⟩ directions do this, and the two transverse cube axes
do this. Only the diagonals themselves, which sample one or the other, keep it.

### The identities, checked exactly

```
phi^2 + phi^-2 = 3                              phi^2 = phi+1, phi^-2 = 2-phi
equatorial edge   (2/sqrt3) * sqrt3/2^(2/3)  =  2^(1/3)        exact
singlet/doublet   2^(1/3) / (sqrt3/2^(2/3))  =  2/sqrt3        exact  (the shortcut ratio)
equatorial rhombus  cos(theta) = sqrt5/3     =  cos(th_KRD)/cos(th_BRD) = (1/3)/(1/sqrt5)
```

The last one is worth flagging on its own: the strained equator's rhombus angle
has a cosine equal to the *ratio of the two generator-angle cosines* that define
the two solids.

---

## What is established and what is not

**Established** (exact arithmetic, exhaustive enumeration, no physics assumed):
Results 1–4. The spectrum, the frame, the six states, the closed forms and the
cancellation identity.

**Not established.** Which observable the physics couples to. The 1+1+1 lives on
the cell's principal axes; the shortcut segments — the things that actually carry
flux — see 1+4. Nothing here says which one sets an energy. That is the open
question, and it is a physics question, not a geometry one.

---

## Open threads

* **Does the vacuum select a state?** Six states per cell, evenly populated in
  the enumeration. Is the selection dynamical, or does the walk pick it?
* **Is the φ² equatorial rhombus visible in the solver?** Direct test: impose the
  strain on a relaxed lattice and measure the equatorial diagonals of a cell.
* **1+1+1 vs 1+4.** These are the same tensor read in two bases. If flux energy
  couples to the ⟨100⟩ segments the split is Bain; if it couples to the principal
  frame it is threefold. Deciding this decides whether the bridge can source a
  generation ladder.
* **The 6-fold label.** 3 orientations × 2 chiralities coincides with the six
  quark types (pu1, pu2, nd1, nd2, pd, nu). Coincidence of counting until a map
  is constructed.
