# The BCC ⟨111⟩ network has exactly 6N−3 floppy modes, and they live on six lines

**2026-08-24** · analytic, verified numerically for N = 2…16
· engine: [`torus_gf.py`](torus_gf.py) · tests: [`test_torus_gf.py`](test_torus_gf.py)

---

## Statement

On an N³-cell torus (2N³ nodes, periodic, no boundary), the dynamical matrix of
the unit-stiffness ⟨111⟩ base network has

    dim ker Phi  =  6N - 3

and Φ(**k**) is singular **exactly** on the set

    k_a = 0    and    k_b = ± k_c        ({a,b,c} = {1,2,3})

— six ⟨110⟩ lines through the origin, one pair lying in each coordinate plane.

Verified for N = 2, 3, 4, 5, 6, 8, 10, 12, 16: the predicate matches the
numerically-computed nullity at every one of the N³ wavevectors, and the total
matches 6N−3 exactly in each case.

---

## Derivation

Write the lattice as simple cubic 2ℤ³ with a two-site basis, site 0 at (0,0,0)
and site 1 at (1,1,1) — exactly the shared-parity integer set. Site 0 in cell
**n** bonds to site 1 in cell **n**+**d** for **d** ∈ {0,−1}³, the separation
being 2**d**+(1,1,1), i.e. all eight ⟨111⟩ sign combinations.

With E = ½ Σ (n̂·(u_i − u_j))², the k-space block is

    Phi(k) = [[ (8/3) I ,   -C(k)    ],
              [ -C(k)^H  ,  (8/3) I  ]]

    C(k) = sum_d  nhat_d nhat_d^T e^{i k.d}

using Σ_d n̂n̂ᵀ = (8/3)I, which holds because the eight directions are all sign
combinations and every cross term cancels. Setting p_i = 1+e^{−ik_i} and
m_i = 1−e^{−ik_i}, the sum factorises:

    C_aa = (1/3) p1 p2 p3
    C_ab = (1/3) m_a m_b p_c        (a != b, c the third index)

Φ is singular iff C has a singular value equal to 8/3.

**Set k_a = 0.** Then m_a = 0, so every off-diagonal entry touching *a* vanishes
and C block-diagonalises. The surviving 2×2 block on {b,c} is
(2/3)·[[p_b p_c, m_b m_c],[m_b m_c, p_b p_c]], with eigenvalues

    (4/3)(1 + e^{-i(k_b+k_c)})     and     (4/3)(e^{-i k_b} + e^{-i k_c})

whose moduli are (8/3)|cos((k_b+k_c)/2)| and (8/3)|cos((k_b−k_c)/2)|. Either
reaches 8/3 exactly when k_b ∓ k_c ≡ 0 (mod 2π).  ∎

---

## Why this matters

**1. It is C′ = 0, made exact.** The vanishing shear modulus is not a vague
softness — it is a precisely located set of *exact* zero modes, and their
location is the ⟨110⟩ shear within a coordinate plane, which is the Bain
channel. The earlier proof (⟨111⟩ bonds are blind to tetragonal ⟨100⟩ strain,
δl/l = e(⅓−⅓) = 0) is the k→0 limit of this statement.

**2. It predicts the displacement PLANES.** A mode supported on a *line* in
k-space transforms to a real-space field that does not decay along the conjugate
direction and decays only transverse to it. Six lines → planar, non-decaying
structure. This is very likely the same object the `081626/` programme has been
measuring empirically as displacement planes, and it says the planes should be
⟨110⟩-oriented. **Testable and currently untested.**

It also explains the non-decay measured at 6346 nodes: a single shortcut excites
these modes, and a line-supported mode has no 1/r² in it to begin with.

**3. The scaling is the interesting part.** 6N−3 is **O(L)**, not O(L³), not O(1):

| floppy dimension | meaning |
|---|---|
| O(1) | ordinary rigid crystal |
| **O(L)** | **this lattice** |
| O(L³) | fluid / unconstrained |

Sub-extensive but unbounded. The lattice is neither rigid nor floppy; it sits at
a marginal point, with one mechanism family per line.

**4. It makes the engine's legality test finite and exact.** Minimising ‖Bu‖²
subject to shortcut constraints Su = c has minimum **zero** — a legal
configuration — iff c lies in S(ker Φ). Since dim ker Φ = 6N−3 is small and its
basis is now known in closed form, that is a small explicit rank test rather
than an iteration that might not have converged.

---

## Status of the acceptance tests

| level | result |
|---|---|
| **L1 structure** | **PASS** — k-space and real-space spectra agree to 6.2e-15; Φ symmetric, PSD, translations in the kernel |
| **L2 physics** — C′ | **PASS** — base bonds change by 3.333e-07 = e²/3 exactly under Bain strain e = 1e-3 (second order); shortcuts change by 1.000e-03 = e (first order) |
| **L2 physics** — soft branch | **FAIL**, and left failing |
| **L3 verdicts** | not yet built |

**On the failing test.** `has_soft_branch` looks for a branch of Φ(k) scaling
faster than k² as k→0. Measured exponents are 0,0,0,2,2,2 along ⟨100⟩, ⟨110⟩
and ⟨111⟩ — three optical, three acoustic, nothing anomalous. The test encodes
the hypothesis that marginal stability shows up as a *fractional power-law
branch*. It does not. It shows up as **exact zero modes on a measure-zero set**,
which a radial scan through generic directions never touches.

The test is not being edited. It is recorded here as having encoded a wrong
hypothesis, and the correct test — nullity of Φ(k) equals 6N−3, supported on the
six ⟨110⟩ lines — is stated above and passes for N = 2…16. Replacing it requires
Zac's explicit consent.

---

## Open

* **Do the ⟨110⟩ lines match the measured displacement planes in `081626/`?**
  The prediction is specific and falsifiable.
* **Build L3.** Needs the constraint layer: c ∈ S(ker Φ) for the first-order
  test, then Newton on the exact lengths with Φ⁺ as preconditioner for the real
  verdict.
* **What do the floppy modes look like in real space?** The k-space basis is
  known; their real-space form has not been looked at.
* **Flat-band analogy — worth one line, not a doc.** Twisted bilayer graphene
  goes correlation-dominated at its magic angle because the bands go *flat*:
  near-zero dispersion, extensive near-degeneracy, kinetic term with nothing to
  say. Our kernel is the same category in a cleaner form — exactly flat, on six
  lines, with no tuning parameter. Possible use: the flat-band and
  rigidity-percolation literatures may already have the tools for "constraints
  against mechanisms," which is the question below. Filed as a lead, not a
  result.
* **Does 6N−3 survive shortcuts?** Every activated shortcut is one more
  constraint. 6N−3 mechanisms against a growing constraint count gives a
  concrete rigidity-percolation threshold — and a prediction for how many
  shortcuts a torus of size N can carry at once.
