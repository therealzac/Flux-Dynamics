# The ⟨111⟩ network is invariant under a 2-sphere of finite deformations

**2026-08-24** · exact · this is why the torus formulation kept escaping

---

## Statement

The eight ⟨111⟩ base bonds are preserved **exactly**, at **finite** amplitude, by
every homogeneous deformation of the form

    A = R . diag(a, b, c)      with      a^2 + b^2 + c^2 = 3

R any rotation. Not approximately, not to second order — exactly, to machine
precision, for arbitrary members of the family.

## Proof

A bond vector is (s₁,s₂,s₃)·S with sᵢ = ±1 and S = 1/√3. Under A,

    |A s|^2  =  s^T M s  =  tr M + 2( s1 s2 M12 + s1 s3 M13 + s2 s3 M23 ),   M = A^T A

The sign-dependent part vanishes for all eight sign choices **iff M is
diagonal**; the remaining value is tr M, which equals the original 3 **iff
tr M = 3**. Writing M = AᵀA diagonal with trace 3 gives A = R·diag(a,b,c) with
a²+b²+c² = 3. ∎

A 2-sphere of bond-preserving finite deformations, times the rotations.

## The member that matters

Setting a = √3/2 makes the ⟨200⟩ segment along x exactly unit while every ⟨111⟩
bond stays exactly unit:

    diag( 0.866025, 1.060660, 1.060660 )        a^2+b^2+c^2 = 3.000000000000
    worst <111> bond error   0.000e+00
    <200> along x            1.000000000000
    strain along x           -(1 - sqrt3/2) = -0.133975

**That is exactly the box strain the free-box torus solver returned — 0.133975,
identical to five decimals at N = 4, 6 and 8.** It was not failing to converge.
It was finding this, correctly, every time.

## Why this is stronger than C′ = 0

C′ = 0 is the *infinitesimal* statement: the tangent to this manifold at the
identity. Differentiating a²+b²+c² = 3 gives da+db+dc = 0 — the traceless
diagonal, i.e. the Bain channel. So:

| | statement |
|---|---|
| C′ = 0 | the tangent space at the identity is nontrivial |
| **a²+b²+c² = 3** | **the whole curved manifold, at finite amplitude** |

The measured second-order stiffness confirms the distinction. A *traceless*
Bain strain of amplitude e leaves a residual of exactly e²/3 (measured
4.166e-04 at e = 0.03536) — because the traceless path leaves the sphere. The
sphere itself has **zero** residual at any amplitude.

## Why the projection failed

The projection removed the free *linear* directions — hydrostatic and the three
shears kept, traceless diagonal removed, Bain content driven to exactly 0. It
still leaked, because **a linear projection cannot remove a curved manifold.**
The tangent was right; finite deformations simply leave along the curvature.

## Consequence for the engine

A periodic cell with any shape freedom can always ride this 2-sphere to make a
shortcut unit-length **for free**. So legality on a torus is vacuous as long as
the cell can deform at all, and this does not dilute with N — a symmetry has no
restoring force, so its amplitude is never set by the defect's size. That was
the flaw in "scale N until the box strain starves."

Two formulations survive:
* **fix the cell completely** — but that also forbids the hydrostatic relaxation
  the defect legitimately needs, and smears a uniform residual over every bond
  (measured: mean 1.8e-3 in every shell, only 5% within 1.5 of the rod)
* **pose the question locally** — which is what the PBD ball does, and why it
  was answering correctly all along

## The physical statement

**A shortcut costs nothing if the whole universe performs it, and costs
something only because it is local.**

Contracting every ⟨200⟩ segment along one axis *is* a move along this manifold —
free, exact, and finite. It is the Bain path, BCC → FCC, and it is a symmetry of
the base network rather than a deformation of it. What makes a shortcut an
excitation is not the contraction; it is that the rest of the lattice does not
come along.

## Open

* Does the 2-sphere have a role beyond an obstruction? It is an exact continuous
  symmetry of the vacuum that the framework has not named.
* The six ⟨110⟩ kernel lines (see `observation_floppy_modes.md`) are the k ≠ 0
  mechanisms; this 2-sphere is the k = 0 one. Whether they are two faces of one
  object has not been checked.
* Formulate L3 locally: a finite region solved exactly with the far field matched
  to the analytic Green's function, so the global manifold is unavailable by
  construction.
