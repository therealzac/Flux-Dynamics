# Generations are orientational variants of a tetragonal elastic dipole

**Date:** 2026-08-14
**Engine:** fd-lattice @ `2c733e5`, 360-node lattice, averaged/from-REST physics
**Status:** measured. The elasticity is textbook; the identification is the claim.

---

## The measurement

A generation in FD is which coordinate plane the octahedron's equator lies in:
XZ = 1 (electron), XY = 2 (muon), YZ = 3 (tau). Each plane has a **dormant
axis** — the normal — along which a mode-locked tet is the only direction it can
travel.

Tri-fermion proton, 48 activations (2 laps) in each generation. The quadrupole
moment tensor of the radial displacement field, diagonalised, principal axis
measured against that generation's own dormant axis:

| gen | dormant | angle to dormant (min / med / max) | angle to the other two | oct centre |
|-----|---------|------------------------------------|------------------------|------------|
| XZ  | Y | 2.42 / **3.13** / 3.35 | 86.9–89.9, 86.7–89.9 | `0, −0.577, 0` |
| XY  | Z | 2.44 / **3.04** / 3.24 | 86.8–89.9, 86.8–89.8 | `0, 0, −0.577` |
| YZ  | X | 2.38 / **2.97** / 3.25 | 87.0–89.9, 86.8–89.9 | `−0.577, 0, 0` |

Falloff exponents, medians over 48 activations, **identical across generations**:

    l0 −1.00    l1 −0.85    l2 −1.50    l3 −2.30    l4 −0.95

Anisotropy `|λ₁| / Σ|λᵢ|` = **0.500 in all 144 samples**, all three generations,
proton and lepton alike.

---

## What is definitional, and what is not

**Definitional.** The generations *are* the three coordinate planes, and the
dormant axis is by construction the normal to the plane. So once the quadrupole
aligns with the dormant axis, mutual orthogonality follows immediately — three
orthogonal planes have three orthogonal normals. That step is arithmetic.

**Not definitional, and the actual content:**

1. The quadrupole aligns with the dormant axis to **~3°**. It did not have to.
   It could have tilted 30°, or tumbled, or tracked the live quark instead.
2. The tensor is **always uniaxial, never triaxial** — see below.
3. The alignment holds to the same ~3° in all three generations, so no plane is
   privileged by the lattice.

---

## The 0.500 is not a mystery — it is what uniaxial means

Recorded here because it was twice written up as unexplained and it is not.

A traceless symmetric tensor that is uniaxial has eigenvalues `(2a, −a, −a)`.
Then

    |λ₁| / Σ|λᵢ| = 2a / (2a + a + a) = 2a / 4a = 1/2

**identically, for any `a`.** The metric can only ever return 0.500 for a
uniaxial tensor, and returns something else only for a triaxial one. So the
finding is *"the tensor is always tetragonal"*, and 0.500 is how that reads out
under this normalisation. Not a separate fact.

---

## The elasticity is textbook, and that is good news

The object being measured is an **elastic dipole tensor** `P_ij` — the standard
description of how a localised defect strains a host crystal, with relaxation
volume tensor `Ω_ij = S_ijkl P_kl`.

And "three mutually orthogonal variants" is classical defect physics: a defect
with **tetragonal symmetry in a cubic host** has exactly three orientational
variants, its axis along one of X, Y, Z. In bcc metals an interstitial C, N or O
sits at an **octahedral site** — tetragonal symmetry — and forms an elastic
dipole aligned with one of the three cubic axes. Three sublattices, mutually
orthogonal.

That is this structure, down to the octahedral site.

Consequences worth taking:

* `P_ij` has established computational definitions and known defect–defect
  interaction laws. They apply here unchanged.
* Variants **reorient under applied stress**. That is the Snoek effect, and it
  has a measurable signature: anelastic relaxation, read by internal friction.

---

## The claim

> Two particles of different generation occupying the same region have
> **mutually orthogonal strain quadrupoles**. Electron, muon and tau are the
> same defect in the three orientations the cubic vacuum admits.

This is *relational* and therefore frame-independent, which matters — see the
falsified reading below.

**ZW's corollary, and the sharpest part of the idea:** muon and tau decay to
generation 1 *is variant reorientation under stress* — the vacuum having locked
into one mode applies the stress that drives the transition. Flavour decay and
Snoek relaxation would be the same process. If that holds, the elasticity
literature supplies a rate.

---

## What this is not

**Not an electric quadrupole.** A spin-½ particle cannot have one — Wigner–Eckart
caps multipole order at 2j, so j=½ admits charge and magnetic dipole and nothing
else, identically zero by rotational symmetry. Electron, muon, tau, proton and
neutron are all spin-½. That constraint is fatal to an electromagnetic reading
and does **not** bite here: this is a quadrupole of the *lattice strain field*,
which couples gravitationally, not electromagnetically.

**Not generation-as-absolute-orientation.** If a muon simply *is* "the XY one",
then rotating a muon 90° turns it into a tau, and generation is direction
dependent in space. Nothing like that is observed — generations differ by mass,
not by which way they point. Taken literally that reading is falsified before it
reaches the quadrupole. The relational statement above is what survives, and it
requires the lattice orientation itself to be unobservable: the three planes
related by a vacuum symmetry, generation being which one a particle's condensate
happened to pick.

---

## Caveats

* **Two laps.** 48 activations per generation. The ~3° figure is stable across
  them but the run is short.
* **The 3° tilt is unexplained.** Suspiciously constant. Either a real tilt or an
  artefact of the 0.5-unit shell binning; a bin-width sweep would separate them.
* **Charged-lepton exponents are unusable.** A lepton is moving, so its shells
  sample a source that shifted during the crossing. Fits came back −37.9, 0.178,
  0.721, −12.9 — noise. Only the *stationary* proton gives a clean spectrum.
  Lepton orientation data is still usable: angles to the dormant axis are
  bimodal, clustering near 0° and near 90° with **nothing between 27° and 64°**
  (XZ 6/10 within 30°, XY 5/10, YZ 7/10).
* **Novelty unverified.** One search found no identification of elastic-dipole
  variants with lepton generations, but that is not a literature review.

---

## Open threads

* Bin-width sweep: is the 3° tilt real?
* Does a variant reorient under imposed lattice strain, and at what rate?
* Ten laps to firm up the numbers.
* The l=2 exponent is **−1.50**, not −2, and reproducible to two decimals in all
  three generations. A single tet gives −1.92. Why does a three-quark object
  fall off more slowly than one tet?

---

*See also:* `observation_proton_neutron_wobble_and_winding.md`,
`method_multipole_centre.md`
