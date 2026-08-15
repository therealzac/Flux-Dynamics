# The Snoek mapping: what it buys, and the degeneracy problem it creates

**Date:** 2026-08-14
**Status:** analysis, not measurement. Contains one sharp objection that the
model must answer, and one that turns into a prediction.

---

## The mapping

FD generations are the three orientational variants of a tetragonal elastic
dipole in a cubic host. In condensed matter that system has a name and a
century of data behind it: an interstitial C, N or O at an **octahedral site**
in a bcc metal, whose elastic dipole aligns with one of the three cubic axes.

What comes with it, unchanged:

| FD | Snoek system |
|----|--------------|
| generation | orientational variant |
| strain quadrupole | elastic dipole tensor `P_ij` |
| the vacuum locked into one flux mode | applied stress |
| muon / tau decay to gen 1 | stress-induced variant reorientation |
| flavour lifetime | **anelastic relaxation time** |

And the relaxation rate is not qualitative. *"The reorientation of interstitial
atoms (dipoles) is the elementary step of interstitial diffusion, and an
**Arrhenius equation** applies for the relaxation rate."* Different species give
peaks at different temperatures — a whole mechanical-loss spectrum — with
tabulated activation energies (C in Fe ≈ 25 kcal/mol, O ≈ 29 kcal/mol).

So the mapping supplies a **functional form for the lifetime hierarchy**:

    τ  =  τ₀ · exp( Q / kT )

An exponential in a barrier height. Small differences in `Q` give enormous
differences in lifetime, which is the right *shape* for a flavour hierarchy
spanning 2.197×10⁻⁶ s (muon) to 2.903×10⁻¹³ s (tau) — a factor of 7.6×10⁶,
which is only `ΔQ/kT ≈ 15.8`. Hierarchies that look absurd as ratios are
unremarkable as barrier differences. That is the attraction.

---

## Two hierarchies, and only one of them is the analogue

An earlier draft of this note said the mapping was "dead on arrival" because
variants are degenerate. That was too strong, and it was true only of the
**unstressed** case. Being precise, the Snoek literature contains two distinct
hierarchies:

**1. Across species — real, tabulated, and NOT the analogue.**
Carbon 25,000 cal/mol against oxygen 29,000 cal/mol, in both tantalum and
α-iron. Different interstitials in one host give different barriers, different
relaxation times, and a whole mechanical-loss spectrum of peaks at different
temperatures. Well established — but these are different *defects*, not
different orientations of one defect.

**2. Across orientations of one species — degenerate until stressed, then split.**
This is the FD analogue, and the literature is explicit that it happens:

> *"The interaction energy between the applied stress and the transformation
> strain of each variant differs, leading to an increase in the amount of the
> variants with a greater energy reduction at the expense of other variants."*

and, decisively,

> *"the applied stress lifted **completely** the triple degeneracy."*

So a natural system does exactly what FD requires: **one** defect species, three
orientational variants, made energetically inequivalent by stress, splitting into
**three distinct levels** rather than a singlet plus a doublet. That is empirical
support for the orthorhombic requirement derived below, not merely group theory.

The site description matches literally: *"three types of octahedral sites are
distinguished by the orientation of their major axis of symmetry relative to the
cube axes."* Those are FD's three planes, in a paper about carbon in iron.

**The standing objection is therefore narrower than first written:** orientation
alone cannot split the variants, so the mapping requires the vacuum stress, and
the vacuum stress must be of a particular kind. It is not dead; it is
conditional.

---

## The answer, which is also where it gets interesting

ZW's own framing supplies the fix: **the vacuum has locked into one mode.** That
is an applied stress, and applied stress is exactly what lifts the degeneracy of
Snoek variants. It is the entire basis of stress-induced ordering.

Under stress the three variants are no longer symmetry-related:

* the variant **aligned** with the locked mode sits lowest — stable — **electron**
* the two **misaligned** variants sit higher and relax down — **muon, tau**

This is a better story than it first looks, because it delivers three things at
once from one assumption:

1. a **mass hierarchy** — misalignment energy
2. a **decay channel** — reorientation toward the aligned variant
3. **exactly two unstable generations**, not one and not three

That last is not a fitted parameter. Three variants, one aligned, two not.

---

## And now the prediction, from group theory

If the vacuum stress is **uniaxial** — say along Z — then in a cubic host it
splits the three tetragonal variants as

    {Z-variant}  +  {X-variant, Y-variant}
       singlet          doublet

**Two levels, not three.** A uniaxial stress cannot split X from Y, because a
rotation about Z is still a symmetry of the stressed system.

Muon and tau are *not* degenerate. Therefore:

> **FD requires the vacuum stress to be orthorhombic — all three principal
> stresses distinct.** A uniaxial vacuum is falsified by the muon–tau mass
> splitting alone.

That is a genuine constraint on the vacuum state, derived rather than assumed,
and it is checkable *inside the simulation*: measure the three principal
stresses of the relaxed FD vacuum with one mode locked. If two of them are
equal, the model predicts degenerate muon and tau and is wrong. If all three
differ, the ratio of the splittings should track the mass ratios.

**This is the cheapest high-value experiment available right now.** It needs no
new physics and no literature — just the strain tensor of the locked vacuum.

---

## Can we check it against real data?

Two very different questions, with very different answers.

### The strain quadrupole itself — no

It couples gravitationally. A Planck-scale strain quadrupole of a single lepton
is not directly measurable by anything.

### Species-dependent spatial anisotropy — **yes, and the data is public**

This is the real opportunity. If generations are orientations, and if the
vacuum's orientation is fixed in space, then **different species have
direction-dependent properties in a common frame**. That is precisely what
Lorentz-invariance tests measure, and the results are compiled in:

> **Kostelecký & Russell, *Data Tables for Lorentz and CPT Violation*,
> arXiv:0801.0287** — updated annually, current edition January 2026.

Why this is the right dataset:

* coefficients are quoted **per species** — separate electron, proton, neutron,
  and charged-lepton/muon sectors
* everything is referred to a **common frame** (Sun-centred celestial equatorial),
  so cross-species comparison is meaningful
* it explicitly contains **anisotropic combinations** carrying `XX`, `YY`, `ZZ`
  indices
* each massive spin-½ Dirac fermion has **44 independent observable
  combinations** in the non-relativistic limit
* there is a dedicated **gravity sector**, which is where a strain coupling would
  live
* a muon-specific treatment exists: *Laboratory tests of Lorentz and CPT symmetry
  with muons*, arXiv:1407.7748

**The catch, stated plainly.** These constraints are ferocious — typically
10⁻²⁷ to 10⁻³³ GeV. If FD's vacuum orientation is fixed in space, the model is
probably already excluded by tables that have existed since 2008. If instead the
orientation is genuinely unobservable — the three planes related by a vacuum
symmetry, each particle's condensate picking one — then SME says nothing, but
neither does anything else, and *mutual orthogonality is only meaningful for
co-located particles of different generation*, which is far harder to get at.

So the honest position:

> **The SME tables are a real, public, immediate test — and the most likely
> outcome is that they constrain FD hard.** That is worth knowing either way,
> and it is a day's work to find out rather than a research programme.

---

## What to do next, in order of cost

1. **Measure the principal stresses of the locked FD vacuum.** Free. Decides
   the uniaxial-vs-orthorhombic question above, which is pass/fail.
2. **Compute `P_ij` in the standard elasticity normalisation** so the FD number
   can sit in the same units as published defect tensors.
3. **Read the SME electron and muon sectors** and work out whether FD's
   orientation is observable in that formalism at all. This is the question that
   decides whether the model is testable or merely consistent.
4. Only then: does the misalignment-energy ratio reproduce the mass ratios?

---

## The quantitative problem, which is now the real one

The mechanism is demonstrated to exist. The **regime** is not.

In the Snoek system the stress-induced splitting is a small perturbation: the
barrier is ~25–29 kcal/mol ≈ 1.1–1.3 eV, and laboratory stress shifts variant
energies by µeV to meV, producing a *slight* preferential occupancy. That is the
entire experimental literature — anelastic relaxation is a small-signal
technique.

FD needs splittings comparable to the whole scale. Muon/electron is a factor of
**207** in mass, tau/muon a factor of **17**. Not a perturbation — the dominant
term.

So FD would operate in a regime the Snoek literature never covers: stress so
large the variants are nowhere near degenerate. Whether the Arrhenius form
survives that far outside the perturbative limit is exactly the kind of thing
that usually does not. **This should be settled before any weight is put on the
lifetime mapping.**

---

## Caveats

* The Arrhenius form is borrowed by analogy, from a small-signal regime FD is not
  in (above). Nothing yet shows FD's reorientation obeys it — that needs the
  barrier measured in the simulation.
* `kT` has no obvious FD counterpart. Without one, "the hierarchy is only
  ΔQ/kT ≈ 15.8" is arithmetic, not physics.
* The degeneracy argument assumes the host is cubic. If the FD vacuum's symmetry
  is lower than cubic even before the mode locks, the splitting pattern changes
  and the group theory above must be redone.

---

## Sources

* [Data Tables for Lorentz and CPT Violation, Kostelecký & Russell (arXiv:0801.0287)](https://arxiv.org/pdf/0801.0287)
* [Laboratory tests of Lorentz and CPT symmetry with muons (arXiv:1407.7748)](https://arxiv.org/pdf/1407.7748)
* [Mining the Data Tables for Lorentz and CPT Violation (arXiv:1912.09620)](https://arxiv.org/pdf/1912.09620)
* [The Snoek relaxation in bcc metals — from steel wire to meteorites](https://www.sciencedirect.com/science/article/abs/pii/S092150930601166X)
* [Kinetics of Snoek ordering and Cottrell atmosphere formation in Fe–N single crystals](https://www.sciencedirect.com/science/article/abs/pii/0001616067903707)
* [Elastic dipole tensor of a defect at finite temperature (Phys. Rev. Materials)](https://link.aps.org/doi/10.1103/PhysRevMaterials.5.073609)
