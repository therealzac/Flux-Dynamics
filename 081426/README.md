# 2026-08-14 — Generations as orientational variants

Session that made all three generations run, then measured what distinguishes
them. The headline is that the structure FD produces is a known one in
elasticity, which brings machinery and an experimental signature with it.

---

## Index

| file | what it says |
|------|--------------|
| [observation_generations_are_orientational_variants.md](observation_generations_are_orientational_variants.md) | Each generation's strain quadrupole aligns with its own dormant axis to ~3°, and the tensor is always tetragonal. This is the elastic dipole tensor of a tetragonal defect in a cubic host — three orientational variants, Snoek physics. The identification with lepton generations is the claim. |
| [observation_proton_neutron_wobble_and_winding.md](observation_proton_neutron_wobble_and_winding.md) | The proton's source centre is an exact period-8 cycle over 8 positions, two y-levels, occupancy 553/554. The neutron visits 10 positions across 4 levels including the oct's own midpoint, with ragged dwell. |
| [observation_winding_ratio_and_magnetic_moments.md](observation_winding_ratio_and_magnetic_moments.md) | Proton and neutron net winding over the same window are exactly 25 and 17 quarter-turns. 25/17 = 1.4706 sits 0.73% from \|μp/μn\| = 1.4599, closer than the quark model's 3/2. The sign is wrong and the window is not a whole period, so it is a coincidence until two named tests run. |
| [observation_snoek_lifetime_hierarchy_mapping.md](observation_snoek_lifetime_hierarchy_mapping.md) | The Snoek system supplies an Arrhenius form for the flavour hierarchy — but three variants of a cubic host are **degenerate**, so orientation alone cannot split them. Vacuum stress lifts it, which then forces the vacuum to be **orthorhombic**, not uniaxial. And the SME data tables are a real public test. |
| [method_multipole_centre.md](method_multipole_centre.md) | Three frame bugs that made the expansion centre a constant. Any multipole number taken before `2c733e5` is measured about the wrong point. |

---

## The one-paragraph version

A generation is which coordinate plane the oct's equator occupies; the dormant
axis is the normal. Measured over 2 laps of the tri-fermion proton in each
generation, the quadrupole principal axis sits **2.4–3.4° from that
generation's dormant axis** and 87–90° from the other two, with identical falloff
exponents across all three (l2 = −1.50). The tensor is uniaxial in every one of
144 samples. That is an **elastic dipole tensor** with **three orientational
variants**, which is exactly what a tetragonal defect at an octahedral site in a
cubic host produces — the Snoek system. The prediction that follows is
relational and frame-independent: *an electron, a muon and a tau in the same
region have mutually orthogonal strain quadrupoles.* And since Snoek variants
reorient under applied stress, **muon and tau decay to generation 1 would be
variant reorientation** driven by the vacuum having locked into one mode.

---

## The next experiment, and it is free

Three orientational variants of a **cubic** host are related by a symmetry of
that host, so they are **exactly degenerate** — orientation alone cannot give
electron, muon and tau different masses. Vacuum stress lifts the degeneracy
(that is the whole Snoek effect), which yields a mass hierarchy, a decay channel,
and exactly two unstable generations from one assumption.

But a **uniaxial** stress in a cubic host splits three tetragonal variants as
**1 + 2**, not 1 + 1 + 1 — a rotation about the stress axis is still a symmetry.
Muon and tau are not degenerate, so:

> **FD requires the locked vacuum to be orthorhombic — all three principal
> stresses distinct.** A uniaxial vacuum is falsified by the muon–tau splitting.

**Measure the three principal stresses of the relaxed FD vacuum with one mode
locked.** Costs nothing, needs no literature, and is pass/fail. If two are equal
the model predicts degenerate muon and tau and is wrong.

---

## Open threads

**Measurement**
- [ ] **Principal stresses of the locked vacuum — uniaxial or orthorhombic?** (top priority)
- [ ] Read the SME electron and muon sectors: is FD's orientation observable in
      that formalism at all? Decides whether the model is testable or merely
      consistent — and the constraints are 10⁻²⁷–10⁻³³ GeV, so this may well
      exclude it.
- [ ] **Measure both walk periods, then re-measure winding over the LCM.** Decides
      whether 25/17 is asymptotic or an artefact of using a 47-step window for
      both particles (`47 mod 8 = 7`). Cheapest way to kill the result.
- [ ] **Flavour-tagged winding `Σ qᵢwᵢ`.** The only version that could produce the
      *sign* of μp/μn, which net winding cannot. Target −1.46.
- [ ] Ten laps, proton and neutron, to firm up ratchet purity (currently ~1σ) and
      to confirm the 36/11 and 32/15 splits — a 64-tick run on 08-14 saw 3/0 and
      no y-level alternation at all.
- [ ] Anti-proton and anti-neutron: do they wind the opposite sign at the same
      magnitude? Cheap, and close to decisive for winding-as-antimatter.
- [ ] Bin-width sweep — is the 3° quadrupole tilt real or a shell-binning artefact?
- [ ] Tag each activation by which quark type is live; do the discrete centre
      positions map onto flavour?
- [ ] Re-check whether proton and neutron still share identical scalar magnitudes
      in the corrected frame. If so they differ *only* in centre dynamics.

**Theory**
- [ ] Why is l2 = **−1.50** for a three-quark object where a single tet gives
      **−1.92**? Reproducible to two decimals in all three generations.
- [ ] Compute `P_ij` in the standard normalisation so it can be compared with
      published defect tensors.
- [ ] Does a variant reorient under imposed lattice strain? At what rate? The
      elasticity literature gives a functional form for Snoek relaxation —
      does the flavour lifetime hierarchy fall out of it?

**Literature**
- [ ] Proper review. One search found no identification of elastic-dipole
      orientational variants with particle generations, but that is not a
      literature review. Next session.

---

## Corrections carried in

* The anisotropy `0.500` reported as unexplained in earlier notes is **not a
  finding** — a traceless uniaxial tensor gives `2a/4a = 1/2` identically. It is
  the readout of "always tetragonal", not a separate fact.
* Charged-lepton falloff exponents are **unusable** (moving source). Orientation
  data from the same runs is fine.
* Everything about the multipole centre before `2c733e5` is superseded.
