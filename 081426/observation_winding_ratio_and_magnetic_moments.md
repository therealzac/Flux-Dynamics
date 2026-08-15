# The proton/neutron winding ratio is exactly 25/17, and that is 0.7% from |μp/μn|

**Date:** 2026-08-14
**Engine:** fd-lattice @ `2c733e5`, gen XZ, 360 nodes
**Status:** the ratio is exact arithmetic on recorded integers. The comparison to
the magnetic moments is a **coincidence until killed or confirmed** — the tests
that would do either are named at the bottom. A partial re-measurement today is
reported in full, including what it failed to reproduce.

---

## The number

From `observation_proton_neutron_wobble_and_winding.md`, the centre azimuth of
each particle steps by exactly ±90° per activation. Recorded step counts:

| | forward | back | steps | net quarter-turns | net degrees |
|---|---|---|---|---|---|
| proton | 36 | 11 | 47 | **25** | 2250 |
| neutron | 32 | 15 | 47 | **17** | 1530 |

Both were measured over the **same 47-step window**, and every step is quantised
to a quarter-turn. So the ratio of net windings is a ratio of small integers:

    W_p / W_n  =  25 / 17  =  1.470588…

This is not a fit and carries no error bar as measured. Note the reason, because
it is not quite the one ZW gave. It is exact **not** because the CA is periodic,
but because the steps are quantised to ±90° and the two windows are the same
length. Periodicity would be needed for a different and stronger claim — see
"What this is not", below.

## The comparison

    25/17                    =  1.470588
    |μp / μn|  (CODATA)      =  1.459898        ← +0.73%
    3/2        (SU(6) quark) =  1.5             ← +2.75%

μp = +2.792847 μN, μn = −1.913043 μN. So FD's integer ratio sits **closer to the
measured moment ratio than the naive quark model does**, on the correct side, and
recovers about three quarters of the 2.7% gap the quark model leaves.

Why the two quantities are even comparable: a magnetic moment is circulation of
charge, and winding is circulation. That is the whole of the motivation — it is
suggestive, not a derivation.

---

## What is wrong with it

**The sign.** Both particles wind *positive* in the simulation. μp and μn have
*opposite* signs. So net circulation cannot be the observable — a neutral
neutron would have to give μn = 0, and it does not. The quark model gets the sign
from the individual quark charges, and FD would have to do the same: the
candidate observable is not net winding but the charge-weighted sum

    Σ qᵢ wᵢ

over the live quark at each activation. That requires tagging activations by
flavour, which is already an open thread. Until that is done, the magnitude
agreement is floating free of any mechanism that could produce the sign.

**The window is not a whole number of periods.** This is the serious one. 25/17
is exact *for a 47-step window*, and 47 is not a multiple of the proton's
position period of 8 — `47 mod 8 = 7`. The neutron's dwell pattern (runs of 3, 4
and 6) suggests its period is not 47 either. A ratio of net displacements over a
partial period is not a ratio of asymptotic rates. The physically meaningful
quantity is

    (W_p / P_p) / (W_n / P_n)

with P the period in activations, and **that has not been measured.** If the two
periods differ, the asymptotic ratio need not be 25/17 at all. The apparent
cleanliness of 25/17 partly reflects the arbitrary choice of equal windows.

**Two simple rationals 2% apart.** 25/17 and 3/2 differ by 2%, and the
experimental value sits between them. With one measurement and no error bar,
picking the closer one is not evidence.

---

## Partial re-measurement, 2026-08-14

Fresh page load, gen XZ, tri-fermion proton, sampling `_SOURCECENTRE(3)` per flux
tick and holding only samples after the octahedron forms — 56 post-formation
samples over ticks 52–116.

**Reproduced.** The octahedron centre occupies four positions in the XZ plane at
fixed y = −0.53032 and radius 0.0204 from the lattice axis:

    (−0.0204, 0.00017) → (0.00017, −0.0204) → (0.0204, 0.00017) → (0.00017, 0.0204)

Azimuths 179.52° → −89.52° → 0.48° → 89.52°, i.e. steps of **+90.95, +90.00,
+89.05** — monotonic, quarter-turn quantised. The 90° quantisation is real, the
winding is real, and y = −0.5303 is exactly the upper level the earlier run
reported. That much stands up.

**Not reproduced.** In this window there were **3 forward steps and 0 back
steps**, against the recorded 36/11. The second y-level (−0.6244) never appeared;
y held at −0.53032 throughout. So the *wobble* — the two-state y alternation that
ZW's reading identifies with charge — did not show up at all in 64 ticks, and
neither did any of the back-steps that the 25/17 ratio depends on entirely.

This is not a contradiction: 64 ticks with 3 transitions is far too short to see
a level alternation with a reported period of 4 activations, and the run may
still have been in a transient. But it means **the input numbers to 25/17 are not
yet independently confirmed**, and the doc should not be read as if they were.

An earlier attempt the same day, started mid-stream on a page with prior state,
produced centre offsets of ~0.4 with no periodic structure and is discarded — the
octahedron had not formed for the first 46 samples, so `centreFromOct` was
reporting a default rather than a measured solid.

---

## What this is not

It is **not** a claim that FD predicts μp/μn. FD currently predicts a ratio of
winding integers. Whether that maps onto magnetic moments at all is unestablished,
and the sign says the naive map is wrong.

It is **not** yet a periodicity result. If the walks are exactly periodic then
winding-per-period is an exact integer invariant and the ratio acquires real
force. Establishing that requires measuring both periods and showing the counts
repeat — which is the first test below.

---

## Tests that settle it

1. **Measure both periods, then re-measure winding over the LCM.** This is the
   one that matters. If W_p/P_p ÷ W_n/P_n also lands near 1.46–1.47, the result
   is real and window-independent. If it wanders, 25/17 was an artefact of
   choosing 47 steps for both.
2. **Flavour-tagged winding.** Compute `Σ qᵢwᵢ` per activation. Target: −1.46,
   *with* the sign. This is the only version that could survive the sign problem.
3. **Ten laps.** Confirms the 36/11 and 32/15 splits, which today's short run did
   not see.
4. **Anti-proton and anti-neutron** (`anti-p`, `anti-n` loops are registered).
   Winding should reverse sign at the same magnitudes; the ratio should be
   unchanged. Cheap, and it tests the winding-as-antimatter reading at the same
   time.

Test 1 is the cheapest and it is the one that can kill the result outright. Do it
before anything is built on 25/17.

---

*See also:* `observation_proton_neutron_wobble_and_winding.md` (source of the step
counts), `method_multipole_centre.md` (why the centre had to be made to move
before any of this was visible).
