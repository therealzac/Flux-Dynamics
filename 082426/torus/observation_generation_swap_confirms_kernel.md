# The generation swap: the displacement planes are the kernel of Φ

**2026-08-24** · falsification test, passed · 2698 nodes, 4-rod oct, both generations

---

## The prediction, made before the measurement

From `observation_floppy_modes.md`: ker Φ is supported on six ⟨110⟩ lines,
k_a = 0 with k_b = ±k_c. Comparing against the XZ measurements in `081626/`, the
oct lit **four** of the six, and the dark pair was the k_y = 0 family — with the
generation set to XZ, which bans Y. That gave a rule and a falsifiable
consequence:

> A kernel line is lit iff it has a nonzero component along the banned axis.
> So switching XZ → XY must send ⟨110⟩ and ⟨1−10⟩ dark and light ⟨101⟩/⟨10−1⟩,
> while ⟨011⟩/⟨01−1⟩ stay lit in both.

Nothing in the derivation of ker Φ refers to generations, octs, or the walk.

## Result

| kernel line | XZ (bans Y) | XY (bans Z) |
|---|---|---|
| ⟨011⟩ | **lit** 0.2648 | **lit** 0.2646 |
| ⟨01−1⟩ | **lit** 0.2644 | **lit** 0.2647 |
| ⟨110⟩ | **lit** 0.2655 | **dark** |
| ⟨1−10⟩ | **lit** 0.2636 | **dark** |
| ⟨101⟩ | dark | **lit** 0.2648 |
| ⟨10−1⟩ | dark | **lit** 0.2643 |

Offsets from the ideal lattice directions: 0.00°–0.73°. Exactly the predicted
swap, on all six lines.

Full rankings:

    XZ   1 <110>  0.73 deg  live 0.2655  f 0.1273
         2 <011>  0.25      live 0.2648  f 0.1276
         3 <01-1> 0.25      live 0.2644  f 0.1524
         4 <1-10> 0.25      live 0.2636  f 0.1662
    XY   1 <101>  0.73      live 0.2648  f 0.1262
         2 <01-1> 0.25      live 0.2647  f 0.1289
         3 <011>  0.25      live 0.2646  f 0.1311
         4 <10-1> 0.00      live 0.2643  f 0.1544

## The rule, in its cleanest form

**The dark pair is the ⟨110⟩ pair lying IN the generation's own equatorial
plane. An oct emanates the four kernel planes whose normals stick out of its
equator, never the two lying in it.**

XZ generation → lines in the XZ plane are ⟨101⟩,⟨10−1⟩ → dark.
XY generation → lines in the XY plane are ⟨110⟩,⟨1−10⟩ → dark.

## Same object, rotated

| | XZ | XY |
|---|---|---|
| field energy E | 85.514 | 85.514 |
| max displacement | 0.129813 | 0.129918 |
| solver base residual | 7.75e-11 | 8.86e-11 |
| live scores | 0.2655 0.2648 0.2644 0.2636 | 0.2648 0.2647 0.2646 0.2643 |

Independently consistent with `081626/`'s finding that the three generations are
exact rotations of one another.

## Two controls that had to pass first

**1. The instrument was rebuilt and had to reproduce the original.** `__MEASK3`
was an injected harness, not in the source, so it was rebuilt from the method
description in `081626/observation_working_plane_detector.md`: radial
normalisation over 16 bins, within-shell shuffles for the null,
f = (live − null)/(1 − null), coarse 800-normal hemisphere → 30-candidate
shortlist at 10° separation → refine every candidate ±4° at 1° → rank on refined
→ 20° non-max suppression, null through the identical pipeline.

Rebuilt XZ:  0.2655 0.2648 0.2644 0.2636
081626 XZ:   0.2655 0.2648 0.2644 0.2636

Identical to four decimals, same directions, same order — before the test case
was touched.

**2. The first XY attempt returned a silent empty field.** E = 0, octs = 0,
max|PD − REST| = 0, and the reported residuals were base 0 / sc 0 / minSep 1.
Cause: `NOSOLVE` is true and the XY configuration is not in the pre-solved pack,
so the engine correctly left the geometry unchanged — which reads as *no
displacement* rather than as an error. This is exactly the trap recorded in
`081626/README.md` ("assert `_MISSING()` is empty before trusting a field").

The peaks it returned were 23°–43° off any lattice direction with f = 0, i.e.
visibly garbage, but the failure mode is silent and would pass an unwary eye.
Fix: `_SOLVE1` arms the in-page solver directly and bypasses the pack. **Both**
generations were then re-run through that identical path, and XZ reproduced its
own numbers exactly under the new path before XY was measured.

## What this establishes

The displacement planes measured empirically across six lattice sizes in
`081626/` **are** the kernel of the base-network dynamical matrix. The
identification was derived analytically, predicted a swap under a variable the
derivation never references, and the swap happened on all six lines.

Consequences already in hand:
* the ⟨110⟩ normals, the 60°/90° angle structure, the one-node-layer plane
  thickness fixed in absolute units, and the scale-free f are all forced by
  line-supported modes — none of them need to be measured again
* the absence of 1/r² decay is not an anomaly to explain; a line-supported mode
  has no 1/r² in it

## Open

* **Why four and not six?** The rule is established empirically. The reason the
  equatorial-plane pair is dark has not been derived.
* A 7-rod nucleon lights **two** planes at 60° in a 2.8:1 ratio (`081626/`).
  Which two, as a function of the rod set, is now a well-posed question against
  a known basis of six.
* Two sources: which lines each lights, and whether the overlap is attractive.
