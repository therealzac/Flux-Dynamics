# Finding the multipole centre: three bugs, all of them frame errors

**Date:** 2026-08-14
**Engine:** fd-lattice @ `2c733e5`
**Status:** fixed and verified. **Any multipole number taken before this commit
was computed about the wrong point and should be re-measured.**

---

## What was wrong

### 1. The centre was constant by construction

`centreFromOct()` averaged the octahedron's six vertices in **REST** — the
undisturbed lattice:

```js
const c = [0,1,2].map(k => v.reduce((s,x) => s + REST[x][...], 0) / 6);
```

The rest centroid never moves however much the solver deforms the oct. So the
"centre" could not vary, and every sample across 216 activations reported the
identical `0, −0.577, 0`. That was not a stable measurement; it was a constant
wearing the costume of one.

### 2. The shells were pinned even if the centre moved

`_MULTIPOLE` binned each node's radius as `|REST[i] − c|`. So even with a moving
centre the shells would still have been drawn about the undisturbed lattice.
Both frames had to change together.

### 3. `perturbCentre` ended in a hardcoded origin

```js
... || centreFromStrain()) || [0,0,0];
```

When a lepton's tet vanished between crossings there were no rods and no oct, so
the centre became the **world origin** and the shells teleported to the middle of
the lattice. That is not a worse estimate of where the source is — it is the
*absence* of one, expressed as a number.

---

## The fix

**One frame.** `centreFromOct`, `centreFromRods`, `centreFromStrain`,
`dipoleAt`, `_MULTIPOLE`, `_QUAD` and `_MOMENTS` all read **PD** — where the
geometry actually is.

**A real fallback.** With no oct and no rods, fall through to the **strain
peak**: the displacement *gradient*, `Σ|u_i − u_j|` over base neighbours,
squared and used as a position weight. A gradient is local, so it tracks the
wake a particle leaves. Weighting by `|u|` instead does **not** work — the
displacement field has a long tail and the outer shells, which hold most of the
nodes, drag the answer to the centroid of the box.

**Null when there is nothing.** If the lattice is genuinely relaxed there is no
source, and drawing shells about the origin invents one. `perturbCentre` returns
`null`; callers hide.

**A marker.** The centre is drawn as a small sphere at the point the shells are
built about. Without it the failure was invisible — the shells looked fine.

---

## Verification

Charged lepton crossing, 340 samples:

| | |
|---|---|
| samples with a centre | 228 |
| samples reporting **null** | 112 |
| frames with zero rods | 130 |
| **frames near the origin** | **0** |
| centre span | x `−2.70 … 2.88`, y `−3.77 … 2.63`, z `−2.71 … 2.89` |

Zero frames near the origin, against every gap collapsing there before. First
sample at `(0.577, −3.175, −2.309)` — the upstream boundary where the lepton is
born, not the middle. 130 frames had no rods but only 112 returned null, so in
18 of them the strain wake was still strong enough to locate the source: the
"next most plausible spot" behaviour, working.

Tri-fermion proton, 294 samples over 14 activations: **9 distinct centre
positions**, two y-values straddling the oct rest centre, four x/z positions at
each. The wobble, as a position rather than an inferred angle.

---

## Which centre to use

`perturbCentre()` prefers, in order:

1. **the octahedron's centroid** — strongest available statement about a source
2. **the active rods' centroid** — exact for a bare tet, which has no oct
3. **the strain peak** — field-only, local, tracks a wake
4. **null**

`_SOURCECENTRE()` offers a fourth: the **dipole-nulling point**, found by descent
and **bounded to 3 base edges**. The bound is the whole difference — integrated
over the entire lattice the dipole is dominated by the uniform contraction the
free boundary imposes, so the minimum sits at the centroid of the box. Measured,
it landed 2.0 away from a tet it was asked to find. Restricted to a few shells it
sees only the source.

Over 48 proton activations the dipole-null centre took 17 distinct positions,
median offset 0.376 from the oct centroid, and was **inside the octahedron
48/48** (circumradius 0.7557).

---

## Caveats

* The 3-edge bound on the dipole search is a choice. The two-band y-structure
  should survive changing it to 2 or 4; if it does not, it is a binning artefact.
* The descent is a coordinate search on a 0.25 grid refined to 1e-5, so clustering
  of returned values may partly reflect the search grid rather than the field.
* `maxJumpBetweenSamples` on the lepton run is 5.19, with 34 jumps over 1 unit.
  Some is real — a new lepton is born at the far boundary — but jumps *within* a
  single crossing would be a residual defect in the strain estimator. Samples
  need tagging by particle index before the trajectory is treated as continuous.

---

## Lesson

All three bugs were the same bug: a quantity computed in one frame and used in
another, with no error and no visible symptom. The constant centre in particular
survived two full analysis passes and produced numbers that looked stable —
because they were *identical*, which reads as precision and was in fact a frozen
input. **A suspiciously exact reproducibility is a reason to check the frame,
not a reason for confidence.**
