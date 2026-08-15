# The proton's source centre is a period-8 cycle; the neutron's is not

**Date:** 2026-08-14
**Engine:** fd-lattice @ `2c733e5`, gen XZ, 360 nodes, 2 laps each
**Status:** measured, with the moving-centre fix in place. Earlier numbers on
this question were taken about a frozen centre and are superseded.

---

## Setup

Tri-fermion proton and neutron, generation XZ (dormant axis Y), 48 activations
each. The multipole expansion centre is now found per-frame and moves with the
source — see `method_multipole_centre.md`, without which none of this was
visible.

Sampled densely (≈1100 frames each) plus once per first-order activation.

---

## Result

| | proton | neutron |
|---|---|---|
| distinct centre positions | **8** | **10** |
| y-levels visited | 2: `−0.6244`, `−0.5303` | **4**: `−0.6244`, `−0.5899`, `−0.5772`, `−0.5303` |
| occupancy per level | 553 / 554 | 553 / **6** / **6** / 576 |
| dwell pattern | strict **4 and 4** | runs of **3, 4 and 6** |
| centre azimuth, net turn | **+2250°** | **+1530°** |
| per activation | **+47.9°** | +32.6° |
| ±90° steps, forward / back | **36 / 11** | 32 / 15 |

### The proton's cycle is exact

Eight positions, period 8, repeating without exception:

    y = −0.5303 (upper):   x−  →  z−  →  x+  →  z+
    y = −0.6244 (lower):   x+  →  z+  →  x−  →  z−

Azimuth advances **+90° every activation** — the winding. The y-level alternates
every **4** activations — the wobble. Occupancy of the two levels: 553 / 554.

### The neutron's is not

Two extra y-levels appear, `−0.5899` and `−0.5772`, for 6 samples each out of
1141. Note that **−0.5772 is the octahedron's rest centre**: the neutron passes
*through* the midpoint, the proton never does. Occupancy is lopsided (553 vs
576 on the outer levels, against the proton's 553/554), and the dwell runs break
into 3s and 6s instead of a clean 4.

---

## Reading

Both particles wobble along Y and both wind about it. What differs:

* the proton's wobble is **strictly two-state**; the neutron's is not
* the proton's ratchet is cleaner: 36 forward to 11 back, against 32 to 15
* the proton winds ~1.5× faster per activation

Under ZW's reading — **charge is the wobble, not the winding rate** — the sharp
statement is not "the neutron doesn't wind" but *"the neutron's wobble is not
two-state"*.

### Superseded

An earlier pass measured the **dipole azimuth** about a frozen centre and found
net turns of −4089.6° (proton) vs −1721.9° (neutron), with ±90° steps 33/0 and
23/1. Direction is opposite to the centre-azimuth numbers above because it is a
different observable, and the whole set was computed in the wrong frame. Kept
only as a record that the proton/neutron split showed up in both observables.

---

## Caveats

* **The intermediate y-levels are 12 samples out of 1141 (~1%) and they
  cluster.** That looks like transitions caught in flight rather than genuine
  dwell states. Denser sampling would settle whether the neutron ever *holds*
  an intermediate height or only passes through.
* **48 activations.** The 36/11 vs 32/15 ratchet difference is roughly 1σ.
  Suggestive, not established. Ten laps would settle it.
* Scalar quantities were **identical** between the two in the earlier fixed-frame
  run — same monopole median to five figures, same dipole magnitude, same 3.13°
  quadrupole tilt. Worth re-checking in the corrected frame: if that survives,
  proton and neutron differ *only* in the dynamics of the centre, not in any
  static magnitude.

---

## Open threads

* Ten laps, both particles.
* Do anti-proton and anti-neutron wind the **opposite** sign at the same
  magnitudes? That is close to a decisive test of the winding-as-antimatter
  reading and it is cheap to run.
* Tag each activation with which of the six quark types is live. If the discrete
  centre positions map onto quark flavour, the proton/neutron difference *is* the
  2:1 vs 1:2 flavour ratio appearing directly in the field.
