# Why the equatorial-plane pair goes dark — derived

**2026-08-24** · derived and verified for all three generations
· **and a correction: "dark" is not "unexcited"**

---

## The chain

**1. Where an oct centre sits.** An oct with equator in plane P has its centre at
the mean of four ring nodes, which form a square of side 2 in P and all share a
parity class. Averaging shifts each of the two **in-plane** coordinates by 1 and
leaves the **out-of-plane** coordinate unchanged:

    corners (0,0,0) (2,0,0) (0,2,0) (2,2,0)   ->   centre (1,1,0)

So the two in-plane coordinates of the centre always have **equal parity** to
each other, whatever the corners' class was. The out-of-plane one is inherited.

**2. Where the ⟨110⟩ lattice planes sit.** For a ⟨110⟩ normal **n**, the lattice
planes pass through nodes with **n·node** even, giving spacing 2/√6 = 0.816497.
(Measured: 0.81649 for all six directions — cubic symmetry is intact.)

**3. The parity test.** Let **n** be a ⟨110⟩ normal with zero component along
axis p. Then

    n · c  ≡  c_q + c_r   (mod 2)         q, r the other two axes

which is **even — the centre lies ON a plane — exactly when c_q ≡ c_r**. By (1)
that happens precisely when p is the out-of-plane axis, i.e. **exactly when n
lies in the oct's equatorial plane**.

Verified for all three generations:

| gen | oct centre | banned axis | ON pair | in-plane pair | rule |
|---|---|---|---|---|---|
| XZ | (−1, 0, −1) | Y | ⟨101⟩ ⟨10−1⟩ | ⟨101⟩ ⟨10−1⟩ | ✓ |
| XY | (0, 0, −1) | Z | ⟨110⟩ ⟨1−10⟩ | ⟨110⟩ ⟨1−10⟩ | ✓ |
| YZ | (0, −1, −1) | X | ⟨011⟩ ⟨01−1⟩ | ⟨011⟩ ⟨01−1⟩ | ✓ |

**4. What the slab then catches.** With half-width W = 0.5 against a spacing of
0.8165:

* centre **BETWEEN** planes (offset exactly 0.40825, half a spacing) — the two
  flanking planes at ±0.408 both fall inside the slab → **420 nodes**
* centre **ON** a plane (offset 0) — only that one plane is inside; its
  neighbours at ±0.816 are outside → **204 nodes**

This is the halved null in the measurement (0.076 against 0.156), and it is pure
geometry: the null shuffles weights within radial shells, so it sees only how
much lattice the slab contains.

**5. Where the displacement actually lies.** Weight per lattice plane, indexed in
half-spacings from the oct centre, radially normalised, as a percentage of total:

    <011>  BETWEEN   -5:5.55  -3:5.68  -1:13.22   1:13.24   3:5.68   5:5.56
    <101>  BETWEEN   -5:5.55  -3:5.69  -1:13.17   1:13.31   3:5.67   5:5.55
    <110>  ON        -6:6.65  -4:7.00  -2: 5.45   0: 7.25   2:5.42   4:7.02  6:6.64
    <1-10> ON        -6:6.65  -4:7.01  -2: 5.45   0: 7.25   2:5.42   4:7.02  6:6.64

The BETWEEN families put **13.2% on each flanking plane against a 5.6%
background — a 2.4× concentration**, and both planes sit inside the slab. The ON
families are **flat**: 7.25% at the centre against 7.0% four half-spacings out,
and they *dip* to 5.4% at ±2. There is no contrast for a slab detector to find,
which is exactly what is measured (f = −0.004 and −0.005, σ = −1.3 and −1.6).

∎

---

## CORRECTION: "dark" is not "unexcited"

The earlier framing in `observation_generation_swap_confirms_kernel.md` — that
the oct "does not emanate" the in-plane pair — overstates what was measured.
Computing the oct's coupling to each labelled kernel mode directly, on an N = 6
torus:

| source | k_x=0 ⟨011⟩ | k_y=0 ⟨101⟩ | k_z=0 ⟨110⟩ |
|---|---|---|---|
| single rod along x | **0.000e+00** | 9.62e-02 | 9.62e-02 |
| single rod along y | 9.62e-02 | **0.000e+00** | 9.62e-02 |
| single rod along z | 9.62e-02 | 9.62e-02 | **0.000e+00** |
| 4-rod oct, XZ equator | 1.361e-01 | **1.443e-01** | 1.361e-01 |

The single-rod rule is exact and worth keeping on its own: **a rod along â
cannot excite any mode with k_â = 0**, because the coupling carries a factor
(1 − e^{i k·d̂}) that vanishes identically there.

But the 4-rod oct couples **most strongly** to the k_y = 0 family — the very
pair the detector reports as dark. So the in-plane family is *excited*, possibly
more than the others. What it lacks is a real-space signature a slab detector
can see, because the oct centre sits **on** one of that family's lattice planes
rather than **between** two of them.

**"Dark" is a statement about the displacement profile, not about kernel
occupation.** The generation-swap result stands — it is a statement about what
the detector sees, and it confirmed a prediction about which planes are
detectable — but it does not license the claim that four modes are excited and
two are not.

---

## Open

* **Is the in-plane family really excited?** The coupling says yes. A detector
  sensitive to it would need to be centred half a spacing off the oct, or use a
  slab wide enough to span the neighbouring planes. Cheap and worth doing: it
  would decide whether the oct emanates four planes or six.
* If it is six, the "four planes for an oct, two for a nucleon" counting in
  `081626/` is a statement about detectability, not about the physics, and the
  nucleon's 2.8:1 ratio needs re-reading in that light.
