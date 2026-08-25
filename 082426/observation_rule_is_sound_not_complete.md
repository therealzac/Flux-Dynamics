# The local rule is sound but not complete — and the veto range is finite

**2026-08-24** · exhaustive sweep, 578 symmetry-distinct classes, 360-node lattice
· corrected protocol: fresh solve from REST, no chaining

---

## The sweep

All symmetry-distinct configurations of the 21 shortcut candidates within radius
1.0 of the lattice centre (five overlapping BCC cells), sizes k = 1…4,
**exhaustive**, both rule verdicts, each decided by a fresh solve from REST.

| k | classes | agree | rule-legal but REFUSED | rule-illegal but ACCEPTED |
|---|---|---|---|---|
| 1 | 4 | 4 | 0 | **0** |
| 2 | 24 | 24 | 0 | **0** |
| 3 | 109 | 105 | 4 | **0** |
| 4 | 441 | 350 | **91** | **0** |

**The rule is SOUND: it never accepts an illegal configuration, at any k.**
Everything it rejects really is rejected by the vacuum.

**The rule is NOT COMPLETE: it accepts configurations the vacuum refuses**, and
increasingly so — 21% of rule-legal 4-rod configurations are refused.

Practical consequence: the rule is a **safe pre-filter**, not a decision
procedure. Anything it rejects can skip the solver; anything it accepts still
has to be solved.

## The two failure patterns

| pattern | axes | shared nodes | base residual |
|---|---|---|---|
| triple junction | x, y, z | 2 | 3.06e-6, 2.17e-6 (marginal) |
| two parallel + one | x, z, z | 0 and 1 | 2.69e-5, 3.41e-5 (clear) |

Both live between cells, where a radius-one rule is blind by construction. In
one case the three rods share **no node at all**.

## Does the veto have unbounded range?  NO.

The proposed argument was: no radius-R rule can exist if, for every R, two
R-indistinguishable configurations differ in legality. Placing rods more than 2R
apart makes every R-ball contain at most one rod, and a lone rod looks the same
everywhere, so the construction needs only **one illegal arrangement that stays
illegal at arbitrary separation**.

Tested directly. Take the illegal x-z-z pattern and scale the two z-rods apart
along ⟨1−10⟩, keeping each rod's own local geometry identical (1588 nodes):

| z-rod separation | base residual | verdict | iters |
|---|---|---|---|
| 1.633 | 2.26e-5 | **illegal** | 291,858 (budget burned) |
| 4.899 | 8.90e-11 | legal | 12,562 |
| 8.165 | 9.16e-8 | legal (marginal) | 291,858 (budget burned) |
| 11.431 | 8.11e-11 | legal | 8,978 |

**The veto disappears with separation.** So these counter-examples show the
rule's *radius* is too small, not that no finite radius suffices.

### A correction to earlier reasoning

It was argued that because the displacement field does not decay (kernel modes
are supported on ⟨110⟩ lines and are therefore constant transverse to the line),
two shortcuts must interact at O(1) at any separation, hence no finite local
rule. **That inference is wrong.** A non-zero field at distance does not mean two
constraints *conflict* at that distance: compatibility is a question of whether
the constraint vectors are jointly reachable inside ker Φ, not of field
magnitude. The field really does not decay; the veto apparently does.

## What is now the open question

**What is the veto radius?** The measurement above brackets it for one pattern
between 1.63 (illegal) and 4.90 (legal). If a single finite radius covers every
pattern, a complete local rule exists and the oracle becomes a decision
procedure rather than a pre-filter.

Concretely:
* bracket the transition for each of the two failure patterns
* check whether the radius is pattern-independent
* re-run the exhaustive sweep with the rule extended to that radius and see
  whether completeness is reached

One anomaly worth chasing: at separation 8.165 the solve burned the full
291,858-sweep budget to reach 9.16e-8, while both its neighbours converged in
under 13,000 sweeps. A marginal configuration sitting between the two regimes.
