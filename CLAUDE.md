# Flux Dynamics — Claude Context Document

# ⛔⛔⛔ ONLY WORK ON THE WORKTREE. PERIOD. ⛔⛔⛔
# All file edits go to the WORKTREE, never the main repo.
# The user will tell you EXPLICITLY when to push to GitHub.
# Never implicitly push. Never edit files outside the worktree.
# WORKTREE MUST BE UPDATED FIRST. Only copy to main repo when
# pushing to GitHub. Never push stale/out-of-sync files.

# ⛔⛔⛔ ONLY WORK ON THE WORKTREE. PERIOD. ⛔⛔⛔
# All file edits go to the WORKTREE, never the main repo.
# The user will tell you EXPLICITLY when to push to GitHub.
# Never implicitly push. Never edit files outside the worktree.
# WORKTREE MUST BE UPDATED FIRST. Only copy to main repo when
# pushing to GitHub. Never push stale/out-of-sync files.

# ⛔⛔⛔ ONLY WORK ON THE WORKTREE. PERIOD. ⛔⛔⛔
# All file edits go to the WORKTREE, never the main repo.
# The user will tell you EXPLICITLY when to push to GitHub.
# Never implicitly push. Never edit files outside the worktree.
# WORKTREE MUST BE UPDATED FIRST. Only copy to main repo when
# pushing to GitHub. Never push stale/out-of-sync files.

# ⚠️⚠️⚠️ SCIENTIFIC SIMULATION — RESEARCHER APPROVAL REQUIRED ⚠️⚠️⚠️
# This is a particle physics simulation. It is EXTREMELY SENSITIVE.
# ALL code updates must be approved by researchers BEFORE implementation.
# If you update something without telling the researchers, we run the
# risk of reaching INCORRECT SCIENTIFIC CONCLUSIONS.
# The importance of this cannot be overstated.
# ⚠️⚠️⚠️ ALWAYS EXPLAIN CHANGES AND GET APPROVAL FIRST ⚠️⚠️⚠️

# ⛔⛔⛔ NEVER PUSH TO GITHUB ⛔⛔⛔
# UNLESS THE USER SAYS "PUSH" IN **THIS EXACT MESSAGE**
#
# COMMIT LOCALLY → STOP → WAIT FOR USER TO SAY "PUSH"
#
# ❌ "Push" from a PREVIOUS message does NOT count
# ❌ "Push to main" from 5 minutes ago does NOT count
# ❌ Bundling a push with a commit is FORBIDDEN
# ✅ ONLY push when the current user message contains "push"
#
# THIS IS THE #1 RULE. VIOLATING IT IS UNACCEPTABLE.
# ⛔⛔⛔ NEVER PUSH TO GITHUB ⛔⛔⛔

---

## Test-First Development Doctrine

**Critical: All features implemented must have a robust test built before implementation can begin.**

WHEN IMPLEMENTING A NEW FEATURE, I WILL WRITE A UNIT TEST FOR THE FEATURE FIRST, AND THEN BUILD THE FEATURE

IF THE FEATURE IS TOO COMPLEX FOR A SINGLE UNIT TEST, I WILL BREAK IT DOWN INTO MULTIPLE UNIT TESTS

IF THIS PROCESS IS AT ALL UNCLEAR, I WILL ASK FOR GUIDANCE

UNIT TESTS MAY NOT BE UPDATED EXCEPT WITHOUT EXPRESS USER CONSENT

ALL UNIT TESTS CAN BE PROGRAMATICALLY VERIFIED AND PROGRAMATICALLY VIOLATED, AND ALL UNIT TESTS ARE DIRECTLY WIRED UP TO DO SO

---

## ABSOLUTE PROHIBITIONS

### NEVER monkey-patch Set/Map prototype methods
**NEVER intercept, wrap, or replace `.add()`, `.delete()`, `.set()`, `.get()`, or any other method on `xonImpliedSet`, `activeSet`, `impliedSet`, `impliedBy`, or any other global Set/Map.** This includes debug interceptors, proxies, or any form of method replacement. Doing so corrupts the unified architecture — shapes stop driving spheres, the solver becomes detached from rendering, and the simulation breaks catastrophically. If you need to trace who adds to a Set, use `console.error` with stack traces at the CALL SITE, not by wrapping the Set method.

### Speed sanity check
If a change causes the Planck-second counter to run ~25x faster than normal, **you probably broke the physics solver.** The solver is the bottleneck — if the simulation suddenly flies, it means constraints are no longer being enforced and the results are meaningless. Treat unexpected speedups as a regression, not an improvement.

### ⛔⛔⛔ NEVER suppress guards to make replay work ⛔⛔⛔
**Guards exist to detect corruption. If guards fail during replay, THE REPLAY IS CORRUPT — fix the replay pipeline, not the guards.** This includes:
- ❌ NEVER add `if (replaying) skip guards` or `if (redoDrain) return` conditions
- ❌ NEVER silently reset failed guards during replay to let corrupt data pass
- ❌ NEVER argue "guards don't make sense during replay" — they absolutely do; they validate that saved data is physically correct
- ❌ NEVER disable movement guards (T20, T26, T27) because "snapshot restores aren't incremental" — the fix is proper prev-state tracking, not guard suppression
- ✅ If guards fail during redo drain, the snapshot data or restore logic is broken — FIX THAT
- ✅ Guards during replay are the ONLY defense against saving and replaying corrupt garbage
- ✅ The replay pipeline must produce tick-to-tick state that is indistinguishable from live play

**This is the same principle as "never modify tests." Suppressing guards during replay IS modifying tests — it's just doing it at the call site instead of the test definition.**

### NEVER gut the framework to bypass tests
**We want to find RULES that make the tests always pass — not change the underlying framework.** If a test fails, the fix is better choreography logic, not replacing a real `Map` with a no-op object, disabling checks, or making data structures lie. The framework (SC sets, solver, etc.) is the physics engine. The rules (movement heuristics, assignment logic, lookahead) are what we tune. **NEVER replace a framework data structure with a fake/no-op version.**

### NEVER sacrifice accuracy or completeness
**This is an actual particle physics simulation. No performance heuristics, no artificial tick limits, no corner-cutting of any kind.** Every possibility must be checked. Every value must be calculated to full precision. Never introduce caps, early exits, sampling, or approximations that reduce the completeness of the search or the accuracy of the physics. If the search takes longer, it takes longer. Correctness is non-negotiable.

### ⛔⛔⛔ Unit tests are sacred — ABSOLUTE ZERO-TOLERANCE RULE ⛔⛔⛔
**Unit tests may NEVER be altered without the explicit written permission of Zac IN THE CURRENT MESSAGE.**
Tests define the physics we are trying to satisfy. If a test fails, the fix is in the choreography/movement logic — never in the test itself. Modifying a test to make it pass is the same as lying about whether the physics works.

**THIS MEANS:**
- ❌ NEVER add mode filters, skip conditions, or exceptions to test check() or projected() functions
- ❌ NEVER weaken a guard by excluding xon types, modes, ticks, or states
- ❌ NEVER "fix" a test failure by making the test accept the broken behavior
- ❌ NEVER touch flux-tests.js unless Zac says "change test X" in THIS message
- ✅ The ONLY fix for a test failure is better choreography/movement logic
- ✅ If tests say the rules are impossible, the CHOREOGRAPHER is wrong, not the tests
- ✅ Use "Test model exhaustiveness" to verify the choreographer is searching correctly

**Zac has corrected this multiple times. There are NO exceptions. Period.**

### Unit test IDs must be globally unique
Every entry in `LIVE_GUARD_REGISTRY` is keyed by its `id` field (e.g. `'T90'`, `'T91'`, `'T92'`). The `_liveGuards` object is built by iterating the registry and assigning `_liveGuards[entry.id]`. **Duplicate IDs cause silent overwrites** — the later entry clobbers the earlier one, and the earlier guard never runs. This happened with T92 (hadron ejection was silently killed by a duplicate idle-oct guard). Always verify a new test ID is unique across the entire registry before adding it.

### `_moveRecord` — useful audit tool, MUST NOT affect physics
`_moveRecord` is a tick-level `Map` that records `destNode → fromNode` for every xon move each tick. It is useful for **auditing, playback, and debugging**. It **MUST NEVER affect physics** — no `.get()` calls to block, reject, or filter moves. `_moveRecord` is a passive observer that records what happened. If a test needs to detect swaps or other patterns, use the live guard snapshot system (`_liveGuardPrev`), not `_moveRecord`.

---

## Guiding Principle: Simplicity

**Make it as simple as possible.** Simplifying the code clarifies the physics _and_ makes it easier to maintain. When choosing between approaches, prefer the one with fewer data structures, fewer code paths, and less indirection. Complexity is the enemy of correctness in a physics simulation — every unnecessary abstraction is a place where bugs hide.

---

## Quick Start
- **Press the "demo" button** to start the deuteron simulation
- Dev server: `python3 -m http.server 8080` from project root, open `flux-v2.html`
- After editing JS, **bump the `?v=N` cache buster** in `flux-v2.html` script tags

---

## 1. What Is Flux Dynamics?

Flux Dynamics is a geometric framework for unification, authored by Zac Wickstrom. The core thesis: reality is a discrete lattice derived from the Voronoi tessellation of the maximal sphere packing (FCC). The void centers form a graph with exactly 10 unit-length traversal directions — 4 base FCC directions and 6 hexagonal close-packed (HCP) conditional shortcuts. This structure embeds as a 3D projection from the D₁₀ root lattice.

### Key Physics from the Paper
- **FCC lattice**: Vertices are void centers (tetrahedral A₄ and octahedral Oₕ) of the densest sphere packing (74.048%)
- **4 base directions** (v₁–v₄): Unit-length FCC edges along ⟨111⟩ permutations
- **6 shortcut directions** (s₁–s₆): HCP reconfigurations ("flux events") that shear rhombi into squares, adding conditional unit-length connections
- **Mutual Exclusivity**: Only one flux mode (orthogonal shortcut pair) can be active at a time — simultaneous activation yields negative metric eigenvalues
- **Void Duality**: Perturbations must manifest as dual pairs across the bipartite tet/oct graph. The tet (A₄) is the "Source" (fermionic condensing knot), the oct (Oₕ) is the "Sink" (bosonic accommodating reservoir)
- **Particle identity from traversal topology**: A 4-hop closed loop on a tet K₄ has 4 distinct shapes → 4 quark flavors (fork=p-up, lollipop=n-down, Hamiltonian CW=p-down, Hamiltonian CCW=n-up). Without an oct face basis, loops are indistinguishable → electron
- **Three generations** from edge composition: Gen1 = 4 base + 2 shortcut (1 flux mode), Gen2 = 3+3 (2 modes, domain wall), Gen3 = 2+4 (2 modes). Gen4 impossible (triple junction overconstrains)
- **Gauge group emergence**: SU(3)c from oct Oₕ symmetry (strong/color), U(1)Y from tet A₄ (hypercharge), SU(2)L from flux-induced chirality breaking (weak force = activated shear mode preference)

### The Deuteron (H-2)
The simulator models a deuteron: 1 proton + 1 neutron sharing an octahedral cage.
- **8 tet faces** in K₄,₄ bipartite structure:
  - Group A (proton): faces {1,3,6,8} with oct nodes {0,5,7,9,11,13}
  - Group B (neutron): faces {2,4,5,7}
- **6 xons** (anonymous fermionic workers) traverse tet loops to actualize shortcuts
- **4 oct SCs** form the bosonic cage (maintained by gluon xons)
- **Anti-phase**: proton activates on A-faces when neutron on B-faces (alternating)
- **Pauli exclusion**: max 1 xon per node at any tick

---

## 2. Project Structure

```
flux-v2.html               — Main HTML entry point (loads all JS via script tags)
src/
  flux-constants.js         — Global constants, geometry, state declarations, tet/quark definitions
  flux-solver-proxy.js      — Solver web worker proxy
  flux-solver-worker.js     — CPU & GPU solver backend (Web Worker)
  flux-solver-render.js     — PBD constraint solver, FCC lattice builder, Three.js rendering
  flux-electrons.js         — Excitation system, vacuum negotiation (canMaterialiseQuick, sever)
  flux-voids.js             — Void detection, tet/oct rendering, void sphere meshes
  flux-nucleus.js           — Nucleus simulation setup (deuteron H-2), NucleusSimulator IIFE
  flux-demo-state.js        — Demo shared state, constants, PRNG, choreo params, loop sequences
  flux-demo-xon.js          — Xon lifecycle: spawn, destroy, advance, trails, occupancy
  flux-demo-backtrack.js    — Backtracker: state snapshots, BFS layers, exclusion ledger
  flux-demo-planner.js      — Move planner: bipartite matching, lookahead, face scoring, assignment
  flux-demo-ui.js           — Demo panels, profiling, choreo logging, pause/resume/stop
  flux-demo.js              — Orchestrator: demoTick phases, opening, schedule, gluons, algo registry
  flux-tests.js             — Test suite (T01-T28), live guards (T19/T21/T26/T27)
  flux-rules-v2.js          — K-complexity metrics, rule tournament engine, QUARK_ALGO_REGISTRY
  flux-tournament.js        — Tournament execution, fitness evaluation, playback system
  flux-ui.js                — UI controls, camera/orbit, jiggle mode, export, raycasting
"Flux Dynamics (1).pdf"     — The foundational paper (20+ pages)
```

All files are loaded as global-scope `<script>` tags in `flux-v2.html`. No module system. Order matters.

---

## 3. The Solver & Rendering (flux-solver-render.js)

### PBD Constraint Solver
`_solve(pairs, maxIter, noBailout)` — Position-Based Dynamics using Gauss-Seidel iteration.
- Each pair `[i,j]` is a distance-1 constraint (base edge or active shortcut)
- Iterates: project vertices onto constraint manifolds until max error < 1e-6
- Returns `{p, converged}` — final positions and convergence status

### FCC Lattice
`buildLattice(level)` — Generates hierarchical FCC lattice:
- L0 = 2 nodes (single cell), L1 = 15 nodes, L2 = 107 nodes (default for nucleus), L3 = 434+ nodes
- Populates `BASE_EDGES`, `baseNeighbors[node]`, `REPULSION_PAIRS`

### Shortcut Geometry
`_computeScGeometry()` — Enumerates all possible shortcuts:
- `ALL_SC[]` — every shortcut: `{id, a, b, stype, label}`
- `SC_BY_ID[id]` — lookup by ID
- `scByVert[node]` — shortcuts incident on node
- `scPairToId` — Map from `pairId(a,b)` → SC ID (for edge lookup)
- `baseNeighbors[node]` — base-edge neighbors with `{node, dirIdx}`

### Three SC Sets (critical global state)
| Set | What | Who manages |
|-----|------|-------------|
| `activeSet` | UI-placed SCs only | UI clicks, rule engine |
| `xonImpliedSet` | Xon-materialized SCs (oct moves, tet loops, cage) | flux-demo.js, flux-demo-planner.js |
| `impliedSet` | Geometrically necessary SCs (derived) | `detectImplied()` after solver |

### Rendering Pipeline
- Three.js scene with `InstancedMesh` for spheres (foreground/background split)
- Per-SC colored wireframe lines (`scLineObjs`)
- Void sphere meshes for tet/oct voids (color-coded by quark type in demo)
- `_ruleAnnotations` object: tetColors, tetOpacity, octColors, scColors, nodeColors — used by demo to color voids

---

## 4. Vacuum Negotiation (flux-electrons.js)

The "vacuum" is the constraint solver. It has final say on every SC materialization.

### `canMaterialiseQuick(scId)` — Dry-run strain check
1. Returns true immediately if SC already in any active set
2. Builds constraint pairs: all current SCs + candidate
3. Runs solver ONCE, measures strain on BASE_EDGES
4. **Thresholds**: worst edge error ≤ 5e-4, average ≤ 1e-5
5. No side effects — safe for lookahead

### `excitationSeverForRoom(targetScId)` — Make room by severing
When strain is too high, sever ONE non-load-bearing electronImplied SC:
1. Build **protected set**: tet pairs, actualized oct cycles, void-bound excitation SCs, oct cage SCs
2. Rank severable candidates by fewest excitation references (orphans first)
3. Try each: sever → check if target can now materialize → keep or undo
4. At most ONE SC severed per call

### Vacuum Negotiation Protocol (used everywhere)
```
1. canMaterialiseQuick(scId)  → YES: add SC, mark solver needed
2. If NO: excitationSeverForRoom(scId)  → try again
3. If still NO: move REJECTED — vacuum wins, xon stays put
```

---

## 5. Demo Mode (flux-demo.js) — The Xon Choreography System

### Xon Lifecycle
6 persistent xons spawn on oct nodes. Three modes:

| Mode | When | Color | Movement |
|------|------|-------|----------|
| `oct` | Cruising octahedral cage | White | Flexible; bipartite matching |
| `tet` | Executing assigned loop | Quark color | Fixed 5-node sequence |
| `idle_tet` | Loitering in actualized tet | Quark color | Fixed sequence, no assignment |

Transitions: `oct → tet` (window assignment), `tet → oct` (loop complete at step 4), `oct → idle_tet` (congestion relief or boxed in)

### Loop Topologies (LOOP_SEQUENCES)
Given tet cycle `[a, b, c, d]`:
| Type | Name | Sequence | Nodes visited |
|------|------|----------|---------------|
| `pu` | Fork | a→b→a→c→a | 3 (hub-spoke) |
| `nd` | Lollipop | a→b→c→b→a | 3 (stem-loop) |
| `pd` | Ham CW | a→b→c→d→a | 4 (Hamiltonian) |
| `nu` | Ham CCW | a→d→c→b→a | 4 (reverse Ham) |

### Window/Tick System
- **1 cycle = 64 ticks** (16 windows × 4 ticks/window)
- **8 triple-face windows** (3 faces active) + **8 single-face windows** (1 face)
- Each window assigns quark types stochastically: proton = 2pu+1pd, neutron = 1nu+2nd
- Schedule reshuffled at cycle boundary

### Coordinated Move Planner (PHASE 1-5)
All moves planned before execution to prevent Pauli violations.

**PHASE 1** — Plan tet/idle_tet moves (fixed paths, highest priority)
- Approve if destination free; if blocked by oct xon, mark for forced vacate
- Vacuum negotiation: SC-only edges must be materialized or move revoked

**PHASE 2** — Coordinated oct movement
- Remove oct xons from `occupied` (enables swaps)
- Score candidates with momentum-aware heuristic
- Pre-filter: reject candidates where `canMaterialiseQuick` would fail
- **Kuhn's algorithm** (max bipartite matching): augmenting paths with arbitrary-depth backtracking
- Proactive congestion relief: if >4 xons on oct cage, demote lowest-scored to idle_tet
- Unassigned xons try `_startIdleTetLoop` (may manifest new tet voids)

**PHASE 3** — Execute all moves
- Oct moves FIRST (vacate destinations for tet xons)
- Then tet moves with final Pauli safety check
- If oct vacuum rejects → revoke dependent tet approvals

**PHASE 4** — Auto-return & scatter
- Return completed loops (step ≥ 4) to oct
- Scatter collisions: move extras to free neighbors or idle_tet

**PHASE 5** — Deadlock detection (non-fatal warning after 8 stuck ticks)

### Key Functions
| Function | Purpose |
|----------|---------|
| `_maxBipartiteAssignment(plans, blocked)` | Kuhn's algorithm for optimal xon→destination matching |
| `_getOctCandidates(xon, occupied, blocked)` | Scored oct neighbor candidates with momentum |
| `_executeOctMove(xon, target)` | Vacuum negotiation + SC materialization for oct hops |
| `_startIdleTetLoop(xon, occupied)` | Pauli-aware idle tet assignment; manifests new voids if needed |
| `_assignXonToTet(xon, face, quarkType)` | Oct→tet transition with BFS walk to face |
| `_walkToFace(xon, targetNodes)` | BFS pathfinding to nearest face node |
| `_returnXonToOct(xon)` | Tet→oct transition after loop completion |
| `_advanceXon(xon)` | Single hop along fixed loop sequence |

### Gluon System
- `_emitGluon(fromFace, toFace)` — creates white spark on oct edge between faces
- `_advanceGluons()` — moves gluons along oct path, materializes SCs with vacuum negotiation
- Gluons are visual connectors; they don't affect xon movement

---

## 6. Test Suite (flux-tests.js)

### LIVE_GUARD_REGISTRY — Unified Architecture (Single Source of Truth)
Each test lives in ONE registry entry in `LIVE_GUARD_REGISTRY` (array at top of flux-tests.js). The entry contains ALL logic for that test:

| Field | Purpose | Called by |
|-------|---------|-----------|
| `id` | Unique test ID (e.g. 'T19') | All systems |
| `name` | Human-readable description | UI panel |
| `init` | Extra state props | `_liveGuards` init |
| `convergence` | Stay null during grace | Grace promotion |
| `projected(states)` | Pre-move validation | Lookahead planner |
| `activate(g)` | Grace-end initialization | `_liveGuardCheck` |
| `snapshot(g)` | Pre-tick state capture | `_liveGuardSnapshot` |
| `check(tick, g, ctx)` | Post-move validation | `_liveGuardCheck` |

**To disable a test: remove its entry from the registry. That's ALL you need to do.** No check blocks, no separate code paths, no second file to edit. The `_liveGuardCheck()` dispatcher iterates the registry and calls each entry's `check()` function — there are no per-test if-blocks.

**`projected()` and `check()` are the same test in two modes.** `projected(states)` runs BEFORE moves with hypothetical future positions. `check(tick, g, ctx)` runs AFTER moves with actual state. Both live in the same registry entry — removing the entry disables both simultaneously.

### Test Inventory (T01-T28)
| ID | Name | What it tests |
|----|------|---------------|
| T01-T04 | Loop topologies | LOOP_SEQUENCES produce correct paths (fork, lollipop, ham CW/CCW) |
| T05 | Bipartite triple composition | Valid triples have 2A+1B or 1A+2B group split |
| T06 | Hadron type assignment | Proton = 2pu+1pd, Neutron = 1nu+2nd |
| T07 | Opposite-hadron decks | A-face singles get neutron types, B-face get proton |
| T08 | Schedule structure | 16 windows: 8 triples + 8 singles |
| T09 | Tet face data | All 8 faces have valid cycle[4] + scIds |
| T10 | Xon spawning | _spawnXon creates valid xon with correct properties |
| T11 | Xon advancement | _advanceXon correctly walks loop sequence |
| T12 | Persistent count | Exactly 6 live xons maintained |
| T13-T14 | No spawn/destroy | Xon count stable after init |
| T15 | Xon state | Valid sign ∈ {+1,-1} and mode ∈ {tet,oct,idle_tet} |
| T16 | Xon always has function | Tet xons have loop; oct xons on oct nodes |
| T16b | Idle only in actualized tets | idle_tet faces have all SCs active |
| T17 | Full tet coverage | All 8 faces visited within 4 cycles (256 ticks) |
| **T19** | **Pauli exclusion** | **LIVE GUARD**: max 1 xon per node |
| T20 | Never stand still | Every xon moves every tick |
| **T21** | **Oct cage permanence** | **LIVE GUARD**: oct SCs stay in at least one SC set |
| T22 | Hadronic composition | pu:pd ≈ 2:1, nd:nu ≈ 2:1 (converges over 1280 ticks) |
| T23 | Sparkle color | White in oct, quark color in tet |
| T24 | Trail color stability | Trail segments retain original color |
| T25 | Oct cage within 12 ticks | All oct SCs active by tick 12 |
| **T26** | **No unactivated SC traversal** | **LIVE GUARD**: SC traversals must use activated SCs |
| **T27** | **No teleportation** | **LIVE GUARD**: xons only move via connected edges |
| T28 | Lifespan slider | DOM integration works |

### Live Guard System (Unified Dispatcher)
- **Grace period**: 12 ticks (LIVE_GUARD_GRACE). Guards stay null (yellow "–") during grace.
- **Activation**: At tick 12, `activate()` called for entries that have it, non-convergence guards promoted to green.
- **Permanent failure**: Once a guard fails, it stays red and **halts the simulation** after 4-tick wind-down.
- **Snapshot system**: `_liveGuardSnapshot()` called BEFORE each tick captures xon positions and calls each entry's `snapshot()`. `_liveGuardCheck()` called AFTER iterates registry and calls each entry's `check(tick, g, ctx)`.
- **Movement guards**: Each `check()` function decides internally whether to skip window boundaries or require `ctx.prev` — no hardcoded categories in the dispatcher.
- **Convergence guards**: Entries with `convergence: true` stay null during grace promotion. Their `check()` handles timing internally.

### T26 Detail (most common failure)
Checks every xon movement `fromNode → toNode`:
1. Is there an SC on this edge? (`scPairToId.get(pairId(from, to))`)
2. Is there also a base edge? If yes, exempt (xon used base edge, not SC)
3. If SC-only: is SC in `activeSet`, `impliedSet`, or `xonImpliedSet`? If not → **FAIL**

---

## 7. Other Modules

### flux-nucleus.js — Nucleus Setup
`NucleusSimulator.simulateNucleus()`:
- Sets L2 lattice, finds central oct void (4-cycle of SCs)
- Maps 8 adjacent tets to faces, builds `_nucleusTetFaceData[face]` = `{voidIdx, allNodes, extNode, scIds, cycle}`
- Spawns 6 xons on oct nodes via `_initPersistentXons()`
- Populates `_octSCIds` (4 cage SCs) and `_octNodeSet` (oct vertex IDs)

### flux-voids.js — Void Detection & Rendering
- `voidNeighborData[]` — per-void: `{type:'tet'|'oct', nbrs, scIds, cycles}`
- `_nodeTetVoids[node]`, `_nodeOctVoids[node]` — per-node void membership
- Void sphere meshes colored by quark type during demo
- `updateVoidSpheres()` refreshes visibility/opacity per frame

### flux-rules-v2.js — Tournament & K-Complexity
- `QUARK_ALGO_REGISTRY[]` — pluggable movement strategies (stepQuark callbacks)
- `computeKComplexity()` — LZ76 compression of SC state string, normalized [0,1]
- `captureTemporalFrame()` — per-tick state snapshot for temporal K tracking
- Tournament fitness: 60% coverage evenness + 30% actualization rate + 10% stability

### flux-ui.js — Interactive Controls
- Jiggle mode: stochastic SC add/remove with strain rollback
- Camera orbit: spherical coordinates (theta, phi, r)
- Raycasting: InstancedMesh picking for sphere/SC selection
- Export: full simulation state serialization

---

## 8. Key Data Structures Reference

### Global SC Lookup
```
pairId(a, b)       → canonical "min:max" string key
scPairToId.get(pid) → SC ID (or undefined if no SC on this edge)
SC_BY_ID[scId]     → {id, a, b, stype, label}
scByVert[node]     → [{id, a, b, ...}, ...] SCs touching this node
baseNeighbors[node] → [{node, dirIdx}, ...] base-edge neighbors
```

### Xon Object
```javascript
{
  node, prevNode, sign,           // position + charge
  _mode: 'oct'|'tet'|'idle_tet', // current mode
  _assignedFace, _quarkType,     // tet assignment (null in oct)
  _loopSeq: [5 nodes],           // fixed path for current loop
  _loopStep: 0-4,                // progress (4 = complete)
  col, group, spark, sparkMat,   // Three.js visuals
  trail: [last 12 nodes],        // position history
  tweenT: 0-1,                   // interpolation progress
  alive: true|false,
}
```

### _nucleusTetFaceData[faceId]
```javascript
{
  voidIdx,                        // index into voidNeighborData
  allNodes: [4 nodes],            // all tet vertices
  extNode,                        // the non-oct apex node
  scIds: [SC IDs],                // shortcuts bounding this tet
  cycle: [a, b, c, d],           // ordered traversal (a=oct, b=ext, c=oct, d=oct)
}
```

---

## 9. Critical Doctrines

### Vacuum Doctrine
- The solver is the ONLY source of physical truth
- NEVER skip the solver to optimize
- NEVER predict solver output — just run it
- Shapes drive spheres (unified architecture)
- SCs must be unit-length after solving
- **KEPLER DENSITY VIOLATION = you didn't ask the physics solver if your move was okay first.** Every SC addition/removal MUST go through `canMaterialiseQuick()` before committing. If you see a Kepler violation, the root cause is always an unchecked SC mutation.
- **KEPLER DENSITY VIOLATION IS THE WORST ONE! ALWAYS TAKE SERIOUSLY.**
- **SOLUTION IS NEVER TO RELAX TIGHT THRESHOLD OF 0.01%; YOU MUST FIX YOUR PHYSICS.**
- **DO NOT DO THINGS LIKE `_demoActive` TO DISABLE CHECKS! Fix the root cause.**

### Movement Doctrine
- **NO MULTI-HOP PATHS.** A xon may move at most ONE hop per tick. Multiple hops in a single tick = FTL = teleportation. NEVER allow it.
- If a xon has already moved this tick, it MUST NOT be moved again by scatter, return-to-oct, or any other code path.
- If a Pauli collision arises from a moved xon, prefer moving the OTHER xon. If neither can move without multi-hop, revert the original move.

### Minimal Action Principle
- Relinquish ONLY what's necessary at window transitions
- Diff-based: remove old, add new, keep rest
- NO clear-and-rebuild (cascade deforms FCC geometry)

### Unified Architecture
- Demo MUST manage tet SCs in `xonImpliedSet`
- Must re-solve lattice so spheres respond physically
- Shapes → Spheres (never decouple rendering from physics)

---

## 10. Traversal Lock & SC Promotion

### Traversal Lock (`_traversalLockedSCs()`)
Returns the set of SC IDs that MUST NOT be removed from any set because a xon is currently using them:
- The SC connecting `prevNode → node` for every alive xon
- ALL face SCs for every alive tet/idle_tet xon's assigned face
- Used by `_relinquishFaceSCs`, window transition relinquishment, `_startIdleTetLoop` rollbacks, and `excitationSeverForRoom`

### SC Promotion (`_promoteFaceSCs(face)`)
When a xon is assigned to a tet face, any face SCs that are only in `impliedSet` (ephemeral, rebuilt each solver tick) get promoted into `xonImpliedSet` (persistent). This prevents the SC from vanishing on the next solver rebuild while the xon is mid-loop.
- Called by `_assignXonToTet` and `_startIdleTetLoop` at both assignment paths

### PHASE 4 Pauli Annihilation
PHASE 4 enforces Pauli exclusion absolutely:
1. Try scatter for ANY mode xon (not just oct)
2. If scatter fails, annihilate pairs unconditionally — no collision survives PHASE 4
3. Pauli exclusion trumps weak force confinement (T38 accepts Pauli annihilation)

### Vacuum Deadlock
- When too many SCs active, `canMaterialiseQuick` rejects all new SCs
- `_startIdleTetLoop` now manifests new tet voids (Pass 2) to create loiter space
- Deadlock detection is non-fatal (warn only, no halt)

### Tet Void Manifestation
- When oct cage is congested, `_startIdleTetLoop` proactively materializes missing SCs for non-actualized faces
- Sets `_idleTetManifested` flag → triggers solver rerun
- Rollback if any SC fails to materialize (respects traversal lock)

---

## 10.5. Known Bugs & Repro Steps

### Laser Pointer Trail (node 0 → opposite oct axis)
A long straight trail line extends from node 0 through the oct to the opposite axis.
**Repro:**
1. Refresh the page
2. Select Rule 9 (Adaptive ejection √n)
3. Select the t260 council run from dropdown
4. Pause at the end of the run
5. Observe: laser pointer trail visible from node 0 to opposite axis of oct

**Root cause (under investigation):** Trail frozen positions contain a discontinuous jump — likely a backtracker snapshot restore that repositions xons without properly rewinding the trail history, creating a segment that spans the entire oct diameter.

---

## 10.6. Sweep Architecture Doctrine

**Sweep mode is the ONLY execution mode. Period.**

There is no separate "replay mode" or "test mode." The sweep (`_runTournament` is dead — use `startSweepTest` → `startDemoLoop` directly) is always running. The only difference between a fresh run and a council replay is that a council replay starts with pre-seeded moves (snapshots loaded from IDB onto the redo stack). That's it.

**Council replay = sweep with pre-seeded redo stack.** The redo drain plays the saved snapshots, then live execution continues seamlessly. There is no mode transition, no separate controller.

**The replay test pipeline (🧪 button) runs WITHIN the sweep**, not as a competing controller. It observes the sweep's behavior and validates guard integrity. The test infrastructure is temporary scaffolding that will be commented out once replay fidelity is proven.

**NEVER create a parallel execution system that fights with the sweep for control of the demo.** One execution loop. One controller. One source of truth for what tick we're on.

---

## 11. Development Conventions

- **Cache busting**: After editing any `src/*.js` file, bump `?v=N` in `flux-v2.html` script tag
- **No module system**: All files share global scope. Load order in HTML matters.
- **Solver calls are expensive**: Minimize `_solve()` calls. Use `canMaterialiseQuick()` for dry-run checks.
- **Test after changes**: Run demo, check left sidebar for test results (green checkmarks). Live guards (T19/T21/T26/T27) monitor continuously during demo.
- **pairId(a,b)**: Always use this for SC edge lookup (canonical `min:max` format)
- **_occAdd/_occDel**: Use these helpers for occupied map manipulation (count-based, not boolean)

### Slider Defaults & Play-Mode Overrides
Visualization slider defaults now live in **one centralized constant**:

- **`DEMO_VISUAL_DEFAULTS`** in `flux-demo-state.js`: Single source of truth for all visual slider overrides applied when the user clicks play (and for council replay init). Both `startDemoLoop()` in `flux-demo.js` and the replay init in `flux-tests.js` reference this constant.
- **HTML defaults** (`flux-v2.html`): `value="..."` attributes on `<input>` sliders control what the user sees on page load (before pressing play). These can differ from `DEMO_VISUAL_DEFAULTS`.
- **`_demoOpDefaultsApplied`** flag (in `flux-demo-state.js`): Ensures defaults are applied only on the FIRST `startDemoLoop()` call. Mid-sweep seed transitions do NOT reapply — user slider changes persist across seeds. The flag resets in `stopDemo()` only when `_sweepActive` is false.

When changing a default: update `DEMO_VISUAL_DEFAULTS` for play-mode, and the HTML `value=` for page-load. They are intentionally independent.

Camera defaults live in `flux-ui.js` (`sph={theta, phi, r}` and `panTarget`).

### IDB Key Convention (`_blacklistRuleKey`)
Keys for IndexedDB storage (blacklist, autosave, council) are built by **concatenating only active rules** in order. A rule that is OFF simply doesn't appear in the key. This means:
- Enabling a new rule appends a new tag → fresh keyspace (no stale data).
- Disabling it removes the tag → key matches what it was before the rule existed.
- Adding a new rule to the codebase doesn't break existing keys (OFF = absent = same key as before).

Pattern: `let k = base; if (rule) k += '|tag'; return k;`

### Blacklist Bucket System

The blacklist records "dead end" state fingerprints from previous sweep seeds. **Too large to fit in memory**, so it's split into time-windowed buckets that load on-demand from IndexedDB.

**Fingerprint**: Canonical string of all xon moves at a tick (`"X0:5->9|X1:stay@0|..."`, computed by `_computeTickFingerprint()`).

**Bucket**: Contains all fingerprints for ticks `[i*64, i*64+63]` (64-tick window, one demo cycle). Each bucket is a separate IDB key: `baseKey|bl|bucketIdx`.

**Data flow**:
1. **Startup**: `_blIDBLoad()` loads metadata only (total count, bucket count), NOT fingerprints. Returns empty `_sweepBlacklist` Map.
2. **Prefetch**: `_blPrefetchBucket(lvl, bucketIdx)` loads one bucket's fingerprints from IDB → merges into `_sweepBlacklist`. Bucket 0 is eagerly prefetched; others loaded when a tick in their range is reached.
3. **Check**: `_btRecordFingerprint()` checks `_sweepBlacklist.get(tick).has(fp)`. If hit → state is dead, skip it. Increments `_sweepBlacklistHits` / `_sweepBlacklistHitsSeed`.
4. **Blacklist**: At seed end, all `_btTriedFingerprints` are promoted into `_sweepBlacklist`.
5. **Save**: `_blIDBSaveBlacklist()` partitions `_sweepBlacklist` back into buckets and writes to IDB. Debounced to 2s.

**Memory model**: Only loaded buckets are in RAM (typically 1-2 buckets = a few MB). Full dataset stays in IDB (can grow to 10s of MB). `_blLoadedBuckets` Set tracks which are in memory — each loaded exactly once per session.

**Bypass during replay**: Blacklist check is skipped during council replay until past the member's recorded peak (`_sweepReplayActive && tick <= peak`). This allows deterministic replay without blacklist interference.

---

## 12. Roadmap

### 12.1 Matter/Anti-Matter Distinction
Matter/anti-matter correspond to the two possible winding directions of the shortcut equator of the oct. During the opening phase, a 'merry go round' motion is scheduled along the shortcut equator. A unit test + live guard must ensure the winding direction stays constant for the entire simulation.

### 12.2 Updated Fermion Model

| Type | Sequence | Identity |
|------|----------|----------|
| Fork | a → b → a → c → a | proton up 1 / neutron down 1 |
| Hook (renamed from "lollipop") | a → b → c → b → a | proton up 2 / neutron down 2 |
| Hamiltonian CW | a → b → c → d → a | proton down (anchor) |
| Hamiltonian CCW | a → d → c → b → a | neutron up (anchor) |

**H-2 atom**: 2 "anchor" quarks (proton down, neutron up) + 4 "follower" quarks (2× proton up, 2× neutron down). Follower types 1 (Fork) and 2 (Hook) should occur in equal measures on average.

**Bipartite constraint**: Quarks of the same hadron must never occur simultaneously as tets with a shared edge. Proton and neutron constituents must occupy separate parts of the bipartite graph when they appear together.

**Coverage target**: Each face should tend towards equal parts pu1, pu2, nd1, nd2, pd, nu.

**Quark loops can only be initiated by oct xons.**

**New colors**:
| Type | Color | Hex |
|------|-------|-----|
| proton down | cyan | #00FFFF |
| proton up 1 | medium blue | #0040FF |
| proton up 2 | emerald | #00FF40 |
| neutron up | red | #FF0000 |
| neutron down 1 | goldenrod | #FFBF00 |
| neutron down 2 | hot pink | #FF00BF |
| weak | violet | #7F00FF |
| gluon | yellow-green | #80FF00 |

Update trails, shapes, sparks, and legend with these colors.

**Permutations** (prune if matter/antimatter directionality forbids):

Fork: `abaca`, `acaba`, `abada`, `adaba`, `acada`, `adaca`
Hook: `abcba`, `acbca`, `abdba`, `adbda`, `acdca`, `adcda`
Ham CW: `abcda`, `acdba`, `adbca`
Ham CCW: `adcba`, `acbda`, `abdca`

### 12.3 Tournament Overhaul

**Xon balance panel**: Give each xon a score for directional balance across all 10 directions. Surface in the xon panel (right side).

**Fitness criteria (priority order)**:
1. Anchor quark balance (even face distribution)
2. Follower quark balance (even face distribution)
3. Anchor/follower ratios per orientation (target 1:2)
4. Follower/follower ratios per orientation (target 1:1)
5. Quark frequency (maximize quarks generated per tick)
6. Periodicity (regularity of oct orientation changes over time)
7. Xonic balance (movement stats + mode stats per xon, displayed in xon panel)

Overhaul tunable genes to optimize for these criteria.

### 12.4 RL Feasibility Investigation

Consider replacing genetic algorithm tournament with reinforcement learning: "graph in, xon moves out." Feed the graph + pre-computed legal moves (including shortcuts with severance costs). Optimize for balanced system. Investigate browser-based RL with GPU acceleration.

### 12.5 Fix Larger Lattice Behavior

Fermions never form on larger lattices. Investigate whether compute-load shortcuts that reduce simulation accuracy were introduced. Remove any such shortcuts.

### 12.6 Choreographer Redesign

**Priority-based quark scheduler**: Replace pattern machine complexity with a function that gives a prioritized list of which quarks are needed right now. The oct can then make optimal decisions, including handling "unscheduled" fermions.

**SC planning intelligence**: The choreographer must know when it needs to create a shortcut to actualize a needed tet. Relevant shortcuts are always from node 0 or node 4. Must plan correct orientation (starting node a), and plan around geometrically mutually exclusive tet schedules. This is the "air traffic controller" problem — the core challenge of choreography.

### 12.7 Planck-Second vs Tick Distinction

- **tick**: Total number of steps taken by any given xon
- **Planck second**: Total number of ticks which featured at least one lattice deformation event (SC added/subtracted)

Display both. Swap the current naming: what was "tick" becomes the above definition.

**⚠️ CURRENT NAMING IS SWAPPED IN CODE vs UI:**
- UI shows `_demoTick` as "Planck Seconds" and `_planckSeconds` as "Flux Events"
- In code: `_planckSeconds` = flux events (SC deformations), `_demoTick` = total ticks
- When polling programmatically, use `_demoTick` to match the number displayed as "Planck Seconds" on screen
- The replay test pipeline uses `_demoTick` for milestones (100/200/300) to match what the user sees

### 12.8 Re-Center Lattice Around Nuclear Octa

Currently centered around a single node. Should be centered around the nuclear octa itself, giving symmetrical space above and below the nucleus for weak force operation.

---

## 13. Replay System — Architecture & Fidelity Guide

**Replays are the primary output of this simulation.** A replay is a complete, deterministic record of a simulation run that can be scrubbed forward/backward and re-played from any point. Replay fidelity is non-negotiable — a replay from t=0 must be structurally identical to the original live run.

### Goal
Streamline onboarding for anyone working on replay correctness. The replay pipeline touches multiple files and has subtle invariants. This section documents the full data flow so you don't have to rediscover it.

### Data Flow: Live Play → IDB → Replay

1. **Live play**: Each tick, `_btCreateSnapshot()` (flux-demo-backtrack.js) captures the full simulation state. Snapshots are stored in `_btSnapshots[]` (mutable, used by backtracker) and `_councilSnapArchive[]` (forward-only, only appended when `_demoTick > _maxTickReached`).

2. **Snapshot storage — O(N) not O(N²)**: Snapshots do NOT store trail arrays. Each xon in a snapshot stores:
   - `trailLen` (integer) — how long the trail was at this tick
   - `_tNode`, `_tCol`, `_tRole`, `_tPos` — the single trail entry for this tick, recorded from the live trail at snapshot time (already reflects `_trailRecolor` wash)
   - `_role` — explicit role string, never inferred

   This keeps memory O(N) across all snapshots. Trail arrays live only on the live xon objects.

3. **Save to IDB**: `_blIDBSaveCouncilMember()` (flux-tests.js) serializes snapshots via `_serializeSnapshot()` which strips any trail arrays and preserves `_t*` fields. No trail reconstruction at save time.

4. **Load from IDB**: `_deserializeSnapshot()` produces snapshots with `trail: null` (modern) or `trail: [...]` (legacy). The `_t*` fields and `trailLen` survive the round-trip.

5. **Replay restore**: `_btRestoreSnapshot()` (flux-demo-backtrack.js) handles two paths:
   - **Rewind** (trail.length > trailLen): truncate trail arrays
   - **Forward** (trail.length < trailLen): push `_tNode`/`_tCol`/`_tRole`/`_tPos` from the snapshot — direct read, no inference
   - **Wash at transitions**: If `_tRole` is gluon/weak and previous trail entry was 'oct', walk back through consecutive oct entries and overwrite (mirrors `_trailRecolor` in live play). Only fires once at the transition, not on every gluon tick.

### Critical Invariants

- **First playthrough ≡ replay from t=0**: These must produce structurally identical trails. If they diverge, the replay is broken.
- **`_tCol`/`_tRole` are recorded, not inferred**: Role inference from `_mode`/`_quarkType` is fragile (stale `_quarkType` on gluon xons). The snapshot records the actual trail entry values from the live xon at snapshot time.
- **`_councilSnapArchive` is the source of truth for saves**: It's forward-only (never popped by backtracking). `_btSnapshots` is mutable and may contain backtracked states.
- **Auto-retry creates multiple runs**: When auto-retry-best restarts from a council replay, `_councilSnapArchive` gets cleared and rebuilt from t=0. If multiple retries happen before save, the archive may contain multiple t=0→t=x runs concatenated. `_blIDBSaveCouncilMember` trims to the last contiguous run (last index where `tick` decreases) before writing.
- **Append-only saves**: When extending a replay (descendant surpasses ancestor), read existing IDB data, keep ancestor's snapshots, append new ticks beyond `_replayAncestorPeak`. The combined result is also trimmed for multi-run contamination.

### Common Bugs & How They Manifest

| Symptom | Root Cause | Fix Location |
|---------|-----------|-------------|
| Trails missing on replay | Snapshots stored `trailLen` but `_t*` fields missing | `_btCreateSnapshot` — ensure `_tNode`/`_tCol`/`_tRole`/`_tPos` recorded |
| Wrong colors on replay from t=0 | Forward restore inferring role instead of reading `_tRole` | `_btRestoreSnapshot` forward path |
| "Gluon clouds" at end of replay | Wash running on every gluon tick instead of only at transitions | Check `wasOct` transition detection in forward restore |
| Replay bouncing t=0→t=x→t=0 | Multiple auto-retry runs in IDB | Trim logic in `_blIDBSaveCouncilMember` |
| Trail data lost on descendant extension | Append-only save not chaining correctly | `_blIDBSaveCouncilMember` append-only path |
| Stale white xon colors | `_role` not recorded, color derived from inference | Ensure `_role` explicit in `_btCreateSnapshot`, `_serializeSnapshot` |

### Key Files
- **flux-demo-backtrack.js**: `_btCreateSnapshot`, `_btRestoreSnapshot` — snapshot lifecycle
- **flux-tests.js**: `_serializeSnapshot`, `_deserializeSnapshot`, `_blIDBSaveCouncilMember`, `_saveCurrentRunToCouncil` — IDB persistence
- **flux-demo.js**: `_councilSnapArchive` population (inside `_demoTick > _maxTickReached` block), autosave milestone trigger
- **flux-demo-state.js**: `_btSnapshots`, `_councilSnapArchive` declarations

---

### Trail Color Invariant
**The color of an xon's most recent trail segment is always the same color as the xon.** Enforced by live guard T93. `_trailPush` records `_xonRole(xon)` as the entry's `.role`; `_trailRecolor` updates it on mode transitions. At tick end, `trail[trail.length-1].role === _xonRole(xon)` must hold for every alive xon.

### Replay Integrity Test (🧪 button)
Automated pipeline: record→save→reload→replay w/guards→scrub→extend, targeting tick milestones 100→200→300. Uses `_demoTick` (not `_planckSeconds`) to match the "Planck Seconds" number shown on screen. When polling simulation state programmatically, always cross-check with a screenshot — the variable names in code do NOT match the UI labels (see §12.7).

**⚠️ Auto-retry best is enabled during the test.** The recording phase runs a normal sweep with auto-retry-best checked. This means the sweep may restart from a council replay mid-recording (when a new best surpasses the old one). This could be a source of replay corruption — the `_councilSnapArchive` may contain spliced snapshots from multiple runs/seeds. Investigate whether auto-retry-best introduces snapshot discontinuities.

### 2026-03-16: Trail Architecture Simplification & Replay Fidelity (In Progress)

**What changed**: Replaced 4 parallel trail arrays (`trail[]`, `trailColHistory[]`, `_trailRoleHistory[]`, `_trailFrozenPos[]`) with a single unified array of `{node, role, pos}` entry objects. Color is now derived at render time via `_roleToColor(entry.role)` — never stored. Snapshot format changed from broken `_tNode/_tCol/_tRole/_tPos` single-entry fields to `_tDelta` (array of all entries pushed this tick) + `_tRecolor` (role-only wash). `_trailLenAtTickStart` set once per tick drives delta computation.

**Two replay paths exist — they diverge on trail data**:

| Path | Source | Trail data | Status |
|------|--------|-----------|--------|
| **Dropdown replay** (IDB → `_hydrateCouncilMember` → redo drain) | `_councilSnapArchive` (tick-END snapshots with correct `_tDelta`) | Full trail arrays reconstructed during hydration | Working |
| **Scrub-to-t=0** (`_timelineScrubTo` → redo drain) | `_btSnapshots` (tick-START snapshots, `_tDelta` always null) | Trail reconstruction added to `resumeDemo()` before drain | Under test |

**Root cause of scrub-to-t=0 failures**: `_btSnapshots` are created at tick START (before choreography runs). At that point, `_tDelta` is correctly null (no moves yet) and `trailLen` reflects the trail length BEFORE moves. During redo forward replay, the delta-based restore path in `_btRestoreSnapshot` can't grow trails because there are no deltas. The fix: `resumeDemo()` now iterates the redo stack chronologically before draining and reconstructs full trail arrays by replaying `trailLen` growth + deltas (same pattern as IDB hydration).

**Safety nets in `_btRestoreSnapshot`**: (1) `while (x.trail.length < tLen)` pads missing entries with xon's current node. (2) Ensures trail head always matches `s.node`. These catch any remaining delta gaps but can create non-unit-distance segments.

**What's validated**: T59 (trail head matches node) passes. T57 (segment length) is the remaining risk — synthetic trail entries from padding may span non-adjacent nodes. The `resumeDemo()` reconstruction should eliminate this by building trails incrementally from chronological snapshots.

**What still needs testing**: (1) Fresh 100+ run → scrub to t=0 → play → verify T57/T59 pass and live continuation works. (2) Dropdown replay of same run → verify trails match. (3) Auto-retry-best on reload → verify redo drain completes cleanly.

---

## 15. Pre-Push Checklist

**Always manually bump the version number before pushing.** This is intentionally manual to track deliberate pushes vs accidental ones.

**Version format**: `v{major}.{minor}.{patch}` — displayed at bottom of left panel in `flux-v2.html`.

**Incrementing rules**:
- Increment patch: `v0.2.0` → `v0.2.1`
- Patch rolls over at 99: `v0.2.99` → `v0.3.1`
- Minor rolls over at 99: `v0.99.99` → `v1.0.1`

**Process**: Before every push to GitHub, find the version string in `flux-v2.html` (bottom of left panel div) and increment the patch number. Do NOT automate this — the manual step ensures every push is intentional.

**Remove the pre-commit hook** — version bumping is manual only.
