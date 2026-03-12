# Plan: Backtracking Choreographer

## Core Idea
Instead of predicting collisions with a speculative lookahead, **run the choreography forward and rewind when violations occur**. This turns the choreographer into a constraint solver with chronological backtracking.

## How It Works

### Tick Flow (modified)

```
1. Snapshot full state (xon positions, modes, SC sets, solver positions)
2. Run normal PHASE 1-5 choreography
3. Run live guards (T19, T20, T26, T27)
4. If guards pass → commit, discard snapshot, continue
5. If guards FAIL → REWIND to snapshot, try different choices
6. If all choices exhausted at tick T → rewind to tick T-1 snapshot
7. Max backtrack depth: 3 ticks (configurable)
```

### State Snapshot (what to capture)

Per xon (6 xons):
- `node`, `prevNode`, `_mode`, `_assignedFace`, `_quarkType`
- `_loopSeq`, `_loopStep`, `_movedThisTick`, `col`
- `trail` array (copy), `trailColHistory` (copy)

Global SC state:
- `activeSet` (copy)
- `xonImpliedSet` (copy)
- `impliedSet` (copy)
- `_scAttribution` (copy)

Solver:
- `pos` array (copy of vertex positions)
- `_moveRecord` (clear on rewind)

### Choice Tracking

Each tick's PHASE 2 bipartite matching produces an assignment. To try "different choices," we need to **exclude the failed assignment** and re-run matching.

Structure:
```js
_backtrackState = {
    depth: 0,              // current backtrack depth (0 = normal)
    maxDepth: 3,           // max ticks to rewind
    snapshots: [],         // stack of state snapshots
    excludedMoves: [],     // per-depth: Set of "xonIdx:destNode" strings to exclude
}
```

When a violation occurs:
1. Pop snapshot from stack, restore state
2. Add the offending move(s) to `excludedMoves[depth]`
3. Re-run the tick with those moves excluded from candidate lists
4. If ALL candidates for a xon are excluded → increment depth, pop another snapshot
5. If depth > maxDepth → accept the violation (halt as before)

### Where to Hook In

**Snapshot**: At the start of `demoTick()`, after clearing `_movedThisTick` flags but before PHASE 1.

**Violation detection**: Replace the halt in `_liveGuardCheck()` with a rewind signal:
```js
// Instead of: simHalted = true;
// Do: _rewindRequested = true;
```

**Rewind loop**: Wrap the tick body in a retry loop:
```js
async function demoTick() {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        _saveSnapshot();
        _runTickBody();  // PHASE 1-5

        if (!_rewindRequested) break;  // clean tick, commit

        // Violation detected — rewind
        _restoreSnapshot();
        _excludeFailedMoves();
        _rewindRequested = false;
    }
}
```

### What Moves to Exclude on Rewind

From the guard failure, we know:
- **T19 (Pauli)**: Which node has 2+ xons → exclude the move that put the later xon there
- **T20 (stuck)**: Which xon is stuck → exclude its current position (force it to move differently)
- **T26 (unactivated SC)**: Which xon traversed a bad SC → exclude that specific edge
- **T27 (teleport)**: Which xon teleported → exclude that move

The `_moveTrace` array already records every move with `{xonIdx, from, to, path}` — perfect for identifying which move to exclude.

### Performance

- Snapshot: ~6 xon copies + 3 Set copies + pos array copy ≈ negligible
- Rewind: restore + re-run tick ≈ 2x tick cost
- Worst case (3 backtracks × ~6 retries): ~18x tick cost for one bad tick
- Normal case (no violations): 1 snapshot overhead per tick ≈ <0.1ms
- This is fine at MAX speed since ticks are ~17ms each

### What This Replaces

The entire existing lookahead system (`_lookahead`, `_lookaheadTetPath`, `_verifyPlan`, 2-step cooperative verification) becomes **unnecessary** once backtracking works. We can remove it incrementally — keep it as a heuristic for now, let backtracking be the safety net.

## Implementation Steps

1. **Add `_saveSnapshot()` / `_restoreSnapshot()`** — capture and restore full choreography state
2. **Add `_backtrackState` tracking** — excluded moves stack, depth counter
3. **Modify `_liveGuardCheck()`** — set `_rewindRequested` flag instead of `simHalted` (for T19/T20 only, keep halt for other guards)
4. **Wrap tick body in retry loop** — snapshot → run → check → rewind or commit
5. **Filter excluded moves in PHASE 2** — skip candidates that match `excludedMoves[depth]`
6. **Apply collision-avoidance hierarchy on rewind** — when excluding an oct move, first try tet diversion, then weak ejection
7. **Bump cache busters, verify**
