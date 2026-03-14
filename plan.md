# Choreography Movie Export/Import Plan

## Goal
Press demo → run 10 Planck seconds → export → refresh → import → exact visual playback.

## Design: Minimal "movie" format (separate from debug tick log)

The existing `_tickLog` stays as-is for debugging. The movie export is a lean format with only what's needed for visual replay.

### Per-tick frame data
```
{
  xons: [ { node, mode, quark } × 6 ],   // 6 integers + 2 short strings each
  active: [SC IDs],                        // activeSet
  xonImplied: [SC IDs],                    // xonImpliedSet
  implied: [SC IDs],                       // impliedSet
  pos: <delta-compressed>                  // node positions
}
```

**Derived at playback (not stored):**
- `prevNode` — previous tick's `node` for each xon
- Trail — rolling window of last N nodes from position history
- Trail colors — derived from xon `mode`/`quark` at each historical tick
- Tween interpolation — `prevNode` → `node` with positions from `pos[][]`

### Position delta compression
- **Tick 0**: full `pos[][]` — 107 × 3 floats, stored as `[[x,y,z], ...]`
- **Subsequent ticks**: only nodes that moved by >1e-4, stored as `[[nodeIdx, x, y, z], ...]`
- On import: carry forward last full frame, apply deltas

### Header
```
{
  version: 3,
  latticeLevel: 2,
  nodeCount: 107,
  totalTicks: N,
  params: { ...choreoParams },
  frames: [ ...per-tick data ]
}
```

## Changes

### 1. Record movie frames (flux-demo.js)
New `_movieFrames[]` array. At end of each `demoTick()`, push a lean frame:
- `xons`: node, mode, quark (6 entries)
- `active`, `xonImplied`, `implied`: SC ID arrays
- `pos`: delta vs previous frame (or full if tick 0)

Keep `_lastMoviePos` for delta computation.

### 2. Export movie (flux-demo-ui.js)
New `exportMovie()` — builds the lean JSON from `_movieFrames[]`, triggers file download.
- Rewire `btn-export` to call `exportMovie()` instead of `exportState()`
- Change button text to "Export"

### 3. Import movie (flux-demo-ui.js)
New `importMovie()` — file picker, parse JSON, validate version 3, store as `_importedMovie`.

### 4. Playback engine (flux-demo-ui.js)
`_playbackMode`, `_playbackFrame`, `_importedMovie` globals.

`playbackApplyFrame(idx)`:
1. Reconstruct `pos[][]` — walk deltas from tick 0 up to `idx` (or cache last full pos)
2. Set `activeSet`, `xonImpliedSet`, `impliedSet` from frame
3. Write `pos[]` into solver's position array
4. For each xon: set node, derive prevNode from frame `idx-1`, set mode/color
5. Build trails: last 12 frames' node positions + colors
6. Call `rebuildShortcutLines()`, `updateVoidSpheres()`, `updateSpheres()`

Transport: reuse existing pause/play/reverse buttons. Stop exits playback.

### 5. UI (flux-v2.html)
- Replace `btn-export` text → "Export"
- Add `btn-import-log` button + hidden `<input type="file">`

### 6. Rewire btn-export (flux-voids.js)
Change click handler from `exportState` → `exportMovie`

## File changes
| File | Changes |
|------|---------|
| `flux-demo.js` | Add `_movieFrames[]`, lean frame recording at end of `demoTick()` |
| `flux-demo-ui.js` | `exportMovie()`, `importMovie()`, `playbackApplyFrame()`, playback loop |
| `flux-demo-state.js` | Declare `_movieFrames`, `_lastMoviePos`, `_playbackMode`, `_playbackFrame`, `_importedMovie` |
| `flux-v2.html` | Replace btn-export text, add Import button + hidden file input |
| `flux-voids.js` | Rewire btn-export click handler |

## Size estimate
Per tick: ~6×3 = 18 bytes (xons) + ~20 SC IDs × 2 bytes = 40 bytes + ~30 changed nodes × 16 bytes = 480 bytes ≈ **~500 bytes/tick**.
640 ticks ≈ **~320KB** uncompressed JSON. Reasonable.
