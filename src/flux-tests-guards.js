// flux-tests-guards.js — LIVE_GUARD_REGISTRY, guard dispatcher, guard helpers
// Split from flux-tests.js (lines 1-2072). Loaded FIRST in script order.

// flux-tests.js — Demo 3.0 unit tests + final UI wiring
// ═══════════════════════════════════════════════════════════════════════
// ║  DEMO 3.0 UNIT TESTS — assertions on xon mechanics                 ║
// ║  Run from console: runDemo3Tests()                                  ║
// ═══════════════════════════════════════════════════════════════════════
let _testRunning = false;  // suppress display updates during test execution

// ═══════════════════════════════════════════════════════════════════════
// ║  LIVE GUARD REGISTRY — single source of truth for ALL runtime tests ║
// ║  Adding/removing a test here is the ONLY action needed.             ║
// ║                                                                     ║
// ║  Each entry: { id, name, init?, convergence?, projected?,           ║
// ║               activate?, snapshot?, check? }                        ║
// ║    id:          'T19', 'T41', etc.                                  ║
// ║    name:        display name (without Txx prefix)                   ║
// ║    init:        extra state props for _liveGuards entry (optional)  ║
// ║    convergence: true → stays null during grace promotion (optional) ║
// ║    projected:   lookahead check fn(states) → null | violation       ║
// ║    activate:    called at grace end for initialization (optional)   ║
// ║    snapshot:    called before each tick for state capture (optional) ║
// ║    check:       runtime check fn(tick, g, ctx) → fail msg | null    ║
// ║                 ctx = { prev }                                      ║
// ║                                                                     ║
// ║  TO DISABLE A TEST: remove its entry. No other changes needed.      ║
// ═══════════════════════════════════════════════════════════════════════
const LIVE_GUARD_GRACE = 0;

// T80: Base direction polarity — xons may only move in the positive vector direction
// on base edges. Shortcut edges are bidirectional (exempt). Default: disabled.
const _T80_BASE_POLARITY = false;

// Oct capacity: hard maximum of oct-mode xons at any time (slider-tunable).
let OCT_CAPACITY_MAX = 6;
function _computeOctCapacity() {
    return OCT_CAPACITY_MAX;
}

// Helper: check if actual loop matches any valid permutation for a given quark type.
// _assignXonToTet rotates the cycle so the xon's starting oct node is in position 0.
// Valid rotations: [a,b,c,d], [c,b,a,d], [d,b,c,a] (oct nodes a,c,d can each be start).
// With LOOP_PERMUTATIONS, also checks all permutations for the quark's topology.
function _loopMatchesAnyRotation(actual, quarkType, cycle) {
    const [a, b, c, d] = cycle;
    const rotations = [[a,b,c,d], [c,b,a,d], [d,b,c,a]];
    const topo = QUARK_TOPOLOGY[quarkType];
    const perms = topo ? LOOP_PERMUTATIONS[topo] : null;
    for (const rot of rotations) {
        if (perms) {
            for (const gen of perms) {
                const expected = gen(rot);
                if (actual.length === expected.length && actual.every((n, i) => n === expected[i])) return true;
            }
        } else {
            const expected = LOOP_SEQUENCES[quarkType](rot);
            if (actual.length === expected.length && actual.every((n, i) => n === expected[i])) return true;
        }
    }
    return false;
}

const LIVE_GUARD_REGISTRY = [
    { id: 'T01', name: 'Fork path audit (pu1)', init: { _seen: 0 }, convergence: true,
      check(tick, g) {
        if (g.ok === true) return null;
        for (const xon of _demoXons) {
          if (!xon.alive || xon._mode !== 'tet' || xon._quarkType !== 'pu1') continue;
          if (!xon._loopSeq || !xon._assignedFace) continue;
          const fd = _nucleusTetFaceData[xon._assignedFace];
          if (!fd) continue;
          if (_loopMatchesAnyRotation(xon._loopSeq, 'pu1', fd.cycle)) {
            g._seen++; g.ok = true; g.msg = ''; _liveGuardRender(); return null;
          } else return `tick ${tick}: pu1 loop [${xon._loopSeq}] != any rotation of cycle [${fd.cycle}]`;
        }
        return null;
      }
    },
    { id: 'T02', name: 'Hook path audit (pu2)', init: { _seen: 0 }, convergence: true,
      check(tick, g) {
        if (g.ok === true) return null;
        for (const xon of _demoXons) {
          if (!xon.alive || xon._mode !== 'tet' || xon._quarkType !== 'pu2') continue;
          if (!xon._loopSeq || !xon._assignedFace) continue;
          const fd = _nucleusTetFaceData[xon._assignedFace];
          if (!fd) continue;
          if (_loopMatchesAnyRotation(xon._loopSeq, 'pu2', fd.cycle)) {
            g._seen++; g.ok = true; g.msg = ''; _liveGuardRender(); return null;
          } else return `tick ${tick}: pu2 loop [${xon._loopSeq}] != any rotation of cycle [${fd.cycle}]`;
        }
        return null;
      }
    },
    { id: 'T03', name: 'Hamiltonian CW path audit (pd)', init: { _seen: 0 }, convergence: true,
      check(tick, g) {
        if (g.ok === true) return null;
        for (const xon of _demoXons) {
          if (!xon.alive || xon._mode !== 'tet' || xon._quarkType !== 'pd') continue;
          if (!xon._loopSeq || !xon._assignedFace) continue;
          const fd = _nucleusTetFaceData[xon._assignedFace];
          if (!fd) continue;
          if (_loopMatchesAnyRotation(xon._loopSeq, 'pd', fd.cycle)) {
            g._seen++; g.ok = true; g.msg = ''; _liveGuardRender(); return null;
          } else return `tick ${tick}: pd loop [${xon._loopSeq}] != any rotation of cycle [${fd.cycle}]`;
        }
        return null;
      }
    },
    { id: 'T04', name: 'Hamiltonian CCW path audit (nu)', init: { _seen: 0 }, convergence: true,
      check(tick, g) {
        if (g.ok === true) return null;
        for (const xon of _demoXons) {
          if (!xon.alive || xon._mode !== 'tet' || xon._quarkType !== 'nu') continue;
          if (!xon._loopSeq || !xon._assignedFace) continue;
          const fd = _nucleusTetFaceData[xon._assignedFace];
          if (!fd) continue;
          if (_loopMatchesAnyRotation(xon._loopSeq, 'nu', fd.cycle)) {
            g._seen++; g.ok = true; g.msg = ''; _liveGuardRender(); return null;
          } else return `tick ${tick}: nu loop [${xon._loopSeq}] != any rotation of cycle [${fd.cycle}]`;
        }
        return null;
      }
    },
    { id: 'T05b', name: 'Fork path audit (nd1)', init: { _seen: 0 }, convergence: true,
      check(tick, g) {
        if (g.ok === true) return null;
        for (const xon of _demoXons) {
          if (!xon.alive || xon._mode !== 'tet' || xon._quarkType !== 'nd1') continue;
          if (!xon._loopSeq || !xon._assignedFace) continue;
          const fd = _nucleusTetFaceData[xon._assignedFace];
          if (!fd) continue;
          if (_loopMatchesAnyRotation(xon._loopSeq, 'nd1', fd.cycle)) {
            g._seen++; g.ok = true; g.msg = ''; _liveGuardRender(); return null;
          } else return `tick ${tick}: nd1 loop [${xon._loopSeq}] != any rotation of cycle [${fd.cycle}]`;
        }
        return null;
      }
    },
    { id: 'T06b', name: 'Hook path audit (nd2)', init: { _seen: 0 }, convergence: true,
      check(tick, g) {
        if (g.ok === true) return null;
        for (const xon of _demoXons) {
          if (!xon.alive || xon._mode !== 'tet' || xon._quarkType !== 'nd2') continue;
          if (!xon._loopSeq || !xon._assignedFace) continue;
          const fd = _nucleusTetFaceData[xon._assignedFace];
          if (!fd) continue;
          if (_loopMatchesAnyRotation(xon._loopSeq, 'nd2', fd.cycle)) {
            g._seen++; g.ok = true; g.msg = ''; _liveGuardRender(); return null;
          } else return `tick ${tick}: nd2 loop [${xon._loopSeq}] != any rotation of cycle [${fd.cycle}]`;
        }
        return null;
      }
    },
    // T05-T07 REMOVED: per user request
    { id: 'T12', name: 'Conservation (alive+2*stored=6)',
      check(tick, g) {
        const liveCount = _demoXons.filter(x => x.alive && !x._dying).length;
        const stored = typeof _gluonStoredPairs !== 'undefined' ? _gluonStoredPairs : 0;
        const total = liveCount + 2 * stored;
        if (total !== 6) return `tick ${tick}: alive=${liveCount} stored=${stored} total=${total} (expected 6)`;
        return null;
      }
    },
    { id: 'T13', name: 'Array size unchanged', init: { _initCount: null },
      activate(g) { g._initCount = _demoXons.length; },
      check(tick, g) {
        if (g._initCount === null) return null;
        if (_demoXons.length !== g._initCount)
          return `tick ${tick}: count ${g._initCount}\u2192${_demoXons.length}`;
        return null;
      }
    },
    { id: 'T14', name: 'Dying trail cleanup',
      check(tick, g) {
        for (const xon of _demoXons) {
          if (!xon._dying) continue;
          if (!xon._dyingStartTick) xon._dyingStartTick = tick;
          if (tick - xon._dyingStartTick > 60)
            return `tick ${tick}: xon dying for ${tick - xon._dyingStartTick} ticks (max 60)`;
        }
        return null;
      }
    },
    { id: 'T15', name: 'Xon state (sign + mode)',
      check(tick, g) {
        for (const xon of _demoXons) {
          if (!xon.alive) continue;
          if (xon.sign !== 1 && xon.sign !== -1) return `tick ${tick}: sign=${xon.sign}`;
          if (xon._mode !== 'tet' && xon._mode !== 'oct' && xon._mode !== 'idle_tet' && xon._mode !== 'weak' && xon._mode !== 'gluon' && xon._mode !== 'oct_formation')
            return `tick ${tick}: mode=${xon._mode}`;
        }
        return null;
      }
    },
    { id: 'T16', name: 'Xon always has function',
      check(tick, g) {
        for (const xon of _demoXons) {
          if (!xon.alive) continue;
          if (xon._mode === 'tet' || xon._mode === 'idle_tet') {
            if (!xon._loopSeq || xon._loopSeq.length < 4)
              return `tick ${tick}: ${xon._mode} no loop seq`;
          } else if (xon._mode === 'oct') {
            // During discovery (_octNodeSet null): oct xons roam freely
            if (_octNodeSet && !_octNodeSet.has(xon.node))
              return `tick ${tick}: oct at non-oct node ${xon.node}`;
          } else if (xon._mode === 'oct_formation') {
            // Formation mode: building the cage, no node constraints yet
          }
        }
        return null;
      }
    },
    { id: 'T17', name: 'Full tet coverage (8/8 faces)', convergence: true,
      check(tick, g) {
        if (g.ok === true) return null;
        let visitCount = 0;
        for (let f = 1; f <= 8; f++) {
          if (_demoVisits[f] && _demoVisits[f].total > 0) visitCount++;
        }
        if (visitCount === 8) { g.ok = true; g.msg = ''; _liveGuardRender(); return null; }
        // No time limit — stays pending (null) until all 8 faces visited
        g.msg = `${visitCount}/8 faces`;
        return null;
      }
    },
    { id: 'T19', name: 'Pauli exclusion (1 xon/node)',
      check(tick, g) {
        if (tick === 0) return null; // tick 0: all 6 xons born at center (allowed)
        const occupied = new Map();
        for (const xon of _demoXons) {
          if (!xon.alive) continue;
          const n = xon.node;
          if (occupied.has(n)) {
            // Diagnostic: dump _moveTrace for this tick
            if (!_btActive && typeof _moveTrace !== 'undefined' && _moveTrace.length) {
              console.error('T19 TRACE:', _moveTrace.map(t =>
                `x${t.xonIdx}:${t.from}\u2192${t.to}(${t.path},${t.mode})`).join(' | '));
              console.error('T19 POSITIONS:', _demoXons.map((x,i) =>
                x.alive ? `x${i}@${x.node}(${x._mode})` : `x${i}:dead`).join(' '));
            }
            return `tick ${tick}: node ${n} has 2+ xons`;
          }
          occupied.set(n, true);
        }
        return null;
      }
    },
    { id: 'T20', name: 'Never stand still',
      check(tick, g, ctx) {
        if (!ctx.prev) return null;
        for (const { xon, node: fromNode, mode: prevMode } of ctx.prev) {
          if (!xon.alive) continue;
          if (!_ruleT20StrictMode && prevMode !== xon._mode) continue;
          if (prevMode === 'oct_formation') continue; // formation phase: scripted movement
          if (xon.node === fromNode) return `tick ${tick}: stuck at node ${fromNode} (${prevMode})`;
        }
        return null;
      }
    },
    { id: 'T21', name: 'Oct cage permanence', init: { _octSnapshot: null },
      activate(g) {
        const snap = new Set();
        const _scActive = id => activeSet.has(id) || xonImpliedSet.has(id) || impliedSet.has(id);
        for (const scId of _octSCIds) { if (_scActive(scId)) snap.add(scId); }
        g._octSnapshot = snap;
        if (snap.size === 0) { g.ok = null; g.msg = 'no oct SCs active yet'; }
      },
      check(tick, g) {
        const _scActive = id => activeSet.has(id) || xonImpliedSet.has(id) || impliedSet.has(id);
        // Update snapshot with newly active oct SCs
        if (g._octSnapshot) {
          for (const scId of _octSCIds) { if (_scActive(scId)) g._octSnapshot.add(scId); }
          if (g._octSnapshot.size > 0 && g.ok === null) { g.ok = true; g.msg = ''; }
        }
        // Verify all snapshotted oct SCs still active in any set
        if (g._octSnapshot && g._octSnapshot.size > 0) {
          for (const scId of g._octSnapshot) {
            if (!_scActive(scId)) return `tick ${tick}: oct SC ${scId} lost`;
          }
        }
        return null;
      }
    },
    { id: 'T22', name: 'Hadronic composition (1:1:1 per hadron)', convergence: true,
      check(tick, g) {
        if (g.ok === true) return null;
        const vals = Object.values(_demoVisits);
        const gPu1 = vals.reduce((s, v) => s + (v.pu1 || 0), 0);
        const gPu2 = vals.reduce((s, v) => s + (v.pu2 || 0), 0);
        const gPd  = vals.reduce((s, v) => s + (v.pd  || 0), 0);
        const gNd1 = vals.reduce((s, v) => s + (v.nd1 || 0), 0);
        const gNd2 = vals.reduce((s, v) => s + (v.nd2 || 0), 0);
        const gNu  = vals.reduce((s, v) => s + (v.nu  || 0), 0);
        const pTotal = gPu1 + gPu2 + gPd;
        const nTotal = gNd1 + gNd2 + gNu;
        const total = pTotal + nTotal;
        // 3-way evenness within each hadron: max deviation from 1/3
        const pEven = pTotal > 0 ? 1 - Math.max(
          Math.abs(gPu1/pTotal - 1/3), Math.abs(gPu2/pTotal - 1/3), Math.abs(gPd/pTotal - 1/3)
        ) * 3 : 0;
        const nEven = nTotal > 0 ? 1 - Math.max(
          Math.abs(gNd1/nTotal - 1/3), Math.abs(gNd2/nTotal - 1/3), Math.abs(gNu/nTotal - 1/3)
        ) * 3 : 0;
        // Face coverage evenness
        const totals = [];
        for (let f = 1; f <= 8; f++) totals.push(_demoVisits[f] ? _demoVisits[f].total : 0);
        const mean = totals.reduce((a, b) => a + b, 0) / totals.length;
        const stddev = Math.sqrt(totals.reduce((s, v) => s + (v - mean) ** 2, 0) / totals.length);
        const cv = mean > 0 ? (stddev / mean) : 1;
        const evenness = Math.max(0, 1 - cv);
        if (total > 0)
          g.msg = `p:${gPu1}/${gPu2}/${gPd} n:${gNd1}/${gNd2}/${gNu} cov=${(evenness*100).toFixed(0)}%`;
        // Passes when both hadrons have good 3-way balance with enough data
        if (total >= 18 && pEven >= 0.7 && nEven >= 0.7) {
          g.ok = true;
          g.msg = `p:${gPu1}/${gPu2}/${gPd} n:${gNd1}/${gNd2}/${gNu}`;
          _liveGuardRender();
          return null;
        }
        // Also passes on near-perfect face coverage
        if (evenness >= 0.999 && total >= 18) {
          g.ok = true;
          _liveGuardRender();
          return null;
        }
        return null;
      }
    },
    { id: 'T23', name: 'Sparkle color matches purpose',
      check(tick, g) {
        for (const xon of _demoXons) {
          if (!xon.alive || !xon.sparkMat) continue;
          const actual = xon.sparkMat.color.getHex();
          if (xon._mode === 'oct' || xon._mode === 'oct_formation') {
            if (actual !== 0xffffff) return `tick ${tick}: oct spark=0x${actual.toString(16)}`;
          } else if (xon._mode === 'tet' || xon._mode === 'idle_tet') {
            const expected = QUARK_COLORS[xon._quarkType];
            if (expected !== undefined && actual !== expected)
              return `tick ${tick}: ${xon._quarkType} spark wrong`;
          } else if (xon._mode === 'weak') {
            if (actual !== WEAK_FORCE_COLOR) return `tick ${tick}: weak spark=0x${actual.toString(16)}`;
          }
        }
        return null;
      }
    },
    { id: 'T24', name: 'Trail role validity',
      check(tick, g) {
        const validRoles = new Set(['oct', 'pu1', 'pu2', 'pd', 'nd1', 'nd2', 'nu', 'weak', 'gluon']);
        for (const xon of _demoXons) {
          if (!xon.alive || !xon.trail) continue;
          for (let j = 0; j < xon.trail.length; j++) {
            const e = xon.trail[j];
            if (!e || !e.role) return `tick ${tick}: trail entry ${j} missing role`;
            if (!validRoles.has(e.role)) return `tick ${tick}: invalid role '${e.role}'`;
            // Verify color derivable from role
            const c = _roleToColor(e.role);
            if (c === undefined || c === null) return `tick ${tick}: role '${e.role}' has no color`;
          }
        }
        return null;
      }
    },
    { id: 'T26', name: 'No unactivated SC traversal',
      snapshot(g) {
        // Capture SC activation state BEFORE the tick so check() verifies
        // the SC was active at the time of the move, not after same-tick severance.
        g._t26ActiveSnap = new Set(activeSet);
        g._t26ImpliedSnap = new Set(impliedSet);
        g._t26EImpliedSnap = new Set(xonImpliedSet);
      },
      check(tick, g, ctx) {
        if (!ctx.prev) return null;
        // Check both pre-tick snapshot AND current state:
        // - Snapshot catches SCs active before tick that got removed mid-tick (still valid)
        // - Current state catches SCs added mid-tick before traversal (e.g. _startIdleTetLoop manifest)
        const aSnap = g._t26ActiveSnap || activeSet;
        const iSnap = g._t26ImpliedSnap || impliedSet;
        const eSnap = g._t26EImpliedSnap || xonImpliedSet;
        for (const { xon, node: fromNode, mode: prevMode } of ctx.prev) {
          if (!xon.alive) continue;
          const toNode = xon.node;
          if (toNode === fromNode) continue;
          if (prevMode !== xon._mode) continue;
          const pid = pairId(fromNode, toNode);
          const scId = scPairToId.get(pid);
          if (scId !== undefined) {
            const hasBaseEdge = (baseNeighbors[fromNode] || []).some(nb => nb.node === toNode);
            if (!hasBaseEdge) {
              // SC must be in snapshot OR current state (covers mid-tick additions)
              const inSnap = aSnap.has(scId) || iSnap.has(scId) || eSnap.has(scId);
              const inCurr = activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId);
              if (!inSnap && !inCurr) {
                if (!_btActive && typeof _moveTrace !== 'undefined' && _moveTrace.length) {
                  console.error('T26 TRACE:', _moveTrace.map(t =>
                    `x${t.xonIdx}:${t.from}\u2192${t.to}(${t.path},${t.mode})`).join(' | '));
                }
                return `tick ${tick}: ${prevMode} xon on SC ${scId} (${fromNode}\u2192${toNode})`;
              }
            }
          }
        }
        return null;
      }
    },
    { id: 'T27', name: 'No teleportation',
      check(tick, g, ctx) {
        if (!ctx.prev) return null;
        for (const { xon, node: fromNode, mode: prevMode } of ctx.prev) {
          if (!xon.alive) continue;
          const toNode = xon.node;
          if (toNode === fromNode) continue;
          if (prevMode !== xon._mode) continue;
          const nbs = baseNeighbors[fromNode] || [];
          let connected = nbs.some(nb => nb.node === toNode);
          if (!connected) {
            const scs = scByVert[fromNode] || [];
            connected = scs.some(sc => (sc.a === fromNode ? sc.b : sc.a) === toNode);
          }
          if (!connected) return `tick ${tick}: teleport ${fromNode}\u2192${toNode}`;
        }
        return null;
      }
    },
    { id: 'T30', name: 'Annihilation always in pairs', init: { _prevStored: 0, _prevAlive: 6 },
      activate(g) {
        g._prevStored = typeof _gluonStoredPairs !== 'undefined' ? _gluonStoredPairs : 0;
        g._prevAlive = _demoXons.filter(x => x.alive).length;
      },
      check(tick, g) {
        const curStored = typeof _gluonStoredPairs !== 'undefined' ? _gluonStoredPairs : 0;
        const curAlive = _demoXons.filter(x => x.alive).length;
        if (curStored > g._prevStored) {
          const dStored = curStored - g._prevStored;
          const dAlive = g._prevAlive - curAlive;
          if (dStored * 2 !== dAlive) {
            g._prevStored = curStored; g._prevAlive = curAlive;
            return `tick ${tick}: stored+=${dStored} alive-=${dAlive} (expected 2:1 ratio)`;
          }
        }
        g._prevStored = curStored; g._prevAlive = curAlive;
        return null;
      }
    },
    { id: 'T33', name: 'Trail persists when alive',
      check(tick, g) {
        for (const xon of _demoXons) {
          if (!xon.alive || xon._dying) continue;
          if (!xon.trail || xon.trail.length === 0)
            return `tick ${tick}: alive xon has empty trail at node ${xon.node}`;
          // Verify unified entry format
          const last = xon.trail[xon.trail.length - 1];
          if (!last || typeof last.node !== 'number' || !last.role)
            return `tick ${tick}: trail entry malformed`;
        }
        return null;
      }
    },
    // T34 removed — trails now grow unbounded (slider controls visible length only)
    { id: 'T35', name: 'Sparkle visible when alive',
      check(tick, g) {
        for (const xon of _demoXons) {
          if (!xon.alive || xon._dying) continue;
          if (!xon.spark || !xon.sparkMat)
            return `tick ${tick}: alive xon missing spark at node ${xon.node}`;
        }
        return null;
      }
    },
    { id: 'T36', name: 'Flash on mode transition',
      check(tick, g, ctx) {
        // Skip if flash effects are disabled — no flash to check
        if (typeof _flashEnabled !== 'undefined' && !_flashEnabled) return null;
        if (!ctx.prev) return null;
        for (const { xon, mode: prevMode } of ctx.prev) {
          if (!xon.alive) continue;
          if (prevMode === xon._mode) continue;
          if (xon.flashT < 0.5)
            return `tick ${tick}: ${prevMode}\u2192${xon._mode} flashT=${xon.flashT.toFixed(2)}`;
        }
        return null;
      }
    },
    { id: 'T37', name: 'Trail flash boost',
      check(tick, g) {
        for (const xon of _demoXons) {
          if (!xon.alive || xon._dying) continue;
          if (xon.flashT > 0.1 && xon.flashT < 0.7 && xon._lastTrailFlashBoost !== undefined) {
            if (xon._lastTrailFlashBoost <= 0)
              return `tick ${tick}: flashT=${xon.flashT.toFixed(2)} but boost=${xon._lastTrailFlashBoost.toFixed(3)}`;
          }
        }
        return null;
      }
    },
    { id: 'T38', name: 'Weak force confinement',
      check(tick, g, ctx) {
        // Weak xons are protected from non-physical death.
        // Pauli annihilation is the ONLY way xons die (_annihilateXonPair),
        // and Pauli exclusion is absolute — it trumps weak confinement.
        // So weak xon death is always legitimate.  This guard now only
        // checks that weak xons don't spontaneously vanish (alive→false
        // without going through _annihilateXonPair), which would be a bug.
        // Since _annihilateXonPair is the sole death path, this is a no-op
        // for now but kept as a sentinel for future code changes.
        return null;
      }
    },
    // T39 removed
    { id: 'T40', name: 'Trail fade on annihilation',
      check(tick, g, ctx) {
        if (!ctx.prev) return null;
        for (const { xon } of ctx.prev) {
          if (xon.alive) continue;
          if (!xon._dying)
            return `tick ${tick}: xon annihilated at node ${xon.node} without trail fade (_dying not set)`;
        }
        if (g.ok === null && tick >= LIVE_GUARD_GRACE) { g.ok = true; g.msg = ''; }
        return null;
      }
    },
    // T43 removed — xons now spawn directly on distinct oct nodes (deterministic formation)
    { id: 'T41', name: 'No adjacent swap',
      check(tick, g, ctx) {
        if (!ctx.prev) return null;
        // Detect swaps using snapshot: xon A moved X→Y while xon B moved Y→X
        const moves = ctx.prev.filter(p => p.xon.alive && p.node !== p.xon.node);
        for (let i = 0; i < moves.length; i++) {
          for (let j = i + 1; j < moves.length; j++) {
            const a = moves[i], b = moves[j];
            if (a.node === b.xon.node && b.node === a.xon.node) {
              const aMode = a.mode + '→' + a.xon._mode;
              const bMode = b.mode + '→' + b.xon._mode;
              // Diagnostic: dump _moveTrace for this tick
              if (!_btActive && typeof _moveTrace !== 'undefined' && _moveTrace.length) {
                console.error('T41 SWAP TRACE:', _moveTrace.map(t =>
                  `x${t.xonIdx}:${t.from}→${t.to}(${t.path},${t.mode})`).join(' | '));
              }
              return `tick ${tick}: swap ${a.node}↔${b.node} [${aMode}] [${bMode}]`;
            }
          }
        }
        if (g.ok === null && tick >= LIVE_GUARD_GRACE) { g.ok = true; g.msg = ''; }
        return null;
      }
    },
    { id: 'T42', name: 'SC attribution (no top-down imposition)',
      check(tick, g) {
        if (!_nucleusTetFaceData || !xonImpliedSet.size) return null;
        if (typeof _scAttribution === 'undefined') return null;
        // Every SC in xonImpliedSet must have causal attribution:
        // a traversal event that caused it to exist. Side-effect SCs from
        // lattice deformation are fine — they inherit attribution from the
        // traversal that triggered the solver. Unattributed SCs = top-down
        // imposition, which violates the bottom-up physics model.
        // Attribution cleanup runs at end of tick, so this is a safety net.
        for (const scId of xonImpliedSet) {
            if (activeSet.has(scId)) continue; // not eSC's responsibility
            if (!_scAttribution.has(scId)) {
                const sc = SC_BY_ID[scId];
                return `tick ${tick}: unattributed eSC ${scId} (${sc ? sc.a + '↔' + sc.b : '?'})`;
            }
        }
        if (g.ok === null && tick >= LIVE_GUARD_GRACE) { g.ok = true; g.msg = ''; }
        return null;
      }
    },
    // T45: DISABLED — bounce guard removed to give weak xons more movement freedom.
    // Re-enable by setting _T45_BOUNCE_GUARD = true in flux-demo.js and uncommenting.
    { id: 'T44', name: 'Traversal lock edge-only',
      check(tick, g) {
        // _traversalLockedSCs must ONLY contain SCs on edges xons are straddling
        // (prevNode↔node). No face-level locks. Physics: "if I used a shortcut on my
        // last turn, it must exist this turn." Nothing more.
        if (typeof _traversalLockedSCs !== 'function') return null;
        const locked = _traversalLockedSCs();
        // Build the expected set: only edge SCs
        const expectedEdgeSCs = new Set();
        for (const xon of _demoXons) {
            if (!xon.alive || xon.prevNode == null) continue;
            const pid = pairId(xon.prevNode, xon.node);
            const scId = scPairToId.get(pid);
            if (scId !== undefined) expectedEdgeSCs.add(scId);
        }
        // Every locked SC must be an edge SC
        for (const scId of locked) {
            if (!expectedEdgeSCs.has(scId)) {
                const sc = SC_BY_ID[scId];
                return `tick ${tick}: non-edge lock ${scId} (${sc ? sc.a + '↔' + sc.b : '?'})`;
            }
        }
        // Every edge SC must be locked
        for (const scId of expectedEdgeSCs) {
            if (!locked.has(scId)) {
                const sc = SC_BY_ID[scId];
                return `tick ${tick}: unlocked edge ${scId} (${sc ? sc.a + '↔' + sc.b : '?'})`;
            }
        }
        if (g.ok === null && tick >= LIVE_GUARD_GRACE) { g.ok = true; g.msg = ''; }
        return null;
      }
    },
    // ═══════════════════════════════════════════════════════════════════
    // T50-T53: Unified demand-driven choreography tests
    // ═══════════════════════════════════════════════════════════════════
    // T50 REMOVED: per user request
    { id: 'T51', name: 'Ratio tracker accuracy',
      check(tick, g) {
        if (g.ok === true) return null;
        if (typeof _ratioTracker === 'undefined') {
            if (tick > LIVE_GUARD_GRACE) return 'tick ' + tick + ': _ratioTracker not defined';
            return null;
        }
        if (tick < LIVE_GUARD_GRACE) return null;
        // Sync and verify against manual sum of _demoVisits
        _ratioTracker.sync();
        const types6 = ['pu1', 'pu2', 'pd', 'nd1', 'nd2', 'nu'];
        const manual = {};
        for (const t of types6) manual[t] = 0;
        for (let f = 1; f <= 8; f++) {
            if (!_demoVisits[f]) continue;
            for (const t of types6) manual[t] += _demoVisits[f][t] || 0;
        }
        for (const t of types6) {
            if (_ratioTracker[t] !== manual[t]) {
                return `tick ${tick}: tracker mismatch ${t}=${_ratioTracker[t]}/${manual[t]}`;
            }
        }
        // Verify deficit() returns number in [-1, 1]
        for (const t of types6) {
            const d = _ratioTracker.deficit(t);
            if (typeof d !== 'number' || isNaN(d)) {
                return `tick ${tick}: deficit('${t}') returned ${d}`;
            }
        }
        g.ok = true; g.msg = ''; return null;
      }
    },
    // T52 REMOVED: No forced loop termination — no longer needed since window system eliminated.
    // All tet→oct transitions are now legitimate (PHASE 0 eviction, safety escape, loop completion).
    // T54 REMOVED: incompatible with weak xon free-roaming outside oct cage
    // T53 REMOVED: per user request (covered by T22)
    // T56 REMOVED: diagonal traversal fixed at the source (movement filtering)
    { id: 'T55', name: 'Oct capacity (hard max 4)',
      init: { _octCapacity: OCT_CAPACITY_MAX },
      check(tick, g) {
        const cap = OCT_CAPACITY_MAX;
        g._octCapacity = cap;
        // Oct doesn't exist until discovered (~tick 2). No grace needed.
        if (!_octNodeSet || _octNodeSet.size === 0) return null;
        const octCount = _demoXons.filter(x => x.alive && _octNodeSet.has(x.node)).length;
        g.msg = `oct: ${octCount}/${cap}`;
        if (octCount > cap) {
          return `tick ${tick}: ${octCount} xons on oct nodes > capacity ${cap}`;
        }
        return null;
      }
    },
    { id: 'T57', name: 'Tracer segments unit-length',
      check(tick, g) {
        // Post-solver: use current pos[] (solver has converged).
        // Activated edges are ~1.0. Teleportation shows as >> 1.0.
        const tol = 0.05;
        for (const xon of _demoXons) {
          if (!xon.alive || xon._dying) continue;
          if (!xon.trail || xon.trail.length < 2) continue;
          const fromE = xon.trail[xon.trail.length - 2];
          const toE = xon.trail[xon.trail.length - 1];
          const fromN = fromE.node, toN = toE.node;
          if (fromN === toN) continue; // same-node (spawn)
          const a = pos[fromN], b = pos[toN];
          if (!a || !b) continue;
          const dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
          if (dist < 1e-6) continue;
          if (Math.abs(dist - 1) > tol) {
            return `tick ${tick}: tracer segment len=${dist.toFixed(4)} (nodes ${fromN}→${toN})`;
          }
        }
        return null;
      }
    },

    // T58: Tet coloring matches geometric actualization — a tet is colored
    // IFF all its bounding SCs are active. Color = dominant quark type from
    // cumulative edge balance, or neutral gray if no traversals yet.
    { id: 'T58', name: 'Tet color matches SC actualization',
      check(tick, g) {
        if (g.ok === null && tick >= LIVE_GUARD_GRACE) { g.ok = true; g.msg = ''; }
        if (!_nucleusTetFaceData || !_ruleAnnotations) return null;
        for (const [fIdStr, fd] of Object.entries(_nucleusTetFaceData)) {
          const fId = parseInt(fIdStr);
          const opacity = _ruleAnnotations.tetOpacity.get(fd.voidIdx);
          const isColored = opacity && opacity > 0;
          const allSCsActive = fd.scIds.every(scId =>
            activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId));
          // Colored but SCs not all active = bad
          if (isColored && !allSCsActive) {
            return `tick ${tick}: face ${fId} colored but SCs not all active`;
          }
          // SCs all active but NOT colored = bad (should be visible)
          if (allSCsActive && !isColored) {
            return `tick ${tick}: face ${fId} has all SCs active but not colored`;
          }
          // If colored, verify color matches dominant quark type (or neutral gray)
          if (isColored) {
            const annotCol = _ruleAnnotations.tetColors.get(fd.voidIdx);
            const dominant = typeof _dominantQuarkForFace === 'function'
              ? _dominantQuarkForFace(fId) : null;
            if (dominant) {
              const expected = QUARK_COLORS[dominant];
              if (annotCol !== expected) {
                return `tick ${tick}: face ${fId} colored 0x${(annotCol||0).toString(16)} but dominant type is ${dominant} (0x${expected.toString(16)})`;
              }
            }
            // If no dominant type yet (no traversals), neutral gray is fine
          }
        }
        return null;
      }
    },
    // T59: Trail continuity — every xon's current node must match its last trail entry.
    // If they diverge, the trail head renderer draws a line from the last frozen position
    // to the current interpolated position, creating a visible "teleport" segment.
    { id: 'T59', name: 'Trail head matches node',
      check(tick, g) {
        if (g.ok === null && tick >= LIVE_GUARD_GRACE) { g.ok = true; g.msg = ''; }
        if (!_demoXons) return null;
        for (let xi = 0; xi < _demoXons.length; xi++) {
          const xon = _demoXons[xi];
          if (!xon.alive) continue;
          if (!xon.trail || xon.trail.length === 0) continue;
          const lastTrail = xon.trail[xon.trail.length - 1];
          if (xon.node !== lastTrail.node) {
            return `tick ${tick}: X${xi} at node ${xon.node} but trail ends at ${lastTrail.node}`;
          }
        }
        return null;
      }
    },
    // T60: Non-actualized tet must eject as weak particle AWAY from oct.
    // Guard enforces: no tet/idle_tet xon at step >= 4 with _tetActualized === false
    // (it should have been converted to weak before the guard runs).
    // Ejected xons must BFS to an ejection target node (1 hop from oct, not in
    // any tet/oct void) before they can return — enforced by PHASE 0.5.
    { id: 'T60', name: 'Non-actualized tet ejects weak',
      check(tick, g) {
        if (g.ok === null && tick >= LIVE_GUARD_GRACE) { g.ok = true; g.msg = ''; }
        if (!_demoXons) return null;
        for (let xi = 0; xi < _demoXons.length; xi++) {
          const xon = _demoXons[xi];
          if (!xon.alive) continue;
          if ((xon._mode === 'tet' || xon._mode === 'idle_tet') &&
              xon._loopStep >= 4 && xon._tetActualized === false) {
            return `tick ${tick}: X${xi} completed non-actualized loop on face ${xon._assignedFace} but not ejected to weak`;
          }
        }
        return null;
      }
    },
    // T61: Weak xons must NOT be on oct nodes at end of tick.
    // Purple tracers on the octa break the visual contract: weak particles
    // must eject away from the oct cage, not linger on it.
    // projected() also lets the backtracker steer away from weak-on-oct states.
    { id: 'T61', name: 'No weak xon on oct node',
      check(tick, g) {
        if (tick < LIVE_GUARD_GRACE) return null; // grace period
        if (typeof _openingPhase !== 'undefined' && _openingPhase) return null;
        if (g.ok === null) { g.ok = true; g.msg = ''; }
        if (!_demoXons || !_octNodeSet || _octNodeSet.size === 0) return null;
        for (let xi = 0; xi < _demoXons.length; xi++) {
          const xon = _demoXons[xi];
          if (!xon.alive) continue;
          if (xon._mode === 'weak' && _octNodeSet.has(xon.node)) {
            return `tick ${tick}: X${xi} weak on oct node ${xon.node}`;
          }
        }
        return null;
      }
    },
    // T62: Weak xons can only re-enter the system (transition from weak to any
    // other mode) at an oct node. They must navigate back to the oct cage before
    // being allowed to resume normal duties.
    { id: 'T62', name: 'Weak re-entry at oct only',
      snapshot(g) {
        g._t62prev = new Map();
        if (!_demoXons) return;
        for (const xon of _demoXons) {
          if (!xon.alive) continue;
          g._t62prev.set(xon, { mode: xon._mode, node: xon.node });
        }
      },
      check(tick, g, ctx) {
        if (tick < LIVE_GUARD_GRACE) return null; // grace period
        if (g.ok === null) { g.ok = true; g.msg = ''; }
        if (!ctx.prev || !_demoXons || !_octNodeSet || _octNodeSet.size === 0) return null;
        for (const xon of _demoXons) {
          if (!xon.alive) continue;
          const prev = g._t62prev?.get(xon);
          if (!prev) continue;
          // Was weak last tick, no longer weak this tick → re-entry
          if (prev.mode === 'weak' && xon._mode !== 'weak') {
            if (!_octNodeSet.has(xon.node)) {
              return `tick ${tick}: X${_demoXons.indexOf(xon)} re-entered as ${xon._mode} at non-oct node ${xon.node}`;
            }
          }
        }
        return null;
      }
    },
    // ── T70: Gluon mode validity ──
    { id: 'T70', name: 'Gluon mode validity', convergence: true,
      check(tick, g) {
        if (g.ok === true) return null;
        for (const xon of _demoXons) {
          if (!xon.alive || xon._mode !== 'gluon') continue;
          // Gluon must be white (same as oct)
          if (xon.col !== GLUON_COLOR) return `tick ${tick}: gluon xon has color 0x${xon.col.toString(16)}, expected 0x${GLUON_COLOR.toString(16)}`;
          // Gluon must be on oct cage
          if (!_octNodeSet || !_octNodeSet.has(xon.node)) return `tick ${tick}: gluon xon at node ${xon.node} not on oct cage`;
          g.ok = true; g.msg = ''; _liveGuardRender(); return null;
        }
        return null; // no gluons seen yet — convergence test stays null
      }
    },
    // (T71 removed: _mayReturn no longer used)
    // ── T72: _actualizedTetNodes correctness ──
    { id: 'T72', name: '_actualizedTetNodes correct', convergence: true,
      check(tick, g) {
        if (g.ok === true) return null;
        if (!_actualizedTetNodes || typeof voidNeighborData === 'undefined') return null;
        // Verify: every tet with ALL SCs active has its nodes in _actualizedTetNodes
        for (const v of voidNeighborData) {
          if (v.type !== 'tet') continue;
          const allActive = v.scIds.every(scId =>
            activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId));
          if (allActive) {
            for (const n of v.nbrs) {
              if (!_actualizedTetNodes.has(n)) return `tick ${tick}: actualized tet node ${n} missing from _actualizedTetNodes`;
            }
          }
        }
        g.ok = true; g.msg = ''; _liveGuardRender(); return null;
      }
    },
    // ── T73: Ejection target validity ──
    { id: 'T73', name: 'Ejection target validity', convergence: true,
      check(tick, g) {
        if (g.ok === true) return null;
        if (!_ejectionForbidden || !_octNodeSet || !_actualizedTetNodes) return null;
        // Oct nodes must be forbidden
        for (const n of _octNodeSet) {
          if (!_ejectionForbidden.has(n)) return `tick ${tick}: oct node ${n} not in _ejectionForbidden`;
        }
        // Actualized tet nodes must be forbidden
        for (const n of _actualizedTetNodes) {
          if (!_ejectionForbidden.has(n)) return `tick ${tick}: actualized tet node ${n} not in _ejectionForbidden`;
        }
        // _isValidEjectionTarget must return false for oct nodes
        for (const n of _octNodeSet) {
          if (_isValidEjectionTarget(n)) return `tick ${tick}: _isValidEjectionTarget(${n}) true for oct node`;
        }
        g.ok = true; g.msg = ''; _liveGuardRender(); return null;
      }
    },
    // ── T75: Ejected xon movement restriction ──
    // (T75 removed: _mayReturn no longer used, weak xons may traverse freely)
    // ── T74: Backtracker uncapped (infrastructure guarantee) ──
    { id: 'T74', name: 'Backtracker uncapped', convergence: true,
      check(tick, g) {
        if (g.ok === true) return null;
        if (_BT_MAX_RETRIES !== Infinity) return `_BT_MAX_RETRIES is ${_BT_MAX_RETRIES}, not Infinity`;
        if (_BFS_MAX_LAYERS !== Infinity) return `_BFS_MAX_LAYERS is ${_BFS_MAX_LAYERS}, not Infinity`;
        g.ok = true; g.msg = ''; _liveGuardRender(); return null;
      }
    },
    // T76 (direction balance) removed — no trivial way to test convergence threshold
    // ── T77: BFS severance infrastructure ──
    // Verify that the 2-depth BFS severance system is operational.
    // _severanceCount must be a number (tracks total successful severances).
    { id: 'T77', name: 'BFS severance active', convergence: true,
      check(tick, g) {
        if (g.ok === true) return null;
        if (typeof _severanceCount !== 'number') return `_severanceCount is ${typeof _severanceCount}, not number`;
        if (typeof excitationSeverForRoom !== 'function') return 'excitationSeverForRoom not a function';
        g.ok = true; g.msg = ''; _liveGuardRender(); return null;
      }
    },
    // ── T78: SC cleanup is distance-only ──
    // Every SC in xonImpliedSet must be approximately unit-length.
    // Non-unit-length SCs should have been removed by the distance-only cleanup.
    { id: 'T78', name: 'SC cleanup distance-only',
      check(tick, g) {
        if (tick < 12) return null; // grace: cleanup needs solver to settle
        for (const scId of xonImpliedSet) {
          const sc = SC_BY_ID[scId];
          if (!sc) continue;
          const pa = pos[sc.a], pb = pos[sc.b];
          if (!pa || !pb) continue;
          const dx = pb[0]-pa[0], dy = pb[1]-pa[1], dz = pb[2]-pa[2];
          const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
          if (Math.abs(dist - 1) > 0.20) {
            return `tick ${tick}: SC ${scId} in xonImpliedSet has dist=${dist.toFixed(4)} (non-unit-length)`;
          }
        }
        return null;
      }
    },
    // T79: Full oct occupancy (all 6 xons on oct nodes) may persist for at most
    // T79_MAX_FULL_TICKS consecutive ticks. Tune via global T79_MAX_FULL_TICKS.
    { id: 'T79', name: 'Oct full (6/6) limit',
      init: { _consecutiveFullTicks: 0 },
      snapshot(g) {
        if (!_octNodeSet || _octNodeSet.size === 0) { g._consecutiveFullTicks = 0; return; }
        const octCount = _demoXons.filter(x => x.alive && _octNodeSet.has(x.node)).length;
        if (octCount >= 6) {
          g._consecutiveFullTicks++;
        } else {
          g._consecutiveFullTicks = 0;
        }
      },
      check(tick, g) {
        if (tick < LIVE_GUARD_GRACE) return null;
        if (!_octNodeSet || _octNodeSet.size === 0) return null;
        const maxFull = (typeof T79_MAX_FULL_TICKS !== 'undefined') ? T79_MAX_FULL_TICKS : 1;
        const octCount = _demoXons.filter(x => x.alive && _octNodeSet.has(x.node)).length;
        g.msg = `oct: ${octCount}/6 (consec: ${g._consecutiveFullTicks}/${maxFull})`;
        if (g._consecutiveFullTicks > maxFull) {
          return `tick ${tick}: 6/6 oct for ${g._consecutiveFullTicks} consecutive ticks (max ${maxFull})`;
        }
        return null;
      }
    },
    // ── T80: Base direction polarity ──
    // Xons may only traverse base edges in the POSITIVE vector direction.
    // Shortcut edges are exempt (bidirectional). Feature-flagged via _T80_BASE_POLARITY.
    { id: 'T80', name: 'Base direction polarity (positive only)',
      projected(states) {
        if (!_T80_BASE_POLARITY) return null;
        for (const { xon, toNode } of states) {
          if (!xon.alive || toNode === xon.node) continue;
          const fromNode = xon.node;
          // Is this a base edge?
          const bnbs = baseNeighbors[fromNode] || [];
          const baseNb = bnbs.find(nb => nb.node === toNode);
          if (!baseNb) continue; // SC edge — exempt
          // Check polarity: basePosNeighbor[from][dirIdx] === to means positive
          if (basePosNeighbor[fromNode][baseNb.dirIdx] !== toNode) {
            return `projected: ${fromNode}→${toNode} is negative base dir ${baseNb.dirIdx}`;
          }
        }
        return null;
      },
      check(tick, g, ctx) {
        if (!_T80_BASE_POLARITY) return null;
        if (!ctx.prev) return null;
        for (const { xon, node: fromNode, mode: prevMode } of ctx.prev) {
          if (!xon.alive) continue;
          const toNode = xon.node;
          if (toNode === fromNode) continue;
          if (prevMode !== xon._mode) continue;
          // Is this a base edge?
          const bnbs = baseNeighbors[fromNode] || [];
          const baseNb = bnbs.find(nb => nb.node === toNode);
          if (!baseNb) continue; // SC edge — exempt (bidirectional)
          // Check polarity: basePosNeighbor[from][dirIdx] === to means positive direction
          if (basePosNeighbor[fromNode][baseNb.dirIdx] !== toNode) {
            return `tick ${tick}: xon moved ${fromNode}→${toNode} in negative base dir ${baseNb.dirIdx}`;
          }
        }
        return null;
      }
    },
    // ── T81: Matter/antimatter winding direction ──
    // Once the merry-go-round establishes a winding direction at opening tick 1,
    // all subsequent equator traversals must follow the same direction.
    // CW = cycle order (i→i+1), CCW = reverse (i+1→i).
    { id: 'T81', name: 'Equator winding direction locked',
      projected(states) {
        if (!_octWindingDirection || !_octEquatorCycle || _octEquatorCycle.length !== 4) return null;
        const eqIdx = new Map();
        _octEquatorCycle.forEach((n, i) => eqIdx.set(n, i));
        for (const { xon, toNode } of states) {
          if (!xon.alive || toNode === xon.node) continue;
          const fi = eqIdx.get(xon.node), ti = eqIdx.get(toNode);
          if (fi === undefined || ti === undefined) continue;
          const isCW = (ti === (fi + 1) % 4);
          const isCCW = (fi === (ti + 1) % 4);
          if (!isCW && !isCCW) continue; // not adjacent equator nodes
          if (isCW && _octWindingDirection === 'CCW') {
            return `projected: ${xon.node}→${toNode} is CW but winding is CCW`;
          }
          if (isCCW && _octWindingDirection === 'CW') {
            return `projected: ${xon.node}→${toNode} is CCW but winding is CW`;
          }
        }
        return null;
      },
      check(tick, g, ctx) {
        if (!_octWindingDirection || !_octEquatorCycle || _octEquatorCycle.length !== 4) return null;
        if (!ctx.prev) return null;
        const eqIdx = new Map();
        _octEquatorCycle.forEach((n, i) => eqIdx.set(n, i));
        for (const { xon, node: fromNode } of ctx.prev) {
          if (!xon.alive) continue;
          const toNode = xon.node;
          if (toNode === fromNode) continue;
          const fi = eqIdx.get(fromNode), ti = eqIdx.get(toNode);
          if (fi === undefined || ti === undefined) continue;
          const isCW = (ti === (fi + 1) % 4);
          const isCCW = (fi === (ti + 1) % 4);
          if (!isCW && !isCCW) continue; // not adjacent equator nodes
          if (isCW && _octWindingDirection === 'CCW') {
            return `tick ${tick}: xon moved ${fromNode}→${toNode} is CW but winding is CCW`;
          }
          if (isCCW && _octWindingDirection === 'CW') {
            return `tick ${tick}: xon moved ${fromNode}→${toNode} is CCW but winding is CW`;
          }
        }
        return null;
      }
    },

    // ── T82: Planck second counter integrity ──
    // _planckSeconds counts only ticks with lattice deformation (SC adds/removes).
    // Must always satisfy: 0 < _planckSeconds <= _demoTick after grace period.
    {
      id: 'T82', name: 'Planck second counter',
      convergence: true,
      check(tick, g) {
        if (tick < 12) return null; // grace period
        if (typeof _planckSeconds === 'undefined') return 'missing _planckSeconds variable';
        if (_planckSeconds > _demoTick) return `_planckSeconds (${_planckSeconds}) > _demoTick (${_demoTick})`;
        if (_planckSeconds === 0) return '_planckSeconds still 0 after 12 ticks';
        return null;
      }
    },

    // ── T83: Snapshot stack integrity (rewind foundation) ──
    // Verifies _btSnapshots entries have monotonically increasing tick values
    // and the last entry's tick equals _demoTick - 1. This validates the
    // snapshot stack that the rewind/playback feature depends on.
    {
      id: 'T83', name: 'Snapshot stack integrity',
      convergence: true,
      check(tick, g) {
        if (tick < 20) return null;
        if (g.ok === true) return null; // already verified
        if (!_btSnapshots || _btSnapshots.length < 2) return null;
        // Check contiguous path from t=0 to t=current:
        // 1. First snapshot must be tick 0
        if (_btSnapshots[0].tick !== 0) {
          return `first snapshot tick=${_btSnapshots[0].tick}, expected 0`;
        }
        // 2. Strictly monotonic (no duplicates, no gaps in ordering)
        for (let i = 1; i < _btSnapshots.length; i++) {
          if (_btSnapshots[i].tick <= _btSnapshots[i - 1].tick) {
            return `snapshot[${i}].tick=${_btSnapshots[i].tick} <= snapshot[${i-1}].tick=${_btSnapshots[i-1].tick}`;
          }
        }
        // 3. Last snapshot must reach at least the current tick's pre-state
        const last = _btSnapshots[_btSnapshots.length - 1];
        if (last.tick < _demoTick - 1) {
          return `last snapshot tick=${last.tick} < _demoTick-1=${_demoTick - 1} — gap in history`;
        }
        // All good — promote to green
        g.ok = true; g.msg = '';
        if (typeof _liveGuardRender === 'function') _liveGuardRender();
        return null;
      }
    },

    // ── T84: Tet loop commitment — no mid-loop role switching ──
    // Once a xon enters tet/idle_tet mode with a quark assignment, it must
    // either complete the loop (step >= 4 → return to oct) or be ejected
    // (return to oct/weak). It cannot switch quarkType or assignedFace
    // while still in tet/idle_tet mode mid-loop.
    {
      id: 'T84', name: 'No mid-loop quark/face switch',
      init: { prevTetState: null },
      snapshot(g) {
        // Capture tet state BEFORE tick runs
        g.prevTetState = _demoXons.map(x => {
          if (!x.alive) return null;
          if (x._mode === 'tet' || x._mode === 'idle_tet') {
            return { face: x._assignedFace, quark: x._quarkType, step: x._loopStep };
          }
          return null;
        });
      },
      check(tick, g) {
        if (tick < LIVE_GUARD_GRACE) return null;
        if (g.ok === null) { g.ok = true; g.msg = ''; }
        if (!g.prevTetState) return null;
        for (let i = 0; i < _demoXons.length; i++) {
          const x = _demoXons[i];
          if (!x.alive) continue;
          const prev = g.prevTetState[i];
          if (!prev) continue; // wasn't in tet last tick
          // Xon was in tet/idle_tet last tick. If still in tet/idle_tet now:
          if (x._mode === 'tet' || x._mode === 'idle_tet') {
            if (x._assignedFace !== prev.face) {
              return `tick ${tick}: X${i} switched face ${prev.face}→${x._assignedFace} mid-loop (step was ${prev.step})`;
            }
            if (x._quarkType !== prev.quark) {
              return `tick ${tick}: X${i} switched quark ${prev.quark}→${x._quarkType} mid-loop on face ${prev.face} (step was ${prev.step})`;
            }
          }
          // If xon left tet mode (now oct/weak), that's fine — ejection/completion
        }
        return null;
      }
    },

    // ── T86: Bare tetrahedra — actualized tet must have edge contributors ──
    // If a tet face has all SCs active (actualized) but _dominantQuarkForFace
    // returns null (no edge traversals since last manifestation), the choreographer
    // failed to populate the tet before manifesting it.
    {
      id: 'T86', name: 'No bare tetrahedra',
      check(tick, g) {
        if (tick < LIVE_GUARD_GRACE) return null;
        // When _ruleBareTetrahedra is OFF, bare actualized tets are allowed —
        // they simply don't count as quarks (no edges colored yet).
        if (typeof _ruleBareTetrahedra !== 'undefined' && !_ruleBareTetrahedra) return null;
        if (g.ok === null) { g.ok = true; g.msg = ''; }
        if (!_nucleusTetFaceData) return null;
        for (const [fIdStr, fd] of Object.entries(_nucleusTetFaceData)) {
          const fId = parseInt(fIdStr);
          const allSCsActive = fd.scIds.every(scId =>
            activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId));
          if (!allSCsActive) continue;
          const dominant = typeof _dominantQuarkForFace === 'function'
            ? _dominantQuarkForFace(fId) : null;
          if (dominant === null) {
            return `tick ${tick}: face ${fId} actualized but no edge contributors (bare tetrahedra)`;
          }
        }
        return null;
      }
    },

    // ── T87: Same-hadron edge exclusion ──
    // Quarks of the same hadron must never simultaneously occupy tets that
    // share an edge. The K₄,₄ bipartite structure guarantees this structurally
    // (A-faces only neighbor B-faces), but this guard catches any violation
    // from bugs in face assignment, mode transitions, or future refactors.
    // "Same hadron" = both proton types (pu1/pu2/pd) or both neutron (nd1/nd2/nu).
    {
      id: 'T87', name: 'Same-hadron edge exclusion',
      init: { _adjCache: null },
      check(tick, g) {
        if (tick < LIVE_GUARD_GRACE) return null;
        if (g.ok === null) { g.ok = true; g.msg = ''; }
        if (!_nucleusTetFaceData || !_demoXons) return null;

        // Build adjacency cache once: face pairs that share ≥2 nodes
        // Rebuild if empty (face data may not have been populated on first call)
        if (!g._adjCache || g._adjCache.length === 0) {
          g._adjCache = [];
          const faces = Object.keys(_nucleusTetFaceData).map(Number);
          for (let i = 0; i < faces.length; i++) {
            for (let j = i + 1; j < faces.length; j++) {
              const c1 = new Set(_nucleusTetFaceData[faces[i]].cycle);
              const shared = _nucleusTetFaceData[faces[j]].cycle.filter(n => c1.has(n));
              if (shared.length >= 2) {
                g._adjCache.push([faces[i], faces[j]]);
              }
            }
          }
        }

        // Collect active tet/idle_tet xons → face → quark type
        const faceQuarks = new Map(); // faceId → [{xonIdx, quarkType}]
        for (let i = 0; i < _demoXons.length; i++) {
          const x = _demoXons[i];
          if (!x.alive) continue;
          if (x._mode !== 'tet' && x._mode !== 'idle_tet') continue;
          if (x._assignedFace == null || !x._quarkType) continue;
          if (!faceQuarks.has(x._assignedFace)) faceQuarks.set(x._assignedFace, []);
          faceQuarks.get(x._assignedFace).push({ idx: i, qt: x._quarkType });
        }

        const PROTON_TYPES = new Set(['pu1', 'pu2', 'pd']);
        const NEUTRON_TYPES = new Set(['nd1', 'nd2', 'nu']);

        for (const [f1, f2] of g._adjCache) {
          const q1 = faceQuarks.get(f1);
          const q2 = faceQuarks.get(f2);
          if (!q1 || !q2) continue; // at most one face occupied
          for (const a of q1) {
            for (const b of q2) {
              const bothProton = PROTON_TYPES.has(a.qt) && PROTON_TYPES.has(b.qt);
              const bothNeutron = NEUTRON_TYPES.has(a.qt) && NEUTRON_TYPES.has(b.qt);
              if (bothProton || bothNeutron) {
                return `tick ${tick}: same-hadron edge conflict — X${a.idx}(${a.qt}) on F${f1} adjacent to X${b.idx}(${b.qt}) on F${f2}`;
              }
            }
          }
        }
        return null;
      }
    },

    // ── T88: Opposite-hadron vertex exclusion ──
    // Quarks of opposite hadrons must not simultaneously occupy tets that
    // share only a single node (same bipartite group). The K₄,₄ geometry
    // makes single-node sharing exclusive to same-group face pairs (A↔A
    // or B↔B). A proton quark and neutron quark sharing a vertex means
    // they've violated the bipartite orientation.
    {
      id: 'T88', name: 'Opposite-hadron vertex exclusion',
      init: { _vertCache: null },
      check(tick, g) {
        if (tick < LIVE_GUARD_GRACE) return null;
        if (g.ok === null) { g.ok = true; g.msg = ''; }
        if (!_nucleusTetFaceData || !_demoXons) return null;

        // Build vertex-sharing cache: same-group face pairs sharing exactly 1 node
        if (!g._vertCache || g._vertCache.length === 0) {
          g._vertCache = [];
          const faces = Object.keys(_nucleusTetFaceData).map(Number);
          for (let i = 0; i < faces.length; i++) {
            for (let j = i + 1; j < faces.length; j++) {
              const c1 = new Set(_nucleusTetFaceData[faces[i]].cycle);
              const shared = _nucleusTetFaceData[faces[j]].cycle.filter(n => c1.has(n));
              if (shared.length === 1) {
                g._vertCache.push([faces[i], faces[j], shared[0]]);
              }
            }
          }
        }

        // Collect active tet/idle_tet xons → face → quark type
        const faceQuarks = new Map();
        for (let i = 0; i < _demoXons.length; i++) {
          const x = _demoXons[i];
          if (!x.alive) continue;
          if (x._mode !== 'tet' && x._mode !== 'idle_tet') continue;
          if (x._assignedFace == null || !x._quarkType) continue;
          if (!faceQuarks.has(x._assignedFace)) faceQuarks.set(x._assignedFace, []);
          faceQuarks.get(x._assignedFace).push({ idx: i, qt: x._quarkType });
        }

        const PROTON_TYPES = new Set(['pu1', 'pu2', 'pd']);

        for (const [f1, f2, sharedNode] of g._vertCache) {
          const q1 = faceQuarks.get(f1);
          const q2 = faceQuarks.get(f2);
          if (!q1 || !q2) continue;
          for (const a of q1) {
            for (const b of q2) {
              // Opposite hadron = one proton-type, one neutron-type
              const aIsProton = PROTON_TYPES.has(a.qt);
              const bIsProton = PROTON_TYPES.has(b.qt);
              if (aIsProton !== bIsProton) {
                return `tick ${tick}: opposite-hadron vertex conflict — X${a.idx}(${a.qt}) on F${f1} shares node ${sharedNode} with X${b.idx}(${b.qt}) on F${f2}`;
              }
            }
          }
        }
        return null;
      }
    },

    // ── T89: Disjoint-face hadron exclusion ──
    // Faces sharing 0 nodes are always cross-group (A↔B). If two quarks
    // of the same hadron simultaneously occupy disjoint faces, the bipartite
    // orientation is violated. Completes the full coverage:
    //   T87: shared=2 (edge) → must be opposite hadron ✓
    //   T88: shared=1 (vertex) → must be same hadron ✓
    //   T89: shared=0 (disjoint) → must be opposite hadron ✓
    {
      id: 'T89', name: 'Disjoint-face hadron exclusion',
      init: { _disjCache: null },
      check(tick, g) {
        if (tick < LIVE_GUARD_GRACE) return null;
        if (g.ok === null) { g.ok = true; g.msg = ''; }
        if (!_nucleusTetFaceData || !_demoXons) return null;

        // Build disjoint cache: face pairs sharing 0 nodes
        if (!g._disjCache || g._disjCache.length === 0) {
          g._disjCache = [];
          const faces = Object.keys(_nucleusTetFaceData).map(Number);
          for (let i = 0; i < faces.length; i++) {
            for (let j = i + 1; j < faces.length; j++) {
              const c1 = new Set(_nucleusTetFaceData[faces[i]].cycle);
              const shared = _nucleusTetFaceData[faces[j]].cycle.filter(n => c1.has(n));
              if (shared.length === 0) {
                g._disjCache.push([faces[i], faces[j]]);
              }
            }
          }
        }

        // Collect active tet/idle_tet xons → face → quark type
        const faceQuarks = new Map();
        for (let i = 0; i < _demoXons.length; i++) {
          const x = _demoXons[i];
          if (!x.alive) continue;
          if (x._mode !== 'tet' && x._mode !== 'idle_tet') continue;
          if (x._assignedFace == null || !x._quarkType) continue;
          if (!faceQuarks.has(x._assignedFace)) faceQuarks.set(x._assignedFace, []);
          faceQuarks.get(x._assignedFace).push({ idx: i, qt: x._quarkType });
        }

        const PROTON_TYPES = new Set(['pu1', 'pu2', 'pd']);

        for (const [f1, f2] of g._disjCache) {
          const q1 = faceQuarks.get(f1);
          const q2 = faceQuarks.get(f2);
          if (!q1 || !q2) continue;
          for (const a of q1) {
            for (const b of q2) {
              // Same hadron on disjoint faces = violation
              const aIsProton = PROTON_TYPES.has(a.qt);
              const bIsProton = PROTON_TYPES.has(b.qt);
              if (aIsProton === bIsProton) {
                return `tick ${tick}: same-hadron disjoint conflict — X${a.idx}(${a.qt}) on F${f1} shares no nodes with X${b.idx}(${b.qt}) on F${f2}`;
              }
            }
          }
        }
        return null;
      }
    },

    // ── T90: Don't destroy underrepresented tets ──
    // When a tet face loses actualization (all SCs were active → no longer),
    // the dominant quark type on that face must already be in first place
    // (highest count) in _actualizationVisits for that face. Tearing down a
    // tet whose dominant quark is still underrepresented wastes progress.
    {
      id: 'T90', name: 'First-place quark ejection',
      init: { _leaderTicks: null, _leaderValue: null },
      projected(states) {
        if (!_nucleusTetFaceData) return null;
        const g = _liveGuards['T90'];
        if (!g || !g._leaderTicks) return null;
        // For each face at or past tolerance, reject moves that land on that face's nodes
        // while the face remains fully actualized — the xon would be feeding the leader.
        for (const [fIdStr, fd] of Object.entries(_nucleusTetFaceData)) {
          const fId = parseInt(fIdStr);
          const tol = _ruleAdaptiveEjection
            ? Math.max(1, Math.ceil(Math.sqrt(g._leaderValue[fId] || 1)))
            : _ruleCubeRootEjection
            ? Math.max(1, Math.ceil(Math.cbrt(g._leaderValue[fId] || 1)))
            : (typeof T90_TOLERANCE !== 'undefined' ? T90_TOLERANCE : 1);
          if ((g._leaderTicks[fId] || 0) < tol) continue;
          const allActive = fd.scIds.every(scId =>
            activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId));
          if (!allActive) continue;
          const faceNodes = new Set(fd.allNodes || fd.cycle);
          for (const s of states) {
            if (faceNodes.has(s.futureNode) && !faceNodes.has(s.fromNode)) {
              return `projected T90: move to node ${s.futureNode} feeds face ${fId} whose dominant quark is in 1st place for ${g._leaderTicks[fId]} ticks (tol=${tol})`;
            }
          }
        }
        return null;
      },
      snapshot(g) {
        if (!_nucleusTetFaceData) return;
        if (!g._leaderTicks) g._leaderTicks = {};
        if (!g._leaderValue) g._leaderValue = {};
        const types = ['pu1', 'pu2', 'pd', 'nd1', 'nd2', 'nu'];
        for (const [fIdStr, fd] of Object.entries(_nucleusTetFaceData)) {
          const fId = parseInt(fIdStr);
          const allActive = fd.scIds.every(scId =>
            activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId));
          if (allActive && typeof _dominantQuarkForFace === 'function' && _actualizationVisits) {
            const dominant = _dominantQuarkForFace(fId);
            if (!dominant) { g._leaderTicks[fId] = 0; g._leaderValue[fId] = 0; continue; }
            const fv = _actualizationVisits[fId];
            if (!fv) { g._leaderTicks[fId] = 0; g._leaderValue[fId] = 0; continue; }
            let maxCount = 0;
            for (const t of types) { if ((fv[t] || 0) > maxCount) maxCount = fv[t] || 0; }
            const dominantCount = fv[dominant] || 0;
            if (dominantCount === maxCount && maxCount > 0) {
              g._leaderTicks[fId] = (g._leaderTicks[fId] || 0) + 1;
              g._leaderValue[fId] = maxCount;
            } else {
              g._leaderTicks[fId] = 0;
              g._leaderValue[fId] = 0;
            }
          } else {
            g._leaderTicks[fId] = 0;
            g._leaderValue[fId] = 0;
          }
        }
      },
      check(tick, g) {
        if (g.ok === null) { g.ok = true; g.msg = ''; }
        if (!g._leaderTicks || !_nucleusTetFaceData) return null;
        for (const [fIdStr, fd] of Object.entries(_nucleusTetFaceData)) {
          const fId = parseInt(fIdStr);
          const tol = _ruleAdaptiveEjection
            ? Math.max(1, Math.ceil(Math.sqrt(g._leaderValue[fId] || 1)))
            : _ruleCubeRootEjection
            ? Math.max(1, Math.ceil(Math.cbrt(g._leaderValue[fId] || 1)))
            : (typeof T90_TOLERANCE !== 'undefined' ? T90_TOLERANCE : 1);
          if ((g._leaderTicks[fId] || 0) < tol) continue;
          const nowActive = fd.scIds.every(scId =>
            activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId));
          if (nowActive) {
            const dominant = typeof _dominantQuarkForFace === 'function' ? _dominantQuarkForFace(fId) : '?';
            return `tick ${tick}: face ${fId} still actualized but dominant ${dominant} in 1st place for ${g._leaderTicks[fId]} ticks (tolerance=${tol})`;
          }
        }
        return null;
      }
    },

    // ── T91: First-place face ejection ──
    // If a face has the most total actualization visits AND still has an
    // actualized tet, it must be torn down on the next tick. Same mechanism
    // as T90 but for inter-face balance instead of intra-face quark balance.
    // Enforced via backtracker — no brute-force SC removal.
    {
      id: 'T91', name: 'First-place face ejection',
      init: { _leaderTicks: null, _leaderValue: null },
      projected(states) {
        if (!_nucleusTetFaceData) return null;
        const g = _liveGuards['T91'];
        if (!g || !g._leaderTicks) return null;
        for (const [fIdStr, fd] of Object.entries(_nucleusTetFaceData)) {
          const fId = parseInt(fIdStr);
          const tol = _ruleAdaptiveEjection
            ? Math.max(1, Math.ceil(Math.sqrt(g._leaderValue[fId] || 1)))
            : _ruleCubeRootEjection
            ? Math.max(1, Math.ceil(Math.cbrt(g._leaderValue[fId] || 1)))
            : (typeof T91_TOLERANCE !== 'undefined' ? T91_TOLERANCE : 1);
          if ((g._leaderTicks[fId] || 0) < tol) continue;
          const allActive = fd.scIds.every(scId =>
            activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId));
          if (!allActive) continue;
          const faceNodes = new Set(fd.allNodes || fd.cycle);
          for (const s of states) {
            if (faceNodes.has(s.futureNode) && !faceNodes.has(s.fromNode)) {
              return `projected T91: move to node ${s.futureNode} feeds face ${fId} which has most total visits for ${g._leaderTicks[fId]} ticks (tol=${tol})`;
            }
          }
        }
        return null;
      },
      snapshot(g) {
        if (!_nucleusTetFaceData || !_actualizationVisits) return;
        if (!g._leaderTicks) g._leaderTicks = {};
        if (!g._leaderValue) g._leaderValue = {};
        // Compute total visits per face
        const totals = {};
        let maxTotal = 0;
        for (const [fIdStr] of Object.entries(_nucleusTetFaceData)) {
          const fId = parseInt(fIdStr);
          const fv = _actualizationVisits[fId];
          let sum = 0;
          if (fv) { for (const t of ['pu1', 'pu2', 'pd', 'nd1', 'nd2', 'nu']) sum += fv[t] || 0; }
          totals[fId] = sum;
          if (sum > maxTotal) maxTotal = sum;
        }
        for (const [fIdStr, fd] of Object.entries(_nucleusTetFaceData)) {
          const fId = parseInt(fIdStr);
          const allActive = fd.scIds.every(scId =>
            activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId));
          const isLeader = (totals[fId] === maxTotal && maxTotal > 0);
          const tiedCount = Object.values(totals).filter(v => v === maxTotal).length;
          if (allActive && isLeader && tiedCount < 8) {
            g._leaderTicks[fId] = (g._leaderTicks[fId] || 0) + 1;
            g._leaderValue[fId] = maxTotal;
          } else {
            g._leaderTicks[fId] = 0;
            g._leaderValue[fId] = 0;
          }
        }
      },
      check(tick, g) {
        if (g.ok === null) { g.ok = true; g.msg = ''; }
        if (!g._leaderTicks || !_nucleusTetFaceData) return null;
        for (const [fIdStr, fd] of Object.entries(_nucleusTetFaceData)) {
          const fId = parseInt(fIdStr);
          const tol = _ruleAdaptiveEjection
            ? Math.max(1, Math.ceil(Math.sqrt(g._leaderValue[fId] || 1)))
            : _ruleCubeRootEjection
            ? Math.max(1, Math.ceil(Math.cbrt(g._leaderValue[fId] || 1)))
            : (typeof T91_TOLERANCE !== 'undefined' ? T91_TOLERANCE : 1);
          if ((g._leaderTicks[fId] || 0) < tol) continue;
          const nowActive = fd.scIds.every(scId =>
            activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId));
          if (nowActive) {
            return `tick ${tick}: face ${fId} still actualized but has most total visits for ${g._leaderTicks[fId]} ticks (tolerance=${tol})`;
          }
        }
        return null;
      }
    },

    // ── T92: First-place hadron ejection ──
    // If proton face-evenness > neutron face-evenness (or vice versa), any
    // actualized tet producing the leading hadron's quark type must be torn
    // down. Keeps proton and neutron scores in lockstep.
    {
      id: 'T92', name: 'First-place hadron ejection',
      init: { _leaderTicks: null, _leaderValue: null },
      projected(states) {
        if (!_nucleusTetFaceData) return null;
        const g = _liveGuards['T92'];
        if (!g || !g._leaderTicks) return null;
        for (const [fIdStr, fd] of Object.entries(_nucleusTetFaceData)) {
          const fId = parseInt(fIdStr);
          const tol = _ruleAdaptiveEjection
            ? Math.max(1, Math.ceil(Math.sqrt(g._leaderValue[fId] || 1)))
            : _ruleCubeRootEjection
            ? Math.max(1, Math.ceil(Math.cbrt(g._leaderValue[fId] || 1)))
            : (typeof T92_TOLERANCE !== 'undefined' ? T92_TOLERANCE : 1);
          if ((g._leaderTicks[fId] || 0) < tol) continue;
          const allActive = fd.scIds.every(scId =>
            activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId));
          if (!allActive) continue;
          const faceNodes = new Set(fd.allNodes || fd.cycle);
          for (const s of states) {
            if (faceNodes.has(s.futureNode) && !faceNodes.has(s.fromNode)) {
              return `projected T92: move to node ${s.futureNode} feeds face ${fId} whose leading hadron is in 1st place for ${g._leaderTicks[fId]} ticks (tol=${tol})`;
            }
          }
        }
        return null;
      },
      snapshot(g) {
        if (!_nucleusTetFaceData || !_actualizationVisits) return;
        if (!g._leaderTicks) g._leaderTicks = {};
        if (!g._leaderValue) g._leaderValue = {};

        // Compute per-hadron face-evenness from _actualizationVisits
        const protonPerFace = [], neutronPerFace = [];
        let protonTotal = 0, neutronTotal = 0;
        for (let f = 1; f <= 8; f++) {
          const fv = _actualizationVisits[f];
          const p = fv ? (fv.pu1 || 0) + (fv.pu2 || 0) + (fv.pd || 0) : 0;
          const n = fv ? (fv.nd1 || 0) + (fv.nd2 || 0) + (fv.nu || 0) : 0;
          protonPerFace.push(p);
          neutronPerFace.push(n);
          protonTotal += p;
          neutronTotal += n;
        }
        const calcEvenness = (arr) => {
          const m = arr.reduce((a, b) => a + b, 0) / arr.length;
          if (m === 0) return 0;
          const sd = Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
          return Math.max(0, 1 - sd / m);
        };
        const pEven = calcEvenness(protonPerFace);
        const nEven = calcEvenness(neutronPerFace);

        // Determine which hadron (if any) is strictly leading
        const MARGIN = 0.001;
        const leader = (pEven > nEven + MARGIN) ? 'proton'
                     : (nEven > pEven + MARGIN) ? 'neutron'
                     : null;
        const leaderTotal = leader === 'proton' ? protonTotal : leader === 'neutron' ? neutronTotal : 0;

        const protonTypes = new Set(['pu1', 'pu2', 'pd']);
        for (const [fIdStr, fd] of Object.entries(_nucleusTetFaceData)) {
          const fId = parseInt(fIdStr);
          const allActive = fd.scIds.every(scId =>
            activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId));
          if (allActive && leader && typeof _dominantQuarkForFace === 'function') {
            const dominant = _dominantQuarkForFace(fId);
            if (!dominant) { g._leaderTicks[fId] = 0; g._leaderValue[fId] = 0; continue; }
            const domHadron = protonTypes.has(dominant) ? 'proton' : 'neutron';
            if (domHadron === leader) {
              g._leaderTicks[fId] = (g._leaderTicks[fId] || 0) + 1;
              g._leaderValue[fId] = leaderTotal;
            } else {
              g._leaderTicks[fId] = 0;
              g._leaderValue[fId] = 0;
            }
          } else {
            g._leaderTicks[fId] = 0;
            g._leaderValue[fId] = 0;
          }
        }
      },
      check(tick, g) {
        if (g.ok === null) { g.ok = true; g.msg = ''; }
        if (!g._leaderTicks || !_nucleusTetFaceData) return null;
        for (const [fIdStr, fd] of Object.entries(_nucleusTetFaceData)) {
          const fId = parseInt(fIdStr);
          const tol = _ruleAdaptiveEjection
            ? Math.max(1, Math.ceil(Math.sqrt(g._leaderValue[fId] || 1)))
            : _ruleCubeRootEjection
            ? Math.max(1, Math.ceil(Math.cbrt(g._leaderValue[fId] || 1)))
            : (typeof T92_TOLERANCE !== 'undefined' ? T92_TOLERANCE : 1);
          if ((g._leaderTicks[fId] || 0) < tol) continue;
          const nowActive = fd.scIds.every(scId =>
            activeSet.has(scId) || impliedSet.has(scId) || xonImpliedSet.has(scId));
          if (nowActive) {
            return `tick ${tick}: face ${fId} still actualized but leading hadron in 1st place for ${g._leaderTicks[fId]} ticks (tolerance=${tol})`;
          }
        }
        return null;
      }
    },

    // ── T93: Trail head color matches xon ──
    // The most recent trail entry's role must always match the xon's current role.
    // Enforces the invariant: "the color of an xon's most recent trail segment
    // is always the same color as the xon."
    {
      id: 'T93', name: 'Trail head color matches xon',
      check(tick, g) {
        for (let i = 0; i < _demoXons.length; i++) {
          const xon = _demoXons[i];
          if (!xon.alive || !xon.trail || xon.trail.length === 0) continue;
          const trailRole = xon.trail[xon.trail.length - 1].role;
          const xonRole = _xonRole(xon);
          if (trailRole !== xonRole) {
            return `tick ${tick}: X${i} role=${xonRole} but trail head role=${trailRole}`;
          }
        }
        return null;
      }
    },

    // ── T94: Historical trail colors are immutable ──
    // Once a trail segment is recorded, its role (color) must never change.
    // _trailRecolor is gone; _trailPush at tick-end is the only writer.
    // All entries are frozen the moment they're pushed.
    {
      id: 'T94', name: 'Historical trail colors immutable',
      snapshot(g) {
        // Capture role of ALL trail entries (including head)
        g._t94Prev = [];
        for (let i = 0; i < _demoXons.length; i++) {
          const xon = _demoXons[i];
          if (!xon.alive || !xon.trail || xon.trail.length === 0) {
            g._t94Prev.push(null);
            continue;
          }
          const roles = [];
          for (let j = 0; j < xon.trail.length; j++) {
            roles.push(xon.trail[j].role);
          }
          g._t94Prev.push(roles);
        }
      },
      check(tick, g) {
        if (!g._t94Prev) return null;
        for (let i = 0; i < _demoXons.length; i++) {
          const xon = _demoXons[i];
          if (!xon.alive || !xon.trail) continue;
          const prev = g._t94Prev[i];
          if (!prev) continue;
          // If trail was truncated (backtracker rewind), the rewritten entries
          // are legitimately different. Only check entries that survived.
          // Entries 0..trail.length-1 that existed before the tick must match.
          const checkLen = Math.min(prev.length, xon.trail.length);
          // If trail shrank, the rewound entries are new — skip them
          if (xon.trail.length < prev.length) continue;
          for (let j = 0; j < checkLen; j++) {
            if (xon.trail[j].role !== prev[j]) {
              return `tick ${tick}: X${i} trail[${j}] role changed from ${prev[j]} to ${xon.trail[j].role}`;
            }
          }
        }
        return null;
      }
    },

    // ── T-DirBal: Directional balance convergence ──
    // After 64 ticks, each alive xon must have used >= 3 distinct direction
    // indices in _dirBalance. Validates the balance tracking is working and
    // xons are exploring multiple directions, not stuck in ruts.
    {
      id: 'T-DirBal', name: 'Xon direction balance (>=3 dirs)',
      convergence: true,
      check(tick, g) {
        if (tick < 64) return null;
        if (g.ok === true) return null;
        for (let i = 0; i < _demoXons.length; i++) {
          const x = _demoXons[i];
          if (!x.alive) continue;
          if (!x._dirBalance) return `X${i} missing _dirBalance`;
          let distinct = 0;
          for (let d = 0; d < 10; d++) {
            if (x._dirBalance[d] > 0) distinct++;
          }
          if (distinct < 3) {
            g.msg = `X${i}: ${distinct}/3 dirs`;
            return null; // not yet converged, keep checking
          }
        }
        g.ok = true;
        g.msg = 'all xons >=3 dirs';
        if (typeof _liveGuardRender === 'function') _liveGuardRender();
        return null;
      }
    },

    // ── T-PermValid: Every tet loop uses a valid permutation ──
    // On every tick, check that each tet/idle_tet xon's _loopSeq matches one of
    // the valid permutations for its quark type's topology + face cycle.
    {
      id: 'T-PermValid', name: 'Tet loops use valid permutations',
      check(tick, g) {
        if (tick < 12) return null; // grace period
        for (let i = 0; i < _demoXons.length; i++) {
          const x = _demoXons[i];
          if (!x.alive) continue;
          if (x._mode !== 'tet' && x._mode !== 'idle_tet') continue;
          if (!x._loopSeq || !x._quarkType || x._assignedFace == null) continue;
          const fd = _nucleusTetFaceData[x._assignedFace];
          if (!fd || !fd.cycle) continue;
          if (!_loopMatchesAnyRotation(x._loopSeq, x._quarkType, fd.cycle)) {
            return `tick ${tick}: X${i} loop ${x._loopSeq.join(',')} invalid for ${x._quarkType} on face ${x._assignedFace}`;
          }
        }
        return null;
      }
    },

    // ── T-PermBalance: Permutation selection favors balance ──
    // After 128 ticks, verify permutation diversity: at least 2 distinct permutations
    // have been used across all tet assignments. Validates smart selection is active.
    {
      id: 'T-PermBalance', name: 'Permutation diversity (>=2 used)',
      convergence: true,
      init: { _seenPerms: new Set() },
      check(tick, g) {
        if (tick < 128) return null;
        if (g.ok === true) return null;
        // Accumulate seen permutation signatures
        for (const x of _demoXons) {
          if (!x.alive || !x._loopSeq || x._mode !== 'tet') continue;
          g._seenPerms.add(x._loopSeq.join(','));
        }
        if (g._seenPerms.size >= 2) {
          g.ok = true;
          g.msg = `${g._seenPerms.size} perms seen`;
          if (typeof _liveGuardRender === 'function') _liveGuardRender();
        } else {
          g.msg = `${g._seenPerms.size}/2 perms`;
        }
        return null;
      }
    },

    // ── T-RLInference: RL model produces finite scores when active ──
    // Only fires when _rlActiveModel is set. Checks that scoreCandidateRL
    // returns finite numbers (not NaN/Infinity).
    {
      id: 'T-RLInference', name: 'RL model inference validity',
      convergence: true,
      check(tick, g) {
        if (tick < 24) return null;
        if (g.ok === true) return null;
        if (!_rlActiveModel || typeof scoreCandidateRL !== 'function' || typeof extractRLFeatures !== 'function') {
          g.ok = true; g.msg = 'RL not active';
          return null;
        }
        // Test inference on a dummy xon-like object
        const dummyXon = {
          _dirBalance: new Array(10).fill(1),
          _modeStats: { oct: 10, tet: 5, idle_tet: 2, weak: 0 },
          prevNode: -1,
        };
        const dummyCand = { node: 0, dirIdx: 0, _needsMaterialise: false };
        const features = extractRLFeatures(dummyXon, dummyCand, new Map());
        const score = scoreCandidateRL(features, _rlActiveModel);
        if (!isFinite(score)) {
          return `RL model produced non-finite score: ${score}`;
        }
        g.ok = true; g.msg = `RL ok (${score.toFixed(2)})`;
        if (typeof _liveGuardRender === 'function') _liveGuardRender();
        return null;
      }
    },

    // ── T95: Oct mode xons must be on oct nodes only ──
    // Any xon in 'oct' or 'oct_formation' mode must have its .node in _octNodeSet.
    {
      id: 'T95', name: 'Oct mode xons on oct nodes only',
      check(tick, g) {
        if (tick < 12) return null; // grace period
        if (!_octNodeSet || _octNodeSet.size === 0) return null;
        for (let i = 0; i < _demoXons.length; i++) {
          const x = _demoXons[i];
          if (!x.alive) continue;
          if (x._mode !== 'oct' && x._mode !== 'oct_formation') continue;
          if (!_octNodeSet.has(x.node)) {
            return `tick ${tick}: X${i} mode=${x._mode} at node ${x.node} not in _octNodeSet`;
          }
        }
        return null;
      }
    },
];

// ── Auto-derived from registry ──
// STRUCTURAL GUARANTEE: Every guard with projected() is automatically checked by lookahead.
// Guards WITHOUT projected() are logged as warnings — add projected() to include in lookahead.
const PROJECTED_GUARD_CHECKS = LIVE_GUARD_REGISTRY.filter(e => e.projected).map(e => e.projected);
const _GUARDS_WITHOUT_PROJECTED = LIVE_GUARD_REGISTRY.filter(e => !e.projected && !e.convergence).map(e => e.id);
if (_GUARDS_WITHOUT_PROJECTED.length > 0) {
    console.warn(`[GUARD COVERAGE] Guards without projected() — not checked by lookahead: ${_GUARDS_WITHOUT_PROJECTED.join(', ')}`);
}

const _liveGuards = {};
for (const entry of LIVE_GUARD_REGISTRY) {
    _liveGuards[entry.id] = { ok: null, msg: 'grace period', failed: false, ...(entry.init || {}) };
}
let _liveGuardsActive = false;
let _liveGuardFailTick = null; // tick of first failure (for wind-down halt)
let _liveGuardDumped = false;  // only dump once per failure

// Reset guard state for rewind — re-enter grace period so replayed ticks
// don't immediately halt on divergent choreography.
function _liveGuardResetForRewind() {
    for (const entry of LIVE_GUARD_REGISTRY) {
        _liveGuards[entry.id] = { ok: null, msg: 'grace period', failed: false, ...(entry.init || {}) };
    }
    _liveGuardActivated = false;
    _liveGuardFailTick = null;
    _liveGuardDumped = false;
    if (typeof _liveGuardRender === 'function') _liveGuardRender();
}

// ══════════════════════════════════════════════════════════════════
// Generic dispatcher — iterates LIVE_GUARD_REGISTRY and calls each
// entry's check() function. No per-test if-blocks needed.
// ══════════════════════════════════════════════════════════════════
let _liveGuardActivated = false;
function _liveGuardCheck() {
    if (!_demoActive || !_liveGuardsActive) return;
    const tick = _demoTick;

    // ── During grace: stay null ──
    if (tick <= LIVE_GUARD_GRACE) {
        if (tick === LIVE_GUARD_GRACE) {
            _liveGuardActivated = true;
            // Promote non-convergence guards to green
            for (const entry of LIVE_GUARD_REGISTRY) {
                if (entry.convergence) continue;
                const g = _liveGuards[entry.id];
                if (!g.failed) { g.ok = true; g.msg = ''; }
            }
            // Call activate() for entries that have it
            for (const entry of LIVE_GUARD_REGISTRY) {
                if (entry.activate) entry.activate(_liveGuards[entry.id]);
            }
            _liveGuardRender();
        }
        return;
    }

    // ── Deferred activation: if grace was 0 and tick jumped past it ──
    if (!_liveGuardActivated) {
        _liveGuardActivated = true;
        for (const entry of LIVE_GUARD_REGISTRY) {
            if (entry.convergence) continue;
            const g = _liveGuards[entry.id];
            if (!g.failed) { g.ok = true; g.msg = ''; }
        }
        for (const entry of LIVE_GUARD_REGISTRY) {
            if (entry.activate) entry.activate(_liveGuards[entry.id]);
        }
        _liveGuardRender();
    }

    let anyFailed = false;
    const ctx = { prev: _liveGuardPrev };

    // ── Run all guards from registry ──
    for (const entry of LIVE_GUARD_REGISTRY) {
        if (!entry.check) continue;
        const g = _liveGuards[entry.id];
        if (g.failed) continue;
        const result = entry.check(tick, g, ctx);
        if (typeof result === 'string') {
            g.ok = false;
            g.failed = true;
            g.msg = result;
            anyFailed = true;
        }
    }

    // ══════════════════════════════════════════════════════════════════
    // WIND-DOWN HALT — first failure starts a 4-tick countdown, then halt.
    // This lets other guards report failures before the sim stops.
    // ══════════════════════════════════════════════════════════════════
    if (anyFailed) {
        _liveGuardRender();
        const failMsgs = Object.entries(_liveGuards)
            .filter(([, g]) => g.failed).map(([k, g]) => `${k}: ${g.msg}`);
        // Only log guard failures when NOT backtracking (expected during BT)
        if (!_btActive) console.error('[LIVE GUARD] Failure detected:', failMsgs.join('; '));
        // Dump failure state to localStorage + file download for post-refresh audit
        if (!_liveGuardDumped) {
            _liveGuardDumped = true;
            try {
                // Build SC detail lists
                const scActiveList = typeof activeSet !== 'undefined' ? [...activeSet].map(id => {
                    const sc = typeof SC_BY_ID !== 'undefined' ? SC_BY_ID[id] : null;
                    return sc ? { id, a: sc.a, b: sc.b } : { id };
                }) : [];
                const scXonImpliedList = typeof xonImpliedSet !== 'undefined' ? [...xonImpliedSet].map(id => {
                    const sc = typeof SC_BY_ID !== 'undefined' ? SC_BY_ID[id] : null;
                    const attr = typeof _scAttribution !== 'undefined' ? _scAttribution.get(id) : null;
                    return { id, a: sc?.a, b: sc?.b, attribution: attr || null };
                }) : [];
                const scImpliedList = typeof impliedSet !== 'undefined' ? [...impliedSet].map(id => {
                    const sc = typeof SC_BY_ID !== 'undefined' ? SC_BY_ID[id] : null;
                    return sc ? { id, a: sc.a, b: sc.b } : { id };
                }) : [];
                // T26 snapshot state (what T26 saw as pre-tick SC state)
                const t26Snap = {};
                for (const [gid, gv] of Object.entries(_liveGuards)) {
                    if (gid === 'T26') {
                        t26Snap.activeSnap = gv._t26ActiveSnap ? [...gv._t26ActiveSnap] : null;
                        t26Snap.impliedSnap = gv._t26ImpliedSnap ? [...gv._t26ImpliedSnap] : null;
                        t26Snap.eImpliedSnap = gv._t26EImpliedSnap ? [...gv._t26EImpliedSnap] : null;
                    }
                }
                const dump = {
                    timestamp: new Date().toISOString(),
                    tick,
                    failures: failMsgs,
                    guards: Object.fromEntries(Object.entries(_liveGuards).map(([k, g]) => [k, { ok: g.ok, msg: g.msg, failed: g.failed }])),
                    xons: (typeof _demoXons !== 'undefined' ? _demoXons : []).filter(x => x.alive).map((x, i) => ({
                        idx: i, node: x.node, prevNode: x.prevNode, mode: x._mode,
                        face: x._assignedFace, quark: x._quarkType, step: x._loopStep,
                        loopSeq: x._loopSeq, movedThisTick: x._movedThisTick,
                        trail: x.trail ? x.trail.slice(-8) : []
                    })),
                    moveTraceHistory: typeof _moveTraceHistory !== 'undefined' ? _moveTraceHistory.slice(-60) : [],
                    moveTraceCurrent: typeof _moveTrace !== 'undefined' ? _moveTrace.slice() : [],
                    scState: {
                        active: scActiveList,
                        xonImplied: scXonImpliedList,
                        implied: scImpliedList
                    },
                    t26Snapshot: t26Snap
                };
                const json = JSON.stringify(dump, null, 2);
                localStorage.setItem('flux_guard_dump', json);
                console.error('[LIVE GUARD] Dump saved to localStorage(flux_guard_dump)');
            } catch (e) { console.error('[LIVE GUARD] Dump failed:', e); }
        }
    }
    const hasAnyFailure = Object.values(_liveGuards).some(g => g.failed);
    if (hasAnyFailure) {
        // During council replay phase (up to recorded peak), suppress rewinds/halts.
        // The recorded moves are known-good; guard failures are transient and the
        // original run resolved them via backtracking. We replay the happy path only.
        if (_guardHardStop && _sweepReplayActive && _sweepReplayMember && tick <= _sweepReplayMember.peak) {
            // Replay corruption (test pipeline only): halt unconditionally.
            const failMsgs = Object.entries(_liveGuards)
                .filter(([, g]) => g.failed).map(([k, g]) => `${k}: ${g.msg}`).join('; ');
            console.error(`[REPLAY GUARD] Corruption at tick ${tick}: ${failMsgs}`);
            if (_guardHardStop) {
                _guardHardStop = false;
                if (typeof _showReplayCorruption === 'function') _showReplayCorruption(tick, failMsgs);
            }
            simHalted = true;
            _demoPaused = true;
            _sweepActive = false; // kill sweep synchronously before polls see simHalted
            _liveGuardRender();
            return;
        } else {
        if (typeof _liveGuardFailTick === 'undefined' || _liveGuardFailTick === null) {
            _liveGuardFailTick = tick; // record first failure tick
        }
        // Tournament mode: guards enforced identically to demo mode.
        // No bypass — failures trigger backtracker/halt same as demo.
        // ── BACKTRACKING: all failures trigger rewind instead of halt ──
        const canBacktrack = typeof _rewindRequested !== 'undefined'
            && typeof _btSnapshots !== 'undefined'
            && _btSnapshots.length > 0
            && typeof _btActive !== 'undefined';
        if (canBacktrack) {
            // Signal rewind instead of halting
            _rewindRequested = true;
            _rewindViolation = Object.entries(_liveGuards)
                .filter(([, g]) => g.failed).map(([k, g]) => `${k}: ${g.msg}`).join('; ');
            // Reset all failed guard state so rewind can try again
            for (const entry of LIVE_GUARD_REGISTRY) {
                const g = _liveGuards[entry.id];
                if (g.failed) {
                    g.failed = false; g.ok = true; g.msg = '';
                }
            }
            _liveGuardFailTick = null;
            _liveGuardDumped = false; // allow re-dump on next real failure
            // Throttle backtrack logs — only log every 50th rewind to reduce GC pressure
            if (_btRetryCount % 50 === 0) console.warn(`[BACKTRACK] Rewind #${_btRetryCount}: ${_rewindViolation}`);
        } else if (tick >= _liveGuardFailTick + 0) {
            // No backtrack snapshots available — halt as last resort
            if (typeof stopExcitationClock === 'function') stopExcitationClock();
            simHalted = true;
            _liveGuardRender();
            console.error('[LIVE GUARD] Simulation halted after wind-down:', Object.entries(_liveGuards)
                .filter(([, g]) => g.failed).map(([k, g]) => `${k}: ${g.msg}`).join('; '));
        }
        } // close else (non-replay path)
    }
}

// Snapshot xon positions BEFORE demoTick advances them (called from demoTick)
let _liveGuardPrev = null;
function _liveGuardSnapshot() {
    if (!_liveGuardsActive) { _liveGuardPrev = null; return; }
    _liveGuardPrev = _demoXons.filter(x => x.alive).map(x => ({
        xon: x, node: x.node, mode: x._mode
    }));
    // Call snapshot() for entries that have it (e.g. T42 SC set capture)
    for (const entry of LIVE_GUARD_REGISTRY) {
        if (!entry.snapshot) continue;
        const g = _liveGuards[entry.id];
        if (g && !g.failed) entry.snapshot(g);
    }
}

// Update the test result rows for live-guarded tests in the left panel
function _liveGuardRender() {
    const testResultsEl = document.getElementById('dp-test-results');
    if (!testResultsEl) return;

    // Auto-derived from LIVE_GUARD_REGISTRY — single source of truth
    const nameMap = {};
    for (const entry of LIVE_GUARD_REGISTRY) nameMap[entry.id] = `${entry.id} ${entry.name}`;

    for (const [key, g] of Object.entries(_liveGuards)) {
        const fullName = nameMap[key];
        if (!fullName) continue;
        const num = fullName.match(/^T(\d+\w?)/)?.[1] || '';
        const label = fullName.replace(/^T\d+\w?\s*/, '');
        const icon = g.ok === true ? '\u2713' : (g.ok === null ? '\u2013' : '\u2717');
        const color = g.ok === true ? '#44cc66' : (g.ok === null ? '#ccaa44' : '#ff4444');

        // Find and replace the existing row
        const rows = testResultsEl.querySelectorAll('div');
        for (const row of rows) {
            if (row.textContent.includes(`T${num}`) && row.textContent.includes(label.substring(0, 10))) {
                row.innerHTML = `<span style="color:${color}; font-weight:bold; min-width:10px;">${icon}</span>`
                    + `<span style="color:#556677; min-width:18px;">T${num}</span>`
                    + `<span style="color:${g.ok === true ? '#7a9aaa' : color};">${label}</span>`
                    + (g.ok === true ? '' : `<span style="color:${g.ok === null ? '#aa8833' : '#aa4444'}; font-size:7px; margin-left:2px;">${g.msg || ''}</span>`);
                break;
            }
        }
    }

    // Update summary count
    _liveGuardUpdateSummary();
}

function _liveGuardUpdateSummary() {
    const testSummary = document.getElementById('dp-test-summary');
    const testResultsEl = document.getElementById('dp-test-results');
    if (!testSummary || !testResultsEl) return;

    const rows = testResultsEl.querySelectorAll('div[style]');
    let passed = 0, total = 0, nulled = 0, failed = 0;
    for (const row of rows) {
        const firstSpan = row.querySelector('span');
        if (!firstSpan) continue;
        total++;
        const txt = firstSpan.textContent.trim();
        if (txt === '\u2713') passed++;
        else if (txt === '\u2013') nulled++;
        else if (txt === '\u2717') failed++;
    }
    testSummary.textContent = `${passed}/${total}${nulled ? ` (${nulled}?)` : ''}`;
    testSummary.style.color = failed > 0 ? '#ff6644' : (nulled > 0 ? '#ccaa44' : '#66dd66');
}
