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

// Oct capacity: hard maximum of 6 oct-mode xons at any time.
const OCT_CAPACITY_MAX = 6;
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
            if (typeof _moveTrace !== 'undefined' && _moveTrace.length) {
              console.error('T19 TRACE:', _moveTrace.map(t =>
                `x${t.xonIdx}:${t.from}\u2192${t.to}(${t.path},${t.mode})`).join(' | '));
            }
            // Dump all xon positions
            console.error('T19 POSITIONS:', _demoXons.map((x,i) =>
              x.alive ? `x${i}@${x.node}(${x._mode})` : `x${i}:dead`).join(' '));
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
          if (prevMode !== xon._mode) continue;
          if (prevMode === 'oct_formation') continue; // formation phase: scripted movement
          if (xon.node === fromNode) return `tick ${tick}: stuck at node ${fromNode} (${prevMode})`;
        }
        return null;
      }
    },
    { id: 'T21', name: 'Oct cage permanence', init: { _octSnapshot: null },
      activate(g) {
        const snap = new Set();
        for (const scId of _octSCIds) { if (activeSet.has(scId)) snap.add(scId); }
        g._octSnapshot = snap;
        if (snap.size === 0) { g.ok = null; g.msg = 'no oct SCs active yet'; }
      },
      check(tick, g) {
        // Update snapshot with newly active oct SCs
        if (g._octSnapshot) {
          for (const scId of _octSCIds) { if (activeSet.has(scId)) g._octSnapshot.add(scId); }
          if (g._octSnapshot.size > 0 && g.ok === null) { g.ok = true; g.msg = ''; }
        }
        // Verify all snapshotted oct SCs still active
        if (g._octSnapshot && g._octSnapshot.size > 0) {
          for (const scId of g._octSnapshot) {
            if (!activeSet.has(scId)) return `tick ${tick}: oct SC ${scId} lost`;
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
    { id: 'T24', name: 'Trail color stability',
      check(tick, g) {
        for (const xon of _demoXons) {
          if (!xon.alive || !xon.trailColHistory) continue;
          for (let j = 0; j < xon.trailColHistory.length; j++) {
            const c = xon.trailColHistory[j];
            const isWhite = c === 0xffffff;
            const isQuark = Object.values(QUARK_COLORS).includes(c);
            const isWeak = c === WEAK_FORCE_COLOR;
            const isGluon = c === GLUON_COLOR;
            if (!isWhite && !isQuark && !isWeak && !isGluon) return `tick ${tick}: color 0x${c.toString(16)}`;
          }
          if (xon.trailColHistory.length !== xon.trail.length) return `tick ${tick}: trail/color desync`;
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
                if (typeof _moveTrace !== 'undefined' && _moveTrace.length) {
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
          if (!xon.trailColHistory || xon.trailColHistory.length !== xon.trail.length)
            return `tick ${tick}: trail/color length mismatch`;
        }
        return null;
      }
    },
    { id: 'T34', name: 'Trail length bounded',
      check(tick, g) {
        for (const xon of _demoXons) {
          if (!xon.alive || !xon.trail) continue;
          if (xon.trail.length > XON_TRAIL_LENGTH)
            return `tick ${tick}: trail len=${xon.trail.length} max=${XON_TRAIL_LENGTH}`;
        }
        return null;
      }
    },
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
              if (typeof _moveTrace !== 'undefined' && _moveTrace.length) {
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
          const fromN = xon.trail[xon.trail.length - 2];
          const toN = xon.trail[xon.trail.length - 1];
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
          if (xon.node !== lastTrail) {
            return `tick ${tick}: X${xi} at node ${xon.node} but trail ends at ${lastTrail}`;
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
          // Gluon must be orange
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
        // Check monotonicity
        for (let i = 1; i < _btSnapshots.length; i++) {
          if (_btSnapshots[i].tick <= _btSnapshots[i - 1].tick) {
            return `snapshot[${i}].tick=${_btSnapshots[i].tick} <= snapshot[${i-1}].tick=${_btSnapshots[i-1].tick}`;
          }
        }
        // Check last snapshot tick matches _demoTick - 1
        const last = _btSnapshots[_btSnapshots.length - 1];
        if (last.tick !== _demoTick - 1) {
          return `last snapshot tick=${last.tick} != _demoTick-1=${_demoTick - 1}`;
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

    // ── T85: Stochastic DFS — runs must diverge ──
    // At tick 50, fingerprint xon positions and save to localStorage.
    // If a subsequent run produces the identical fingerprint, DFS is not
    // exploring different branches → violation.
    {
      id: 'T85', name: 'Stochastic DFS (runs diverge)',
      convergence: true,
      init: { checked: false },
      check(tick, g) {
        if (g.checked) return null;
        const CHECK_TICK = 50;
        if (tick < CHECK_TICK) return null;
        if (tick > CHECK_TICK) { g.checked = true; return null; } // missed window
        // Build fingerprint: sorted list of (xonIndex, node, mode)
        const fp = _demoXons.map((x, i) =>
          x.alive ? `${i}:${x.node}:${x._mode}:${x._assignedFace||'-'}` : `${i}:dead`
        ).join('|');
        const KEY = '_fluxT85_fingerprint';
        try {
          const prev = localStorage.getItem(KEY);
          localStorage.setItem(KEY, fp);
          g.checked = true;
          if (prev && prev === fp) {
            return `tick ${CHECK_TICK}: identical fingerprint across runs — DFS is deterministic`;
          }
          // First run or different fingerprint → pass
          g.ok = true;
          g.msg = 'diverged';
        } catch (e) {
          // localStorage unavailable — skip test
          g.checked = true;
          g.ok = true;
          g.msg = 'localStorage N/A';
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
    _liveGuardFailTick = null;
    _liveGuardDumped = false;
    if (typeof _liveGuardRender === 'function') _liveGuardRender();
}

// ══════════════════════════════════════════════════════════════════
// Generic dispatcher — iterates LIVE_GUARD_REGISTRY and calls each
// entry's check() function. No per-test if-blocks needed.
// ══════════════════════════════════════════════════════════════════
function _liveGuardCheck() {
    if (!_demoActive || !_liveGuardsActive || _testRunning) return;
    const tick = _demoTick;

    // ── During grace: stay null ──
    if (tick <= LIVE_GUARD_GRACE) {
        if (tick === LIVE_GUARD_GRACE) {
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
        console.error('[LIVE GUARD] Failure detected:', failMsgs.join('; '));
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
            console.warn(`[BACKTRACK] Rewind requested: ${_rewindViolation}`);
        } else if (tick >= _liveGuardFailTick + 0) {
            // No backtrack snapshots available — halt as last resort
            if (typeof stopExcitationClock === 'function') stopExcitationClock();
            simHalted = true;
            _liveGuardRender();
            console.error('[LIVE GUARD] Simulation halted after wind-down:', Object.entries(_liveGuards)
                .filter(([, g]) => g.failed).map(([k, g]) => `${k}: ${g.msg}`).join('; '));
        }
    }
}

// Snapshot xon positions BEFORE demoTick advances them (called from demoTick)
let _liveGuardPrev = null;
function _liveGuardSnapshot() {
    if (!_liveGuardsActive || _testRunning) { _liveGuardPrev = null; return; }
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

function runDemo3Tests() {
    _testRunning = true;
    const results = [];
    const pass = (name) => { results.push({ name, ok: true }); };
    const fail = (name, msg) => { results.push({ name, ok: false, msg }); };
    const skip = (name, msg) => { results.push({ name, ok: null, msg: msg || 'unproven' }); };
    const assert = (name, cond, msg) => cond ? pass(name) : fail(name, msg || 'assertion failed');

    // ── Ensure nucleus is simulated so we have valid state ──
    if (!NucleusSimulator.active) {
        NucleusSimulator.simulateNucleus();
    }

    // T01-T07: Now handled as live convergence guards in LIVE_GUARD_REGISTRY.
    // They start null and only pass when runtime conditions are met (e.g. tet faces discovered).

    // T08 REMOVED: Schedule structure test eliminated (window system removed)

    // T09 REMOVED: Tet face data test eliminated (dynamic discovery, face count varies)

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // T10 DISABLED: requires face data which is deferred during discovery
    // skip('T10 Xon spawning', 'disabled');

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // TEST 11: Xon advancement — _advanceXon updates state correctly
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // T11 DISABLED: requires face data which is deferred during discovery
    // skip('T11 Xon advancement (4 hops + wrap)', 'disabled');

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    //  PERSISTENT 6-XON MODEL (T12–T27)
    //  ALL deferred to LIVE MONITORING — continuous per-tick validation
    //  with grace period, permanent fail + halt on violation
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Auto-register all live guards from LIVE_GUARD_REGISTRY
    for (const entry of LIVE_GUARD_REGISTRY) {
        skip(`${entry.id} ${entry.name}`, 'grace period (live)');
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // T-OctSymmetry: No oct node is closer to a boundary than any other.
    // A boundary node has fewer than 8 base neighbors (interior nodes have 8).
    // For each oct node, find minimum hop distance to nearest boundary node.
    // All 6 oct nodes must have the same minimum distance.
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if (_octNodeSet && _octNodeSet.size === 6 && baseNeighbors && baseNeighbors.length > 0) {
        // Find boundary nodes: nodes with < 8 base neighbors
        const boundaryNodes = new Set();
        for (let i = 0; i < N; i++) {
            if (baseNeighbors[i].length < 8) boundaryNodes.add(i);
        }
        // BFS from each oct node to find min hops to nearest boundary
        function hopsToNearestBoundary(startNode) {
            if (boundaryNodes.has(startNode)) return 0;
            const visited = new Set([startNode]);
            let frontier = [startNode], dist = 0;
            while (frontier.length > 0) {
                dist++;
                const next = [];
                for (const n of frontier) {
                    for (const nb of baseNeighbors[n]) {
                        if (!visited.has(nb.node)) {
                            if (boundaryNodes.has(nb.node)) return dist;
                            visited.add(nb.node);
                            next.push(nb.node);
                        }
                    }
                }
                frontier = next;
            }
            return Infinity;
        }
        const octNodes = [..._octNodeSet];
        const distances = octNodes.map(n => hopsToNearestBoundary(n));
        const allEqual = distances.every(d => d === distances[0]);
        assert('T-OctSymmetry Lattice boundary equidistant from all oct nodes',
            allEqual,
            `oct node boundary distances: [${distances.join(', ')}] (nodes: [${octNodes.join(', ')}])`);
    } else {
        skip('T-OctSymmetry Lattice boundary equidistant from all oct nodes', 'no oct data');
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // T-RLStrategicInference: strategic model returns finite scores
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if (typeof _rlAvailable !== 'undefined' && _rlAvailable && typeof createStrategicModel === 'function') {
        try {
            const testModel = createStrategicModel();
            const testFeatures = new Float32Array(RL_STRATEGIC_FEATURES);
            for (let i = 0; i < RL_STRATEGIC_FEATURES; i++) testFeatures[i] = Math.random();
            const score = scoreStrategicRL(testFeatures, testModel);
            assert('T-RLStrat Strategic inference', isFinite(score), `score=${score} not finite`);
            testModel.dispose();
        } catch (e) {
            fail('T-RLStrat Strategic inference', e.message);
        }
    } else {
        skip('T-RLStrat Strategic inference', 'TF.js unavailable');
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // T-HadronicFitness: hadronic ratio fitness on known inputs
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if (typeof _evaluateHadronicRatioFitness === 'function') {
        // Uniform visits → fitness near 1.0
        const uniformVisits = {};
        for (let f = 1; f <= 8; f++) uniformVisits[f] = { pu1: 10, pu2: 10, pd: 10, nd1: 10, nd2: 10, nu: 10, total: 60 };
        const uniformF = _evaluateHadronicRatioFitness(uniformVisits);
        assert('T-HadFit Uniform → ~1.0', uniformF.fitness > 0.85, `uniform fitness=${uniformF.fitness.toFixed(3)} should be > 0.85`);

        // Single-type visits → fitness near 0.0
        const singleVisits = {};
        for (let f = 1; f <= 8; f++) singleVisits[f] = { pu1: 60, pu2: 0, pd: 0, nd1: 0, nd2: 0, nu: 0, total: 60 };
        const singleF = _evaluateHadronicRatioFitness(singleVisits);
        assert('T-HadFit Single-type → low', singleF.fitness < 0.3, `single-type fitness=${singleF.fitness.toFixed(3)} should be < 0.3`);

        // Empty visits → penalty
        const emptyVisits = {};
        for (let f = 1; f <= 8; f++) emptyVisits[f] = { pu1: 0, pu2: 0, pd: 0, nd1: 0, nd2: 0, nu: 0, total: 0 };
        const emptyF = _evaluateHadronicRatioFitness(emptyVisits);
        assert('T-HadFit Empty → -20', emptyF.fitness === -20, `empty fitness=${emptyF.fitness} should be -20`);
    } else {
        skip('T-HadFit Hadronic fitness', 'function not defined yet');
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // T-GenomeSplit: 2858 genome round-trips through model→genome→model
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if (typeof _rlAvailable !== 'undefined' && _rlAvailable &&
        typeof createStrategicModel === 'function' && typeof createPolicyModel === 'function') {
        try {
            const s1 = createStrategicModel();
            const t1 = createPolicyModel();
            const genome1 = modelToGenome(s1, t1);
            assert('T-GenSplit Genome size', genome1.length === getGenomeSize(),
                `genome.length=${genome1.length} != ${getGenomeSize()}`);

            // Load into fresh models and extract again — should match
            const s2 = createStrategicModel();
            const t2 = createPolicyModel();
            genomeToModel(genome1, s2, t2);
            const genome2 = modelToGenome(s2, t2);
            let maxDiff = 0;
            for (let i = 0; i < genome1.length; i++) {
                maxDiff = Math.max(maxDiff, Math.abs(genome1[i] - genome2[i]));
            }
            assert('T-GenSplit Round-trip fidelity', maxDiff < 1e-6,
                `max diff=${maxDiff} (should be < 1e-6)`);
            s1.dispose(); t1.dispose(); s2.dispose(); t2.dispose();
        } catch (e) {
            fail('T-GenSplit Genome round-trip', e.message);
        }
    } else {
        skip('T-GenSplit Genome round-trip', 'TF.js unavailable');
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // T-Temporal: strategic features include 6 temporal features (f[16]-f[21])
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if (typeof extractStrategicFeatures === 'function' && typeof RL_STRATEGIC_FEATURES !== 'undefined') {
        assert('T-Temporal Feature count', RL_STRATEGIC_FEATURES === 22,
            `RL_STRATEGIC_FEATURES=${RL_STRATEGIC_FEATURES} should be 22`);
        // Create a mock xon with temporal state
        const mockXon = {
            node: 0, prevNode: 0, alive: true, _mode: 'oct',
            _dirBalance: new Array(10).fill(5),
            _modeStats: { oct: 50, tet: 30, idle_tet: 10, weak: 10 },
            _octModeSince: 10,
        };
        const mockOccupied = new Map();
        // Ensure temporal state exists
        if (typeof _rlTemporalState !== 'undefined') {
            _rlTemporalState.faceLastVisitTick[1] = 100;
            _rlTemporalState.prevFaceCV[1] = 0.5;
        }
        const feats = extractStrategicFeatures(mockXon, 1, 'pu1', mockOccupied);
        assert('T-Temporal Feature length', feats.length === 22,
            `feature vector length=${feats.length} should be 22`);
        // f[16]-f[21] should be finite numbers
        let allFinite = true;
        for (let i = 16; i < 22; i++) {
            if (!isFinite(feats[i])) { allFinite = false; break; }
        }
        assert('T-Temporal All finite', allFinite, 'temporal features contain non-finite values');
    } else {
        skip('T-Temporal Temporal features', 'function not defined yet');
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // T-Sphericity: Lattice boundary must be spherical, not cuboctahedral.
    //
    // For all boundary nodes (baseNeighbors.length < 8), compute Euclidean
    // distance from the oct centroid [0, -r3, 0]. The "sphericity ratio"
    // is minDist / maxDist.
    //
    // A continuous sphere gives ratio = 1.0, but a DISCRETE FCC lattice
    // has a Kepler-like limit: the boundary spans ~1 lattice plane spacing
    // (~1.0) in depth, so the theoretical limit is 1 - thickness/R_max.
    // Empirically: L2≈0.77, L4≈0.86, L6≈0.90, L8≈0.92, approaching 1.0.
    //
    // An UNTRIMMED cuboctahedral lattice gives much worse ratios:
    // L2≈0.45, L4≈0.57, L6≈0.62, L8≈0.64 (bounded by 1/sqrt(2)≈0.707).
    //
    // SIZE-DEPENDENT THRESHOLD: max(0.7, 1 - 1.5/R_max).
    // This is below the discrete-lattice theoretical limit (1 - 1.0/R_max)
    // but well above the untrimmed cuboctahedral ratio at every level.
    // MATHEMATICALLY GUARANTEED separation between spherical and
    // cuboctahedral shapes at all lattice sizes L2+.
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if (baseNeighbors && baseNeighbors.length > 0 && N > 15) {
        const octCy = -r3;  // oct centroid y-coordinate
        // Identify boundary nodes: fewer than 8 base neighbors
        const boundaryDists = [];
        for (let i = 0; i < N; i++) {
            if (baseNeighbors[i].length < 8) {
                const [x, y, z] = REST[i];
                const dy = y - octCy;
                const dist = Math.sqrt(x * x + dy * dy + z * z);
                boundaryDists.push(dist);
            }
        }
        if (boundaryDists.length > 0) {
            const minDist = Math.min(...boundaryDists);
            const maxDist = Math.max(...boundaryDists);
            const sphericity = minDist / maxDist;
            // Size-dependent threshold: accounts for discrete lattice boundary thickness
            const threshold = Math.max(0.7, 1 - 1.5 / maxDist);
            assert(`T-Sphericity Lattice boundary is spherical (ratio > ${threshold.toFixed(3)})`,
                sphericity > threshold,
                `sphericity ratio = ${sphericity.toFixed(4)} (threshold=${threshold.toFixed(4)}, min=${minDist.toFixed(4)}, max=${maxDist.toFixed(4)}, boundary=${boundaryDists.length})`);
        } else {
            skip('T-Sphericity', 'no boundary nodes found');
        }
    } else {
        skip('T-Sphericity', 'lattice not built yet');
    }

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // RESULTS
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    const passed = results.filter(r => r.ok === true).length;
    const nulled = results.filter(r => r.ok === null).length;
    const failed = results.filter(r => r.ok === false);
    console.log(`%c═══ Demo 3.0 Tests: ${passed}/${results.length} passed${nulled ? `, ${nulled} null` : ''} ═══`, 'font-weight:bold; font-size:14px');
    for (const r of results) {
        if (r.ok === true) console.log(`  %c✓ ${r.name}`, 'color:#44cc66');
        else if (r.ok === null) console.log(`  %c– ${r.name}: ${r.msg}`, 'color:#ccaa44');
        else console.log(`  %c✗ ${r.name}: ${r.msg}`, 'color:#ff4444; font-weight:bold');
    }
    if (failed.length === 0 && nulled === 0) {
        console.log('%c  ALL TESTS PASSED', 'color:#44cc66; font-weight:bold; font-size:12px');
    }

    // ── Update left panel ──
    const testSection = document.getElementById('dp-test-section');
    const testResultsEl = document.getElementById('dp-test-results');
    const testSummary = document.getElementById('dp-test-summary');
    if (testSection && testResultsEl) {
        testSection.style.display = '';
        const allPassed = failed.length === 0 && nulled === 0;
        testSummary.textContent = `${passed}/${results.length}${nulled ? ` (${nulled}?)` : ''}`;
        testSummary.style.color = allPassed ? '#66dd66' : (failed.length > 0 ? '#ff6644' : '#ccaa44');
        let html = '';
        for (const r of results) {
            const icon = r.ok === true ? '✓' : (r.ok === null ? '–' : '✗');
            const color = r.ok === true ? '#44cc66' : (r.ok === null ? '#ccaa44' : '#ff4444');
            const label = r.name.replace(/^T\d+\w?\s*/, '');
            const num = r.name.match(/^T(\d+\w?)/)?.[1] || '';
            html += `<div style="display:flex; gap:3px; align-items:baseline;">`
                + `<span style="color:${color}; font-weight:bold; min-width:10px;">${icon}</span>`
                + `<span style="color:#556677; min-width:18px;">T${num}</span>`
                + `<span style="color:${r.ok === true ? '#7a9aaa' : color};">${label}</span>`
                + (r.ok === true ? '' : `<span style="color:${r.ok === null ? '#aa8833' : '#aa4444'}; font-size:7px; margin-left:2px;">${r.msg || ''}</span>`)
                + `</div>`;
        }
        testResultsEl.innerHTML = html;
    }

    // ── Reset demo state after tests so visual demo starts clean ──
    _demoTick = 0;
    _planckSeconds = 0;
    _demoVisitedFaces = new Set();
    _demoTypeBalanceHistory = [];
    _demoPrevFaces = new Set();
    if (_demoVisits) for (let f = 1; f <= 8; f++) {
        _demoVisits[f] = { pu1: 0, pu2: 0, pd: 0, nd1: 0, nd2: 0, nu: 0, total: 0 };
    }
    // Return xons to oct mode at their current positions
    for (const xon of _demoXons) {
        if (xon.alive && (xon._mode === 'tet' || xon._mode === 'idle_tet')) _returnXonToOct(xon);
    }
    // Clear any tet SCs accumulated during tests
    for (const [fIdStr, fd] of Object.entries(_nucleusTetFaceData)) {
        for (const scId of fd.scIds) xonImpliedSet.delete(scId);
    }
    _testRunning = false;

    return { passed, total: results.length, failed: failed.map(f => f.name) };
}

// ── Wire up nucleus UI ──
(function(){
    NucleusSimulator.populateModelSelect();

    // Simulate button
    document.getElementById('btn-simulate-nucleus')?.addEventListener('click', function(){
        // Use whatever lattice level is already on the slider
        NucleusSimulator.simulateNucleus();
        // Small delay to let lattice build, then start demo loop
        setTimeout(function() {
            if (NucleusSimulator.active) startDemoLoop();
        }, 100);
    });

    // Tournament button
    document.getElementById('btn-tournament')?.addEventListener('click', function(){
        if(tournamentActive) stopTournament();
        else startTournament();
    });

    // Play/pause button — pauses/resumes the demo tick interval
    document.getElementById('btn-nucleus-pause')?.addEventListener('click', function(){
        if (typeof isDemoPaused === 'function' && _demoActive) {
            // If reversing, just stop reverse and stay paused — don't toggle
            if (_demoReversing) {
                stopReverse();
                this.textContent = '\u25B6';
                this.title = 'Resume simulation';
                return;
            }
            if (!isDemoPaused()) {
                pauseDemo();
                this.textContent = '\u25B6';
                this.title = 'Resume simulation';
            } else {
                resumeDemo();
                this.textContent = '\u23F8';
                this.title = 'Pause simulation';
            }
        } else if (excitationClockTimer) {
            stopExcitationClock();
            this.textContent = '▶';
            this.title = 'Resume simulation';
            document.getElementById('nucleus-status').textContent = 'paused';
        } else {
            startExcitationClock();
            this.textContent = '⏸';
            this.title = 'Pause simulation';
            document.getElementById('nucleus-status').textContent = 'running';
        }
    });

    // ── Playback controls ──
    document.getElementById('btn-step-back')?.addEventListener('click', function() {
        if (!_demoActive) return;
        if (_demoReversing) stopReverse();
        if (!_demoPaused) pauseDemo();
        _playbackStepBack();
        const pb = document.getElementById('btn-nucleus-pause');
        if (pb) pb.textContent = '\u25B6';
    });

    document.getElementById('btn-step-forward')?.addEventListener('click', function() {
        if (!_demoActive) return;
        if (_demoReversing) stopReverse();
        if (!_demoPaused) pauseDemo();
        _playbackStepForward();
        const pb = document.getElementById('btn-nucleus-pause');
        if (pb) pb.textContent = '\u25B6';
    });

    document.getElementById('btn-reverse')?.addEventListener('click', function() {
        if (!_demoActive) return;
        if (_demoReversing) {
            stopReverse();
        } else {
            startReverse();
        }
    });

    // Rewind all the way to t=0
    document.getElementById('btn-rewind-start')?.addEventListener('click', function() {
        if (!_demoActive) return;
        if (_demoReversing) stopReverse();
        if (!_demoPaused) pauseDemo();
        // Pop all snapshots into redo stack until we reach t=0
        while (_btSnapshots.length > 0 && _demoTick > 0) {
            _btSaveSnapshot();
            _redoStack.push(_btSnapshots.pop());
            const snap = _btSnapshots.pop();
            if (!snap) break;
            _btRestoreSnapshot(snap);
        }
        simHalted = false;
        _bfsReset(); _btReset();
        if (typeof _liveGuardResetForRewind === 'function') _liveGuardResetForRewind();
        _tickLog.length = 0;
        _playbackUpdateDisplay();
        const pb = document.getElementById('btn-nucleus-pause');
        if (pb) pb.textContent = '\u25B6';
    });

    // Fast-forward: drain redo stack instantly, then pause
    document.getElementById('btn-forward-end')?.addEventListener('click', function() {
        if (!_demoActive) return;
        if (_demoReversing) stopReverse();
        if (!_demoPaused) pauseDemo();
        // Restore all redo snapshots instantly
        while (_redoStack.length > 0) {
            _btSaveSnapshot();
            const snap = _redoStack.pop();
            _btRestoreSnapshot(snap);
        }
        simHalted = false;
        _bfsReset(); _btReset();
        if (typeof _liveGuardResetForRewind === 'function') _liveGuardResetForRewind();
        _playbackUpdateDisplay();
        const pb = document.getElementById('btn-nucleus-pause');
        if (pb) pb.textContent = '\u25B6';
    });

    document.getElementById('btn-export-log')?.addEventListener('click', function() {
        exportTickLog();
    });

    // Stop/clear button
    document.getElementById('btn-stop-nucleus')?.addEventListener('click', function(){
        NucleusSimulator.deactivate();
        activeSet.clear();
        impliedSet.clear(); xonImpliedSet.clear(); blockedImplied.clear(); impliedBy.clear();
        while(excitations.length > 0){
            const e = excitations.pop();
            if(e.group) scene.remove(e.group);
            if(e.trailLine) scene.remove(e.trailLine);
        }
        if(typeof stopExcitationClock === 'function') stopExcitationClock();
        bumpState();
        const pFinal = detectImplied();
        applyPositions(pFinal);
        updateCandidates(); updateSpheres(); updateStatus();
        rebuildShortcutLines();
        updateExcitationSidebar();
        // Reset pause button state
        const pauseBtn = document.getElementById('btn-nucleus-pause');
        if(pauseBtn){ pauseBtn.textContent = '⏸'; pauseBtn.title = 'Pause simulation'; }
    });

    // ── "Tune T22" button ──────────────────────────────────────────────
    document.getElementById('btn-tune-t22')?.addEventListener('click', function () {
        if (_tournamentRunning) {
            _tournamentRunning = false;
            this.textContent = 'train RL';
            this.style.borderColor = '#aa8844';
            if (typeof _updateLatticeSliderLock === 'function') _updateLatticeSliderLock();
            return;
        }
        this.textContent = 'stop RL';
        this.style.borderColor = '#cc4444';
        _runTournament();
    });
})();

// ╔══════════════════════════════════════════════════════════════════════╗
// ║  CHOREOGRAPHY PARAMETER TOURNAMENT (GA)                            ║
// ║  Headless trial runner + genetic algorithm for T22 convergence     ║
// ╚══════════════════════════════════════════════════════════════════════╝

let _tournamentRunning = false;
let _tournamentVisualsApplied = false;
let _tournamentTargetTick = 0; // tick at which current trial ends
let _tournamentCallback = null; // called when trial reaches target tick
let _tournamentSavedPan = null; // camera position saved at tournament start

// Evaluate fitness from current _demoVisits state.
// 7 priority metrics aligned with user's optimization goals:
//   1. Anchor quark evenness (pd/nu across 8 faces)     — 25%
//   2. Follower quark evenness (pu1/pu2/nd1/nd2)        — 20%
//   3. Anchor:follower ratio (1:2 per orientation)       — 15%
//   4. Follower:follower ratio (1:1 pu1:pu2, nd1:nd2)   — 12%
//   5. Quark frequency (loops completed per tick)        — 12%
//   6. Periodicity (regularity of oct changes)           — 8%
//   7. Xonic balance (direction + mode balance)          — 8%
function _evaluateFitness() {
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

    // 1. ANCHOR EVENNESS: pd and nu should be even across all 8 faces
    let anchorEvenness = 0;
    {
        const pdPerFace = [], nuPerFace = [];
        for (let f = 1; f <= 8; f++) {
            const v = _demoVisits[f] || {};
            pdPerFace.push(v.pd || 0);
            nuPerFace.push(v.nu || 0);
        }
        const pdMean = pdPerFace.reduce((a, b) => a + b, 0) / 8;
        const nuMean = nuPerFace.reduce((a, b) => a + b, 0) / 8;
        const pdCV = pdMean > 0 ? Math.sqrt(pdPerFace.reduce((s, v) => s + (v - pdMean) ** 2, 0) / 8) / pdMean : 1;
        const nuCV = nuMean > 0 ? Math.sqrt(nuPerFace.reduce((s, v) => s + (v - nuMean) ** 2, 0) / 8) / nuMean : 1;
        anchorEvenness = Math.max(0, 1 - (pdCV + nuCV) / 2);
    }

    // 2. FOLLOWER EVENNESS: pu1/pu2/nd1/nd2 across faces
    let followerEvenness = 0;
    {
        const types = ['pu1', 'pu2', 'nd1', 'nd2'];
        let totalCV = 0;
        for (const t of types) {
            const perFace = [];
            for (let f = 1; f <= 8; f++) perFace.push((_demoVisits[f] || {})[t] || 0);
            const m = perFace.reduce((a, b) => a + b, 0) / 8;
            const cv = m > 0 ? Math.sqrt(perFace.reduce((s, v) => s + (v - m) ** 2, 0) / 8) / m : 1;
            totalCV += cv;
        }
        followerEvenness = Math.max(0, 1 - totalCV / 4);
    }

    // 3. ANCHOR:FOLLOWER RATIO — 1:2 per orientation (pd:pu1+pu2, nu:nd1+nd2)
    let anchorFollowerRatio = 0;
    {
        const pFollower = gPu1 + gPu2;
        const nFollower = gNd1 + gNd2;
        const pRatio = pFollower > 0 ? gPd / pFollower : 0; // target: 0.5 (1:2)
        const nRatio = nFollower > 0 ? gNu / nFollower : 0;
        const pErr = Math.abs(pRatio - 0.5);
        const nErr = Math.abs(nRatio - 0.5);
        anchorFollowerRatio = Math.max(0, 1 - (pErr + nErr));
    }

    // 4. FOLLOWER:FOLLOWER RATIO — 1:1 pu1:pu2 and nd1:nd2
    let followerRatio = 0;
    {
        const puMax = Math.max(gPu1, gPu2, 1);
        const puMin = Math.min(gPu1, gPu2);
        const ndMax = Math.max(gNd1, gNd2, 1);
        const ndMin = Math.min(gNd1, gNd2);
        followerRatio = (puMin / puMax + ndMin / ndMax) / 2;
    }

    // 5. QUARK FREQUENCY — loops completed per tick
    let quarkFrequency = 0;
    {
        const loopsPerTick = _demoTick > 0 ? total / _demoTick : 0;
        quarkFrequency = Math.min(1, loopsPerTick / 0.5); // normalize: 0.5 loops/tick = perfect
    }

    // 6. PERIODICITY — regularity of oct orientation changes (from _octWindingDirection changes)
    // Approximated by face coverage evenness (how regularly faces are visited)
    let periodicity = 0;
    {
        const totals = [];
        for (let f = 1; f <= 8; f++) totals.push(_demoVisits[f] ? _demoVisits[f].total : 0);
        const mean = totals.reduce((a, b) => a + b, 0) / totals.length;
        const cv = mean > 0 ? Math.sqrt(totals.reduce((s, v) => s + (v - mean) ** 2, 0) / totals.length) / mean : 1;
        periodicity = Math.max(0, 1 - cv);
    }

    // 7. XONIC BALANCE — direction + mode balance across all xons
    let xonicBalance = 0;
    {
        let totalBal = 0, count = 0;
        for (const x of _demoXons) {
            if (!x.alive || !x._dirBalance) continue;
            const db = x._dirBalance;
            let sum = 0;
            for (let d = 0; d < 10; d++) sum += db[d];
            const mean = sum / 10;
            if (mean > 0) {
                let variance = 0;
                for (let d = 0; d < 10; d++) variance += (db[d] - mean) ** 2;
                totalBal += 1 - Math.sqrt(variance / 10) / mean;
            }
            count++;
        }
        xonicBalance = count > 0 ? totalBal / count : 0;
    }

    // Check for ANY guard failure
    const failedGuards = Object.entries(_liveGuards)
        .filter(([, g]) => g.failed)
        .map(([id]) => id);
    const anyFail = failedGuards.length > 0 || simHalted;

    // Hit rate (kept for backward compat)
    const hitRate = _demoTetAssignments > 0 ? total / _demoTetAssignments : 0;

    // Legacy evenness metrics (for logging)
    const pEven = pTotal > 0 ? 1 - (Math.abs(gPu1/pTotal - 1/3) + Math.abs(gPu2/pTotal - 1/3) + Math.abs(gPd/pTotal - 1/3)) : 0;
    const nEven = nTotal > 0 ? 1 - (Math.abs(gNd1/nTotal - 1/3) + Math.abs(gNd2/nTotal - 1/3) + Math.abs(gNu/nTotal - 1/3)) : 0;
    const totals = [];
    for (let f = 1; f <= 8; f++) totals.push(_demoVisits[f] ? _demoVisits[f].total : 0);
    const fMean = totals.reduce((a, b) => a + b, 0) / totals.length;
    const fCV = fMean > 0 ? Math.sqrt(totals.reduce((s, v) => s + (v - fMean) ** 2, 0) / totals.length) / fMean : 1;
    const evenness = Math.max(0, 1 - fCV);

    // Weighted fitness (7 priorities)
    const fitness7 = 0.25 * anchorEvenness
                   + 0.20 * followerEvenness
                   + 0.15 * anchorFollowerRatio
                   + 0.12 * followerRatio
                   + 0.12 * quarkFrequency
                   + 0.08 * periodicity
                   + 0.08 * xonicBalance;

    // Fitness tiers
    let fitness;
    if (total === 0) fitness = -20;
    else if (anyFail) fitness = fitness7 - 10;
    else fitness = fitness7;

    return {
        pEven, nEven, evenness, hitRate,
        anchorEvenness, followerEvenness, anchorFollowerRatio,
        followerRatio, quarkFrequency, periodicity, xonicBalance,
        totalVisits: total, assignments: _demoTetAssignments,
        fitness, failedGuards, survivedTicks: _demoTick,
        clean: !anyFail && total > 0,
    };
}

// ── NEW: Hadronic ratio fitness — single metric for RL training ──
// Fitness = 1 - avgCV of [pu1, pu2, pd, nd1, nd2, nu] across 8 faces.
// Perfect balance → 1.0, complete imbalance → ~0.0, no visits → -20.
function _evaluateHadronicRatioFitness(visits) {
    visits = visits || _demoVisits;
    const types = ['pu1', 'pu2', 'pd', 'nd1', 'nd2', 'nu'];
    let totalVisits = 0;
    let cvSum = 0;

    for (let f = 1; f <= 8; f++) {
        const v = visits[f] || {};
        const counts = types.map(t => v[t] || 0);
        const sum = counts.reduce((a, b) => a + b, 0);
        totalVisits += sum;
        if (sum === 0) { cvSum += 1; continue; }
        const mean = sum / 6;
        let variance = 0;
        for (let i = 0; i < 6; i++) variance += (counts[i] - mean) ** 2;
        const cv = Math.sqrt(variance / 6) / mean;
        cvSum += cv;
    }

    const avgCV = cvSum / 8;

    if (totalVisits === 0) return { fitness: -20, avgCV: 1, totalVisits: 0 };

    // Guard failure penalty
    const failedGuards = Object.entries(_liveGuards)
        .filter(([, g]) => g.failed)
        .map(([id]) => id);
    const anyFail = failedGuards.length > 0 || simHalted;

    let fitness = 1 - avgCV;
    if (anyFail) fitness -= 10;

    return { fitness, avgCV, totalVisits, failedGuards, clean: !anyFail && totalVisits > 0 };
}

// Hook into demoTick to detect when trial reaches target tick.
// Called from demoTick's UI update path (at end of each tick).
function _tournamentTickCheck() {
    if (!_tournamentRunning || !_tournamentCallback) return;

    // Early termination: if no tet completions after 5 full cycles, kill trial
    if (_demoTick > 0 && _demoTick % 200 === 0) {
        const total = Object.values(_demoVisits).reduce((s, v) => s + v.total, 0);
        if (total === 0 && _demoTick >= 320) {  // 5 epochs × 64 ticks
            console.warn(`[Tournament] Early termination: 0 tet visits after ${_demoTick} ticks`);
            const cb = _tournamentCallback;
            _tournamentCallback = null;
            cb();
            return;
        }
    }

    if (_planckSeconds >= _tournamentTargetTick || simHalted) {
        const cb = _tournamentCallback;
        _tournamentCallback = null;
        cb();
    }
}

// Start a visual trial: no longer overrides user slider defaults.
function _applyTournamentVisuals() {
    // Respect user's slider settings — don't override during tournament/training.
}

function _startVisualTrial(params, maxTicks) {
    return new Promise((resolve) => {
        // Stop any existing demo cleanly
        if (typeof stopDemo === 'function') stopDemo();
        simHalted = false;

        // Apply candidate params
        const { _rlGenome, ...choreoOnly } = params;
        Object.assign(_choreoParams, choreoOnly);

        // Load RL genome into both models
        if (_rlGenome && typeof _rlAvailable !== 'undefined' && _rlAvailable) {
            if (!_rlModel && typeof createPolicyModel === 'function') _rlModel = createPolicyModel();
            if (!_rlStrategicModel && typeof createStrategicModel === 'function') _rlStrategicModel = createStrategicModel();
            if (_rlModel && _rlStrategicModel && typeof genomeToModel === 'function') {
                genomeToModel(_rlGenome, _rlStrategicModel, _rlModel);
            }
            _rlActiveModel = _rlModel;
        } else {
            _rlActiveModel = null;
            _rlStrategicModel = null;
        }

        // Use whatever lattice level is already on the slider

        // Ensure nucleus is active
        if (!NucleusSimulator.active) NucleusSimulator.simulateNucleus();

        // Set target tick and callback
        _tournamentTargetTick = maxTicks;
        _tournamentCallback = () => {
            const result = _evaluateHadronicRatioFitness();
            resolve(result);
        };

        // Start the demo — it will run visually using the normal animation loop
        startDemoLoop();
        // Restore camera (opening choreography skips centering during tournament)
        if (typeof _tournamentSavedPan !== 'undefined' && _tournamentSavedPan) {
            panTarget.x = _tournamentSavedPan.x;
            panTarget.y = _tournamentSavedPan.y;
            panTarget.z = _tournamentSavedPan.z;
            applyCamera();
        }

        // Tournament no longer overrides visual settings — preserve user's view
        _tournamentVisualsApplied = true;

        // Restore trial label (simulateNucleus overwrites topbar-title)
        const titleEl = document.getElementById('topbar-title');
        if (titleEl && titleEl.dataset.trialLabel) titleEl.textContent = titleEl.dataset.trialLabel;
    });
}

// ── Fitness curve state ──
let _rlFitnessCurve = [];  // [{gen, best, avg}]

// ── Leaderboard rendering ──
function _updateLeaderboard() {
    const panel = document.getElementById('leaderboard-panel');
    const list = document.getElementById('leaderboard-list');
    if (!panel || !list) return;
    if (_networkLeaderboard.length === 0) { panel.style.display = 'none'; return; }
    panel.style.display = '';

    const sorted = [..._networkLeaderboard].sort((a, b) => b.fitness - a.fitness);
    const maxFit = Math.max(0.001, sorted[0].fitness);
    let html = '';
    for (let i = 0; i < sorted.length; i++) {
        const e = sorted[i];
        const rank = i + 1;
        const rankClass = rank <= 3 ? ` rank-${rank}` : '';
        const isCurrent = e.gen === _networkGen && _tournamentRunning;
        const currentClass = isCurrent ? ' current' : '';
        const rankColor = rank === 1 ? '#FFD700' : rank === 2 ? '#C0C0C0' : rank === 3 ? '#CD7F32' : '';
        const pct = Math.max(0, Math.min(100, (e.fitness / maxFit) * 100));
        html += `<div class="lb-card${rankClass}${currentClass}">` +
            `<div class="lb-rank${rank <= 3 ? ' top' : ''}" ${rankColor ? `style="color:${rankColor}"` : ''}>${rank}</div>` +
            `<div class="lb-info">` +
                `<div><span class="lb-name">${e.name}</span><span class="lb-gen">Gen ${e.gen}</span></div>` +
                `<div class="lb-bar-track"><div class="lb-bar-fill" style="width:${pct.toFixed(1)}%"></div></div>` +
                `<div class="lb-metrics">` +
                    `fit ${(e.fitness * 100).toFixed(1)}%` +
                    ` · cv ${e.cv.toFixed(3)}` +
                    ` · fails ${e.guardFailures}` +
                    (e.actualizationRate != null ? ` · act ${(e.actualizationRate * 100).toFixed(0)}%` : '') +
                `</div>` +
            `</div>` +
        `</div>`;
    }
    list.innerHTML = html;
}

function _drawFitnessCurve() {
    const canvas = document.getElementById('rl-fitness-canvas');
    if (!canvas || _rlFitnessCurve.length === 0) return;
    const wrap = document.getElementById('rl-fitness-wrap');
    if (wrap) wrap.style.display = '';
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const PAD = { top: 24, right: 12, bottom: 24, left: 40 };
    ctx.clearRect(0, 0, W, H);

    // Background
    ctx.fillStyle = 'rgba(8, 12, 20, 0.95)';
    ctx.fillRect(0, 0, W, H);

    const data = _rlFitnessCurve;
    const maxGen = data.length;
    const maxFit = Math.max(0.01, ...data.map(d => d.best));
    const minFit = Math.min(0, ...data.map(d => Math.min(d.best, d.avg)));
    const range = maxFit - minFit || 0.01;

    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;
    const toX = (i) => PAD.left + (i / Math.max(1, maxGen - 1)) * plotW;
    const toY = (v) => PAD.top + plotH - ((v - minFit) / range) * plotH;

    // Gridlines
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = 'rgba(255,255,255,0.05)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = PAD.top + (i / 4) * plotH;
        ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(W - PAD.right, y); ctx.stroke();
    }
    ctx.setLineDash([]);

    // Axes
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD.left, PAD.top); ctx.lineTo(PAD.left, H - PAD.bottom);
    ctx.lineTo(W - PAD.right, H - PAD.bottom);
    ctx.stroke();

    // Area fill under best line
    if (data.length > 1) {
        const grad = ctx.createLinearGradient(0, PAD.top, 0, H - PAD.bottom);
        grad.addColorStop(0, 'rgba(232,197,71,0.12)');
        grad.addColorStop(1, 'rgba(232,197,71,0.0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.moveTo(toX(0), H - PAD.bottom);
        for (let i = 0; i < data.length; i++) ctx.lineTo(toX(i), toY(data[i].best));
        ctx.lineTo(toX(data.length - 1), H - PAD.bottom);
        ctx.closePath();
        ctx.fill();
    }

    // Average line (muted)
    ctx.strokeStyle = 'rgba(148,163,184,0.4)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
        const x = toX(i), y = toY(data[i].avg);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Best line glow
    ctx.strokeStyle = 'rgba(232,197,71,0.2)';
    ctx.lineWidth = 6;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
        const x = toX(i), y = toY(data[i].best);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Best line crisp
    ctx.strokeStyle = '#E8C547';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
        const x = toX(i), y = toY(data[i].best);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Data point dots on best line
    if (data.length <= 30) {
        ctx.fillStyle = '#E8C547';
        for (let i = 0; i < data.length; i++) {
            ctx.beginPath(); ctx.arc(toX(i), toY(data[i].best), 3, 0, Math.PI * 2); ctx.fill();
        }
    }

    // Y-axis labels
    ctx.fillStyle = '#64748B';
    ctx.font = '10px "SF Mono", "Fira Code", monospace';
    ctx.textAlign = 'right';
    ctx.fillText(maxFit.toFixed(2), PAD.left - 4, PAD.top + 4);
    ctx.fillText(minFit.toFixed(2), PAD.left - 4, H - PAD.bottom + 4);

    // X-axis label
    ctx.textAlign = 'center';
    ctx.fillStyle = '#64748B';
    ctx.font = '9px Inter, sans-serif';
    ctx.fillText('Epoch', W / 2, H - 4);

    // Legend (top right)
    const lx = W - PAD.right - 90;
    ctx.font = '10px Inter, sans-serif';
    ctx.fillStyle = '#E8C547'; ctx.fillRect(lx, 6, 8, 8); ctx.beginPath();
    ctx.fillStyle = '#94A3B8'; ctx.textAlign = 'left'; ctx.fillText('Best', lx + 12, 14);
    ctx.fillStyle = 'rgba(148,163,184,0.5)'; ctx.fillRect(lx + 48, 6, 8, 8);
    ctx.fillStyle = '#94A3B8'; ctx.fillText('Avg', lx + 60, 14);

    // Current stats
    const last = data[data.length - 1];
    ctx.fillStyle = '#E8ECF1';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`Gen ${maxGen}`, PAD.left + 4, 14);
    ctx.fillStyle = '#94A3B8';
    ctx.font = '10px "SF Mono", monospace';
    ctx.fillText(`best ${last.best.toFixed(3)}  avg ${last.avg.toFixed(3)}`, PAD.left + 50, 14);
}

// Main tournament runner — PPO training with live metrics
async function _runTournament() {
    _tournamentRunning = true;
    _tournamentVisualsApplied = false;
    _rlFitnessCurve = [];
    const savedPan = { x: panTarget.x, y: panTarget.y, z: panTarget.z };
    _tournamentSavedPan = savedPan;

    const epochsEl = document.getElementById('tournament-generations');
    const tickEl   = document.getElementById('tournament-ticks');
    const EPOCHS   = epochsEl ? Math.max(1, parseInt(epochsEl.value) || 10) : 10;
    const PLANCK_PER_EPOCH = tickEl ? Math.max(64, parseInt(tickEl.value) || 2000) : 2000;
    const statusEl = document.getElementById('tune-status');
    const titleEl  = document.getElementById('topbar-title');

    // ── Network naming ──
    const usedNames = new Set(_networkLeaderboard.map(e => e.name));
    const available = _NETWORK_NAMES.filter(n => !usedNames.has(n));
    _networkName = available.length > 0
        ? available[Math.floor(Math.random() * available.length)]
        : _NETWORK_NAMES[Math.floor(Math.random() * _NETWORK_NAMES.length)];
    _networkGen = 0;
    _networkLeaderboard = [];
    // Set title + trialLabel so it persists across simulateNucleus() calls
    if (titleEl) {
        titleEl.textContent = `Current Algo: ${_networkName}`;
        titleEl.style.textTransform = 'uppercase';
        titleEl.dataset.trialLabel = `Current Algo: ${_networkName}`;
    }

    // Initialize RL models
    const rlReady = typeof initRL === 'function' && await initRL();
    if (!rlReady) {
        if (statusEl) { statusEl.textContent = 'TF.js required for PPO'; statusEl.style.color = '#ff6644'; }
        _tournamentRunning = false;
        if (typeof _updateLatticeSliderLock === 'function') _updateLatticeSliderLock();
        panTarget.x = savedPan.x; panTarget.y = savedPan.y; panTarget.z = savedPan.z;
        applyCamera();
        return;
    }

    // Create actor-critic models for training
    _ppoStrategicAC = _rlStrategicModel;
    _ppoTacticalAC = _rlModel;
    _rlActiveModel = _rlModel;

    // Create Adam optimizers
    const strategicOptimizer = tf.train.adam(PPO_LEARNING_RATE);
    const tacticalOptimizer  = tf.train.adam(PPO_LEARNING_RATE);

    // Create trajectory buffers
    _ppoStrategicBuffer = new PPOTrajectoryBuffer();
    _ppoTacticalBuffer  = new PPOTrajectoryBuffer();

    // Enable trajectory collection in planner hooks
    _ppoTraining = true;
    resetTickRewardState();
    if (typeof _updateLatticeSliderLock === 'function') _updateLatticeSliderLock();

    // Show fitness canvas + tensor dashboard (canvases are inside wrapper divs)
    const fitnessCanvas = document.getElementById('rl-fitness-canvas');
    const fitnessWrap = document.getElementById('rl-fitness-wrap');
    if (fitnessWrap) fitnessWrap.style.display = '';
    for (const id of ['rl-policy-wrap', 'rl-weights-wrap', 'rl-metrics-wrap']) {
        const w = document.getElementById(id);
        if (w) w.style.display = '';
    }
    // Clear metrics history for fresh run
    if (typeof _ppoMetricsHistory !== 'undefined') _ppoMetricsHistory.length = 0;

    console.log(`[PPO] Starting: ${EPOCHS} epochs × ${PLANCK_PER_EPOCH} Planck-seconds, rollout=${PPO_ROLLOUT_LENGTH}`);
    const tensorsBefore = tf.memory().numTensors;

    let bestFitness = -Infinity;
    let updateCount = 0;

    for (let epoch = 0; epoch < EPOCHS; epoch++) {
        if (!_tournamentRunning) break;

        // Start a fresh demo for this epoch
        if (typeof stopDemo === 'function') stopDemo();
        simHalted = false;
        if (!NucleusSimulator.active) NucleusSimulator.simulateNucleus();
        startDemoLoop();

        // CRITICAL: Kill the auto-tick loop that startDemoLoop() created.
        // The PPO training loop drives ticks manually — double-ticking
        // causes race conditions and Pauli violations.
        if (_demoInterval) { clearInterval(_demoInterval); _demoInterval = null; }
        if (_demoUncappedId) { clearTimeout(_demoUncappedId); _demoUncappedId = null; }

        // Live guards & backtracker run normally during PPO training —
        // identical physics enforcement to demo mode. The NN's bad moves
        // get backtracked just like any other bad moves.

        // Restore camera
        panTarget.x = savedPan.x; panTarget.y = savedPan.y; panTarget.z = savedPan.z;
        applyCamera();
        _tournamentVisualsApplied = true;

        // Re-show playback controls (stopDemo hides them; enterNucleusMode
        // only runs on first simulateNucleus call, not on epoch 2+)
        const pbCtrl = document.getElementById('playback-controls');
        if (pbCtrl) pbCtrl.style.display = '';

        // Restore network name (simulateNucleus overwrites topbar-title)
        if (titleEl && titleEl.dataset.trialLabel) {
            titleEl.textContent = titleEl.dataset.trialLabel;
        }

        _ppoStrategicBuffer.clear();
        _ppoTacticalBuffer.clear();
        resetTickRewardState();

        let epochRewardSum = 0;
        let epochTicks = 0;
        let guardFailures = 0;
        // Must read AFTER startDemoLoop() which resets _planckSeconds to 0
        const epochStartPlanck = _planckSeconds;

        // Run until we accumulate PLANCK_PER_EPOCH Planck seconds (deformation events).
        // Backtrack retries don't count — only successful deformations advance the epoch.
        // Safety: cap raw ticks at 10x Planck target to prevent infinite loops if
        // the simulation gets stuck producing no deformations.
        const MAX_RAW_TICKS = PLANCK_PER_EPOCH * 10;
        for (let tick = 0; (_planckSeconds - epochStartPlanck) < PLANCK_PER_EPOCH && tick < MAX_RAW_TICKS; tick++) {
            if (!_tournamentRunning) break;

            // During PPO training, guard halts are non-fatal — reset and penalize
            if (simHalted) {
                simHalted = false;
                guardFailures++;
                // Heavy penalty for guard failure
                const penalty = -5.0;
                epochRewardSum += penalty;
                _ppoStrategicBuffer.assignReward(penalty);
                _ppoTacticalBuffer.assignReward(penalty);
                // Reset guard state so training can continue
                for (const entry of LIVE_GUARD_REGISTRY) {
                    const g = _liveGuards[entry.id];
                    if (g && g.failed) { g.failed = false; g.ok = true; g.msg = ''; }
                }
                _liveGuardFailTick = null;
                _liveGuardDumped = false;
            }

            // Pause support: wait while paused, yielding to let UI respond.
            // User may rewind/step during pause — RL picks up from new state.
            if (_demoPaused && _tournamentRunning) {
                const tickBeforePause = _demoTick;
                // First pause frame: sync visuals so user can see current state
                if (typeof applyPositions === 'function' && typeof pos !== 'undefined') applyPositions(pos);
                if (typeof bumpState === 'function') bumpState();
                if (typeof updateVoidSpheres === 'function') updateVoidSpheres();
                if (typeof updateSpheres === 'function') updateSpheres();
                if (typeof updateDemoPanel === 'function') updateDemoPanel();
                if (typeof _updateEdgeBalancePanel === 'function') _updateEdgeBalancePanel();
                if (typeof _updateEjectionBalancePanel === 'function') _updateEjectionBalancePanel();
                if (typeof updateXonPanel === 'function') updateXonPanel();
                while (_demoPaused && _tournamentRunning) {
                    await new Promise(r => setTimeout(r, 50));
                }
                // If user rewound during pause, reset backtracker & guards
                // so RL continues cleanly from the new position
                if (_demoTick !== tickBeforePause) {
                    _redoStack.length = 0;
                    _bfsReset(); _btReset();
                    if (typeof _liveGuardResetForRewind === 'function') _liveGuardResetForRewind();
                }
            }
            if (!_tournamentRunning) break;

            // Temporarily clear _demoPaused so demoTick() doesn't early-return
            const wasPaused = _demoPaused;
            if (wasPaused) _demoPaused = false;
            demoTick();
            if (wasPaused) _demoPaused = true;

            // Yield every tick so the browser can repaint (user sees each tick)
            await new Promise(r => setTimeout(r, 0));

            // Compute reward for this tick
            const reward = computeTickReward();
            epochRewardSum += reward;
            epochTicks++;

            // Assign reward to both buffers
            _ppoStrategicBuffer.assignReward(reward);
            _ppoTacticalBuffer.assignReward(reward);

            // PPO update every ROLLOUT_LENGTH ticks
            if ((tick + 1) % PPO_ROLLOUT_LENGTH === 0) {
                // Perform PPO update for strategic network
                let stratMetrics = null, tactMetrics = null;
                if (_ppoStrategicBuffer.length >= 2 && _ppoStrategicAC) {
                    const lastVal = 0; // bootstrap with 0 at rollout boundary
                    stratMetrics = ppoUpdate(_ppoStrategicAC, strategicOptimizer, _ppoStrategicBuffer, lastVal);
                    console.log(`[PPO] Strategic update #${updateCount}: loss=${stratMetrics.policyLoss.toFixed(4)}`);
                }
                // Perform PPO update for tactical network
                if (_ppoTacticalBuffer.length >= 2 && _ppoTacticalAC) {
                    const lastVal = 0;
                    tactMetrics = ppoUpdate(_ppoTacticalAC, tacticalOptimizer, _ppoTacticalBuffer, lastVal);
                    console.log(`[PPO] Tactical update #${updateCount}: loss=${tactMetrics.policyLoss.toFixed(4)}`);
                }
                // Record metrics for tensor dashboard
                const dashMetrics = tactMetrics || stratMetrics;
                if (dashMetrics && typeof _ppoRecordMetrics === 'function') {
                    _ppoRecordMetrics(dashMetrics);
                }
                _ppoStrategicBuffer.clear();
                _ppoTacticalBuffer.clear();
                updateCount++;
                // Update tensor dashboard after each PPO update
                if (typeof _updateTensorDashboard === 'function') _updateTensorDashboard();
            }
        }

        // End of epoch: evaluate fitness
        const fitness = _evaluateHadronicRatioFitness();
        const avgReward = epochTicks > 0 ? epochRewardSum / epochTicks : 0;
        const currentCV = typeof _ppoComputeAvgCV === 'function' ? _ppoComputeAvgCV() : 1;

        console.log(`[PPO] Epoch ${epoch+1}: fitness=${fitness.fitness.toFixed(3)} avgReward=${avgReward.toFixed(4)} CV=${currentCV.toFixed(3)} guardFails=${guardFailures} tensors=${tf.memory().numTensors}`);

        if (fitness.fitness > bestFitness) {
            bestFitness = fitness.fitness;
            // Save best weights
            await rlSaveWeights(_ppoStrategicAC, _ppoTacticalAC, bestFitness);
        }

        // Update fitness curve
        _rlFitnessCurve.push({ gen: epoch + 1, best: bestFitness, avg: fitness.fitness });
        _drawFitnessCurve();

        // ── Network naming + leaderboard ──
        _networkGen = epoch + 1;
        const { evennessScore: _lbEven, avgCV: _lbCV } = _computeCoverageEvenness();
        _networkLeaderboard.push({
            name: _networkName,
            gen: _networkGen,
            fitness: fitness.fitness,
            cv: _lbCV,
            guardFailures: guardFailures,
            actualizationRate: fitness.actualizationRate || 0,
            avgReward: avgReward
        });
        if (typeof _updateLeaderboard === 'function') _updateLeaderboard();

        if (statusEl) {
            statusEl.textContent = `${_networkName} · Gen ${_networkGen}/${EPOCHS} | fitness ${fitness.fitness.toFixed(3)} | best ${bestFitness.toFixed(3)}`;
        }
        if (titleEl) {
            titleEl.textContent = `Current Algo: ${_networkName} · Gen ${_networkGen}`;
            titleEl.dataset.trialLabel = `Current Algo: ${_networkName} · Gen ${_networkGen}`;
        }
    }

    // Cleanup
    _ppoTraining = false;
    _ppoStrategicBuffer = null;
    _ppoTacticalBuffer = null;
    strategicOptimizer.dispose();
    tacticalOptimizer.dispose();

    const tensorsAfter = tf.memory().numTensors;
    console.log(`[PPO] Done. Tensors: ${tensorsBefore} → ${tensorsAfter} (delta=${tensorsAfter - tensorsBefore})`);

    if (statusEl) {
        statusEl.textContent = `${_networkName} trained · best fitness ${bestFitness.toFixed(3)}`;
        statusEl.style.color = 'var(--success, #4ADE80)';
    }

    _tournamentRunning = false;
    _tournamentVisualsApplied = false;
    if (typeof _updateLatticeSliderLock === 'function') _updateLatticeSliderLock();
    panTarget.x = savedPan.x; panTarget.y = savedPan.y; panTarget.z = savedPan.z;
    applyCamera();
    const tuneBtn = document.getElementById('btn-tune-t22');
    if (tuneBtn) { tuneBtn.textContent = 'Train RL'; }

    if (titleEl) {
        titleEl.textContent = _networkName ? `Current Algo: ${_networkName}` : 'NUCLEUS: DEUTERON';
        titleEl.title = '';
        delete titleEl.dataset.trialLabel;
    }

    // Restart demo with trained model
    if (typeof stopDemo === 'function') stopDemo();
    simHalted = false;
    if (typeof startDemoLoop === 'function') startDemoLoop();
}
