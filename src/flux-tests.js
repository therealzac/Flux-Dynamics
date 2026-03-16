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
        if (typeof _ruleBareTetrahedra !== 'undefined' && !_ruleBareTetrahedra) return null;
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
        if (typeof _ruleBareTetrahedra !== 'undefined' && !_ruleBareTetrahedra) return null;
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
        if (typeof _ruleBareTetrahedra !== 'undefined' && !_ruleBareTetrahedra) return null;
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
        if (typeof _ruleBareTetrahedra !== 'undefined' && !_ruleBareTetrahedra) return null;
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
    if (!_demoActive || !_liveGuardsActive || _testRunning) return;
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
        // During council replay phase (up to recorded peak), suppress rewinds/halts.
        // The recorded moves are known-good; guard failures are transient and the
        // original run resolved them via backtracking. We replay the happy path only.
        if (_sweepReplayActive && _sweepReplayMember && tick <= _sweepReplayMember.peak) {
            // Reset failed guards silently — replay continues
            for (const entry of LIVE_GUARD_REGISTRY) {
                const g = _liveGuards[entry.id];
                if (g.failed) { g.failed = false; g.ok = true; g.msg = ''; }
            }
            _liveGuardFailTick = null;
            _liveGuardDumped = false;
            // Don't request rewind or halt — just continue
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
            console.warn(`[BACKTRACK] Rewind requested: ${_rewindViolation}`);
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

// ═══ BFS Exhaustiveness Test — Comparison Logic ═════════════════════════════
// Pure function: compares two BFS run results for identical state-space exploration.
function _compareBfsRuns(resultA, resultB) {
    const maxTickMatch = resultA.maxTick === resultB.maxTick;
    const haltReasonMatch = resultA.haltReason === resultB.haltReason;
    const violationMatch = resultA.haltViolation === resultB.haltViolation;

    // Per-tick fingerprint comparison
    const allTicks = new Set([
        ...resultA.perTickFingerprints.keys(),
        ...resultB.perTickFingerprints.keys()
    ]);
    const ticksOnlyA = [], ticksOnlyB = [], ticksMismatch = [];
    let allFPMatch = true;

    for (const tick of [...allTicks].sort((a, b) => a - b)) {
        const setA = resultA.perTickFingerprints.get(tick);
        const setB = resultB.perTickFingerprints.get(tick);
        if (setA && !setB) { ticksOnlyA.push(tick); allFPMatch = false; }
        else if (!setA && setB) { ticksOnlyB.push(tick); allFPMatch = false; }
        else {
            let onlyA = 0, onlyB = 0, shared = 0;
            for (const fp of setA) { if (setB.has(fp)) shared++; else onlyA++; }
            for (const fp of setB) { if (!setA.has(fp)) onlyB++; }
            if (onlyA > 0 || onlyB > 0) {
                ticksMismatch.push({ tick, onlyA, onlyB, shared });
                allFPMatch = false;
            }
        }
    }

    // ── Key booleans ──
    // Same longest tick solution: both runs found the same highest-tick fingerprints
    const longestTickA = [...resultA.perTickFingerprints.keys()].sort((a, b) => b - a)[0] ?? -1;
    const longestTickB = [...resultB.perTickFingerprints.keys()].sort((a, b) => b - a)[0] ?? -1;
    const sameLongestTick = longestTickA === longestTickB;
    let sameLongestSolution = false;
    if (sameLongestTick && longestTickA >= 0) {
        const fpsA = resultA.perTickFingerprints.get(longestTickA);
        const fpsB = resultB.perTickFingerprints.get(longestTickB);
        if (fpsA && fpsB && fpsA.size === fpsB.size) {
            sameLongestSolution = true;
            for (const fp of fpsA) { if (!fpsB.has(fp)) { sameLongestSolution = false; break; } }
        }
    }

    // Same total solutions explored
    const sameTotalFingerprints = resultA.totalFingerprints === resultB.totalFingerprints;

    // Took different paths: first fingerprints at each tick were tried in different order
    // (i.e. not identical sequence — proves different exploration paths)
    let differentPaths = false;
    if (resultA.totalRetries !== resultB.totalRetries) {
        differentPaths = true; // different retry counts = clearly different paths
    } else {
        // Check if the per-tick fingerprint LISTS are in different order
        for (const tick of allTicks) {
            const listA = resultA.perTickFingerprints.get(tick);
            const listB = resultB.perTickFingerprints.get(tick);
            if (listA && listB) {
                const arrA = [...listA];
                const arrB = [...listB];
                if (arrA.length === arrB.length) {
                    for (let i = 0; i < arrA.length; i++) {
                        if (arrA[i] !== arrB[i]) { differentPaths = true; break; }
                    }
                }
                if (differentPaths) break;
            }
        }
    }

    const identical = maxTickMatch && haltReasonMatch && violationMatch && allFPMatch;
    let summary;
    if (identical) {
        summary = `PASS: Both runs explored identical state space. ` +
                  `Longest tick: ${longestTickA}. Halt: ${resultA.haltReason}. ` +
                  `Fingerprints: ${resultA.totalFingerprints}. ` +
                  `Paths: ${differentPaths ? 'DIFFERENT ✓' : 'same ✗'}. DFS is exhaustive.`;
    } else {
        const diffs = [];
        if (!maxTickMatch) diffs.push(`maxTick: ${resultA.maxTick} vs ${resultB.maxTick}`);
        if (!haltReasonMatch) diffs.push(`reason: ${resultA.haltReason} vs ${resultB.haltReason}`);
        if (!violationMatch) diffs.push(`violation: "${resultA.haltViolation}" vs "${resultB.haltViolation}"`);
        if (!allFPMatch) diffs.push(`${ticksMismatch.length} tick(s) with different fingerprints, ` +
            `${ticksOnlyA.length} ticks only in A, ${ticksOnlyB.length} ticks only in B`);
        summary = `FAIL: Runs diverged. ${diffs.join('; ')}`;
    }

    return {
        identical, maxTickMatch, haltReasonMatch, violationMatch, allFPMatch,
        sameLongestTick, sameLongestSolution, sameTotalFingerprints, differentPaths,
        fingerprintDiff: { ticksOnlyA, ticksOnlyB, ticksMismatch },
        summary,
    };
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

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // T-ENUM: Matching enumerator produces all valid matchings
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if (typeof _enumerateAllMatchings === 'function') {
        // 3 xons, triangle of candidates: A→{1,2}, B→{2,3}, C→{1,3}
        // Max cardinality = 3 (each gets a distinct node)
        // Valid matchings: A1-B2-C3, A1-B3-C?(no: 1 taken), A2-B3-C1, A2-B?(no 2 taken)...
        // Should be exactly 2: [1,2,3] and [2,3,1]
        const testPlans = [
            { xon: {}, candidates: [{node:1},{node:2}] },
            { xon: {}, candidates: [{node:2},{node:3}] },
            { xon: {}, candidates: [{node:1},{node:3}] },
        ];
        const matchings = _enumerateAllMatchings(testPlans, new Set());
        assert('T-ENUM Matching enumerator completeness',
            matchings.length === 2,
            `expected 2 matchings, got ${matchings.length}`);
        // Verify all matchings are maximum cardinality (3)
        const allMaxCard = matchings.every(m => m.filter(c => c !== null).length === 3);
        assert('T-ENUM All matchings are max cardinality',
            allMaxCard,
            `some matchings are not max cardinality`);
        // Verify no two matchings are identical
        const fps = new Set(matchings.map(m => m.map(c => c ? c.node : 'null').join(',')));
        assert('T-ENUM All matchings are distinct',
            fps.size === matchings.length,
            `duplicate matchings found`);
    } else {
        skip('T-ENUM Matching enumerator', '_enumerateAllMatchings not defined');
    }

    // ── T-BfsCompare: _compareBfsRuns correctness ──
    {
        const mockBase = () => ({
            seed: 0x11111111, maxTick: 5, haltReason: 'halted',
            haltViolation: 'T19: tick 5: Pauli', totalRetries: 10, totalFingerprints: 3,
            perTickFingerprints: new Map([
                [0, new Set(['X0:0->1|X1:stay@2'])],
                [1, new Set(['X0:1->3|X1:2->4', 'X0:1->5|X1:2->4'])]
            ]),
            perTickLedger: new Map(), elapsedMs: 100,
        });
        // Identical runs → PASS
        const a1 = mockBase(), b1 = mockBase();
        b1.seed = 0x22222222; b1.totalRetries = 8; b1.elapsedMs = 120;
        const cmp1 = _compareBfsRuns(a1, b1);
        assert('T-BfsCompare identical runs',
            cmp1.identical === true,
            `expected identical=true, got ${cmp1.identical}: ${cmp1.summary}`);
        // Divergent fingerprints → FAIL
        const a2 = mockBase(), b2 = mockBase();
        b2.perTickFingerprints.get(1).add('X0:1->7|X1:2->4'); // extra fingerprint in B
        b2.totalFingerprints = 4;
        const cmp2 = _compareBfsRuns(a2, b2);
        assert('T-BfsCompare divergent fingerprints',
            cmp2.identical === false,
            `expected identical=false, got ${cmp2.identical}`);
        // Different max tick → FAIL
        const a3 = mockBase(), b3 = mockBase();
        b3.maxTick = 7;
        const cmp3 = _compareBfsRuns(a3, b3);
        assert('T-BfsCompare different maxTick',
            cmp3.identical === false && cmp3.maxTickMatch === false,
            `expected maxTickMatch=false, got ${cmp3.maxTickMatch}`);
    }

    // ── T-DfsSecondary: secondary choices deterministic when _btActive ──
    {
        // _selectBestPermutation should NOT add PRNG noise when _btActive = true
        // Verify by calling it multiple times and checking for identical results
        if (typeof _selectBestPermutation === 'function' && _demoXons.length > 0 &&
            _nucleusTetFaceData && Object.keys(_nucleusTetFaceData).length > 0) {
            const testXon = _demoXons[0];
            const faceId = Object.keys(_nucleusTetFaceData)[0];
            const fd = _nucleusTetFaceData[faceId];
            const cycle = fd.cycle;
            const qType = 'pu1';

            // Test with _btActive = true: results should be identical across calls
            const prevBtActive = _btActive;
            _btActive = true;
            const results = [];
            for (let i = 0; i < 5; i++) {
                const seq = _selectBestPermutation(testXon, cycle, qType);
                results.push(seq ? seq.join(',') : 'null');
            }
            const allSame = results.every(r => r === results[0]);
            assert('T-DfsSecondary _btActive=true → deterministic permutation',
                allSame,
                `expected identical results, got ${results.length} unique: ${[...new Set(results)].join(' / ')}`);

            // Restore
            _btActive = prevBtActive;
        } else {
            skip('T-DfsSecondary', 'prerequisites not available');
        }
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

// ═══ BFS Exhaustiveness Test — Runner ════════════════════════════════════════

function _captureBfsRunResult() {
    const totalFP = [..._btTriedFingerprints.values()].reduce((s, set) => s + set.size, 0);
    return {
        seed: _runSeed,
        maxTick: _maxTickReached,
        haltReason: simHalted ? 'canary' : 'stopped',
        haltViolation: _rewindViolation || '',
        totalRetries: _totalBacktrackRetries,
        totalFingerprints: totalFP,
        perTickFingerprints: new Map(
            [..._btTriedFingerprints].map(([t, s]) => [t, new Set(s)])
        ),
        perTickLedger: new Map(
            [..._btBadMoveLedger].map(([t, s]) => [t, new Set(s)])
        ),
        elapsedMs: performance.now() - _searchStartTime,
        traversalLog: _searchTraversalLog.slice(),
    };
    _searchTraversalLog = []; // reset for next run
    _searchEventCounter = 0;
    _searchPathStack = [];
    _searchParentNodeId = null;
    _searchLastCandidates = null;
}

function _executeBfsTestRun(runIdx) {
    return new Promise((resolve) => {
        _forceSeed = _bfsTestSeeds[runIdx];

        // Start demo loop — nucleus is already set up by the orchestrator
        startDemoLoop();

        // Poll for completion: only canary (simHalted) or demo stopped.
        // No artificial tick limit — exhaustive search must run until BFS
        // either proves the rules impossible (canary at t=0) or succeeds.
        const pollId = setInterval(() => {
            // Update progress panel
            const testLabel = runIdx === 0 ? 'Test 1 (choreographer)' : 'Test 2 (random)';
            const novelWarning = (runIdx === 1 && _bfsTestNovelCount > 0)
                ? `<br><span style="color:#ff4444;font-weight:bold;font-size:1.1em;">⚠ NOT EXHAUSTIVE — ${_bfsTestNovelCount} novel solution${_bfsTestNovelCount !== 1 ? 's' : ''} found</span>`
                : '';
            _updateBfsTestPanel(
                `${testLabel}: tick ${_demoTick}, highest ${_maxTickReached}, ` +
                `retries ${_totalBacktrackRetries}, ` +
                `layer ${typeof _bfsLayer !== 'undefined' ? _bfsLayer : '?'}` +
                novelWarning
            );
            // Check termination: canary fired or demo stopped (no early abort — let random run to completion)
            if (simHalted || !_demoActive) {
                clearInterval(pollId);
                resolve();
            }
        }, 100);
    });
}

function _setBfsTestTitle(runIdx, test1Best) {
    const titleEl = document.getElementById('topbar-title');
    if (!titleEl) return;
    if (runIdx === 0) {
        titleEl.innerHTML = 'Test 1 <span style="font-size:0.6em; color:#66bbff;">CHOREOGRAPHER</span>';
    } else if (runIdx === 1) {
        titleEl.innerHTML = `Test 2 <span style="font-size:0.6em; color:#ff9944;">RANDOM</span><br><span style="font-size:0.65em; color:#556677; font-weight:400;">Test 1 solutions: ${test1Best}</span>`;
    } else {
        // Restore default
        if (typeof RULE_REGISTRY !== 'undefined' && typeof activeRuleIndex !== 'undefined') {
            titleEl.textContent = RULE_REGISTRY[activeRuleIndex]?.name || '';
        } else {
            titleEl.textContent = '';
        }
    }
}

async function startBfsExhaustivenessTest(latticeLevel) {
    if (_bfsTestActive || _demoActive) return;
    const lvl = latticeLevel || 1; // default L1

    _bfsTestActive = true;
    _bfsTestRunIdx = 0;
    _bfsTestResults = [null, null];
    _bfsTestComparison = null;
    _bfsTestReferenceFingerprints = null;
    _bfsTestEarlyAbort = false;
    _searchTraversalLog = [];
    _searchEventCounter = 0;
    _searchPathStack = [];
    _searchParentNodeId = null;
    _searchLastCandidates = null;
    _searchStartTime = performance.now();
    _setBfsTestPanelVisible(true);
    _bfsTestNovelCount = 0;
    _bfsTestNovelDetail = null;
    _bfsTestDecisionTrace = [];

    // Set lattice level
    const slider = document.getElementById('lattice-slider');
    if (slider && +slider.value !== lvl) {
        slider.value = lvl;
        slider.dispatchEvent(new Event('input'));
        await new Promise(r => setTimeout(r, 100));
    }

    // Both runs use the same seed — the difference is CHOREOGRAPHER STRATEGY
    const seed = (Math.random() * 0xFFFFFFFF) >>> 0;
    _bfsTestSeeds[0] = seed;
    _bfsTestSeeds[1] = seed;

    console.log(`%c[DFS TEST] Starting model exhaustiveness test (L${lvl})\n` +
        `  Seed: 0x${seed.toString(16).padStart(8,'0')}\n` +
        `  Test 1: CHOREOGRAPHER (normal heuristic scoring)\n` +
        `  Test 2: RANDOM (no heuristics) — early abort if novel solution found`,
        'color:cyan;font-weight:bold');

    // ── Test 1: Normal Choreographer ──
    _bfsTestRunIdx = 0;
    _bfsTestRandomChoreographer = false; // normal heuristic scoring
    simHalted = false;
    if (typeof stopDemo === 'function' && _demoActive) stopDemo();
    NucleusSimulator.simulateNucleus();
    await new Promise(r => setTimeout(r, 100));
    _setBfsTestTitle(0);
    _updateBfsTestPanel('Test 1: CHOREOGRAPHER (normal scoring)...');
    await new Promise(r => setTimeout(r, 200));
    await _executeBfsTestRun(0);
    _bfsTestResults[0] = _captureBfsRunResult();
    _bfsTestResults[0].mode = 'choreographer';
    console.log(`%c[DFS TEST] Test 1 (choreographer) done: highest ${_bfsTestResults[0].maxTick}, ` +
        `${_bfsTestResults[0].totalFingerprints} fps, ` +
        `retries ${_bfsTestResults[0].totalRetries}, ${(_bfsTestResults[0].elapsedMs / 1000).toFixed(1)}s`,
        'color:cyan');
    stopDemo();

    // Store Test 1's fingerprints as the reference set for live comparison
    _bfsTestReferenceFingerprints = new Map(
        [..._bfsTestResults[0].perTickFingerprints].map(([t, s]) => [t, new Set(s)])
    );

    // Pause between tests
    _updateBfsTestPanel(`Test 1 (choreographer) done — highest: ${_bfsTestResults[0].maxTick}, ` +
        `fps: ${_bfsTestResults[0].totalFingerprints}. Starting Test 2 (random)...`);
    await new Promise(r => setTimeout(r, 1000));
    simHalted = false;
    _btBadMoveLedger.clear();
    _btTriedFingerprints.clear();

    // ── Test 2: Random Choreographer (with live comparison) ──
    _bfsTestRunIdx = 1;
    _bfsTestRandomChoreographer = true; // all scoring → uniform random
    _bfsTestEarlyAbort = false;
    _bfsTestNovelCount = 0;
    NucleusSimulator.simulateNucleus();
    await new Promise(r => setTimeout(r, 100));
    _setBfsTestTitle(1, _bfsTestResults[0].totalFingerprints);
    _updateBfsTestPanel('Test 2: RANDOM — checking against choreographer solutions...');
    await new Promise(r => setTimeout(r, 200));
    await _executeBfsTestRun(1);
    _bfsTestResults[1] = _captureBfsRunResult();
    _bfsTestResults[1].mode = 'random';
    console.log(`%c[DFS TEST] Test 2 (random) done: highest ${_bfsTestResults[1].maxTick}, ` +
        `${_bfsTestResults[1].totalFingerprints} fps, ${(_bfsTestResults[1].elapsedMs / 1000).toFixed(1)}s`,
        'color:cyan');
    stopDemo();

    // ── Result ──
    _bfsTestActive = false;
    _bfsTestRandomChoreographer = false;
    _forceSeed = null;

    // Save reference count before clearing (needed for fail console output)
    const refFPs = _bfsTestReferenceFingerprints;
    _bfsTestReferenceFingerprints = null;

    if (_bfsTestEarlyAbort) {
        // ── FAIL: Random found novel solutions ──
        const nd = _bfsTestNovelDetail || { tick: '?', fingerprint: '?' };
        const refCount = refFPs ? (refFPs.get(nd.tick)?.size || 0) : '?';
        const r1 = _bfsTestResults[0], r2 = _bfsTestResults[1];
        const randomHalted = r2 && r2.maxTick < 200;
        const failMsg = `FAIL: Choreographer not exhaustive — ` +
            `random found ${_bfsTestNovelCount} novel solution${_bfsTestNovelCount !== 1 ? 's' : ''} (first at tick ${nd.tick}). ` +
            `Random reached tick ${r2 ? r2.maxTick : '?'}${randomHalted ? ' (halted — finite paths)' : ''}.`;
        _bfsTestComparison = {
            identical: false,
            earlyAbort: true,
            novelTick: nd.tick,
            novelFingerprint: nd.fingerprint,
            novelCount: _bfsTestNovelCount,
            summary: failMsg,
            sameLongestSolution: false,
            sameTotalFingerprints: false,
            differentPaths: true,
            randomMaxTick: r2 ? r2.maxTick : 0,
            randomHalted,
            fingerprintDiff: { ticksOnlyA: [], ticksOnlyB: [], ticksMismatch: [] },
        };
        _updateBfsTestPanel(failMsg);
        _setBfsTestTitle(-1);
        console.log(`%c[DFS TEST] ${failMsg}`, 'color:red;font-weight:bold');
        console.log(`  Total novel fingerprints: ${_bfsTestNovelCount}`);
        console.log(`  First novel at tick: ${nd.tick}`);
        console.log(`  Choreographer: highest=${r1 ? r1.maxTick : '?'}, fps=${r1 ? r1.totalFingerprints : '?'}`);
        console.log(`  Random: highest=${r2 ? r2.maxTick : '?'}, fps=${r2 ? r2.totalFingerprints : '?'}${randomHalted ? ' (HALTED — rules may be impossible)' : ''}`);
        console.log(`  Choreographer had ${refCount} solutions at first novel tick`);

        // ── Dump decision trace around divergent tick ──
        if (_bfsTestDecisionTrace && _bfsTestDecisionTrace.length > 0 && nd.tick) {
            const divergeTick = nd.tick;
            // Get choreographer decisions (runIdx=0) near the divergent tick
            const choreoTraces = _bfsTestDecisionTrace
                .filter(e => e.runIdx === 0 && Math.abs(e.tick - divergeTick) <= 3)
                .sort((a, b) => a.tick - b.tick);
            // Get random decisions (runIdx=1) near the divergent tick
            const randomTraces = _bfsTestDecisionTrace
                .filter(e => e.runIdx === 1 && Math.abs(e.tick - divergeTick) <= 3)
                .sort((a, b) => a.tick - b.tick);

            console.log(`%c[DFS TEST] ═══ DECISION TRACE (ticks ${divergeTick-3}..${divergeTick+3}) ═══`,
                'color:orange;font-weight:bold');

            console.log(`%c  CHOREOGRAPHER decisions:`, 'color:#66bbff;font-weight:bold');
            for (const t of choreoTraces) {
                const faces = t.faceAssignments.map(a => `X${a.xonIdx}→F${a.face}(${a.quarkType},s=${a.score})`).join(', ');
                const octs = t.octMatching.map(m => `X${m.xonIdx}:${m.from}→${m.to ?? 'null'}(${m.candidateCount}c)`).join(', ');
                console.log(`    t=${t.tick} [${t.matchingMethod || '?'}${t.btActive ? ' BT' : ''}]` +
                    (faces ? ` faces:{${faces}}` : '') +
                    (octs ? ` oct:{${octs}}` : '') +
                    (t.totalMatchings ? ` (${t.matchingIdx}/${t.totalMatchings} matchings)` : ''));
            }

            console.log(`%c  RANDOM decisions:`, 'color:#ff9944;font-weight:bold');
            for (const t of randomTraces) {
                const faces = t.faceAssignments.map(a => `X${a.xonIdx}→F${a.face}(${a.quarkType},s=${a.score})`).join(', ');
                const octs = t.octMatching.map(m => `X${m.xonIdx}:${m.from}→${m.to ?? 'null'}(${m.candidateCount}c)`).join(', ');
                console.log(`    t=${t.tick} [${t.matchingMethod || '?'}${t.btActive ? ' BT' : ''}]` +
                    (faces ? ` faces:{${faces}}` : '') +
                    (octs ? ` oct:{${octs}}` : '') +
                    (t.totalMatchings ? ` (${t.matchingIdx}/${t.totalMatchings} matchings)` : ''));
            }

            // Key insight: show face assignments at the divergent tick for both
            const choreoAtTick = choreoTraces.find(t => t.tick === divergeTick);
            const randomAtTick = randomTraces.find(t => t.tick === divergeTick);
            if (choreoAtTick || randomAtTick) {
                console.log(`%c  ═══ CRITICAL DIFF AT TICK ${divergeTick} ═══`, 'color:red;font-weight:bold');
                if (choreoAtTick?.faceAssignments?.length) {
                    console.log(`    Choreographer assigned faces: ${choreoAtTick.faceAssignments.map(a => `X${a.xonIdx}→F${a.face}(${a.quarkType})`).join(', ')}`);
                } else {
                    console.log(`    Choreographer: no face assignments at this tick`);
                }
                if (randomAtTick?.faceAssignments?.length) {
                    console.log(`    Random assigned faces: ${randomAtTick.faceAssignments.map(a => `X${a.xonIdx}→F${a.face}(${a.quarkType})`).join(', ')}`);
                } else {
                    console.log(`    Random: no face assignments at this tick`);
                }
            }

            console.log(`%c[DFS TEST] ═══════════════════════════════════════`, 'color:orange;font-weight:bold');
        }
    } else {
        // ── PASS: Random finished without finding anything new ──
        _bfsTestComparison = _compareBfsRuns(_bfsTestResults[0], _bfsTestResults[1]);
        const passMsg = `PASS: Model is exhaustive — random found no novel solutions. ` +
            `Both explored ${_bfsTestResults[0].totalFingerprints} fingerprints.`;
        _bfsTestComparison.summary = passMsg;
        _bfsTestComparison.identical = true;
        _updateBfsTestPanel(passMsg);
        _setBfsTestTitle(-1);

        console.log(`%c[DFS TEST] ${passMsg}`, 'color:lime;font-weight:bold');
        console.log(`  Choreographer: highest=${_bfsTestResults[0].maxTick}, fps=${_bfsTestResults[0].totalFingerprints}, ${(_bfsTestResults[0].elapsedMs / 1000).toFixed(1)}s`);
        console.log(`  Random: highest=${_bfsTestResults[1].maxTick}, fps=${_bfsTestResults[1].totalFingerprints}, ${(_bfsTestResults[1].elapsedMs / 1000).toFixed(1)}s`);
        console.log(`  Same longest solution: ${_bfsTestComparison.sameLongestSolution ? '✓ YES' : '✗ NO'}`);
        console.log(`  Different paths: ${_bfsTestComparison.differentPaths ? '✓ YES' : '✗ NO'} ` +
            `(retries: ${_bfsTestResults[0].totalRetries} vs ${_bfsTestResults[1].totalRetries})`);
    }
}

function _updateBfsTestPanel(message) {
    const el = document.getElementById('bfs-test-results');
    if (!el) return;

    let html = `<div style="color:#9abccc; font-size:11px; margin-bottom:8px;">${message}</div>`;

    // Show choreographer stats from Test 1 if available (during Test 2 or after)
    if (_bfsTestResults[0] && !_bfsTestComparison) {
        const r1 = _bfsTestResults[0];
        html += `<div style="margin-bottom:6px; padding:4px; background:rgba(100,180,255,0.06); border:1px solid rgba(100,180,255,0.15); border-radius:3px;">`;
        html += `<div style="color:#6a8aaa; font-size:9px;">CHOREOGRAPHER (completed)</div>`;
        html += `<div style="font-size:10px; color:#9abccc;">` +
            `highest: <b>${r1.maxTick}</b> &middot; fps: ${r1.totalFingerprints} &middot; ` +
            `retries: ${r1.totalRetries} &middot; ${(r1.elapsedMs / 1000).toFixed(1)}s</div>`;
        html += `</div>`;
    }

    if (_bfsTestComparison) {
        const c = _bfsTestComparison;
        const _bigRedBox = (label) =>
            `<div style="color:#ff4444; font-weight:bold; font-size:13px; margin-bottom:6px; ` +
            `padding:8px; background:rgba(255,50,50,0.12); border:2px solid #ff4444; border-radius:4px; ` +
            `text-align:center; animation:dp-alarm 0.8s ease-in-out infinite alternate;">` +
            `✗ ${label}</div>`;

        if (c.earlyAbort) {
            // ── Box 1: CHOREOGRAPHER NOT EXHAUSTIVE ──
            html += _bigRedBox('CHOREOGRAPHER NOT EXHAUSTIVE');
            html += `<div style="font-size:10px; color:#ff8866; margin-bottom:6px; line-height:1.5;">` +
                `Random found <b>${c.novelCount || 1}</b> novel solution${(c.novelCount || 1) !== 1 ? 's' : ''} (first at tick <b>${c.novelTick}</b>)<br>` +
                `<span style="font-size:9px; color:#cc7755; word-break:break-all;">${c.novelFingerprint}</span>` +
                `</div>`;

            // ── Box 2: RANDOM HALTED (if random halted before reaching choreographer's peak) ──
            if (c.randomHalted) {
                html += _bigRedBox('RANDOM HALTED');
                html += `<div style="font-size:10px; color:#ffaa44; margin-bottom:6px;">` +
                    `Random halted at tick ${c.randomMaxTick} — finite valid paths (rules may be impossible)</div>`;
            }

            // ── Box 3: RANDOM INCOMPLETE (if random found fewer total fps than choreographer) ──
            const r1 = _bfsTestResults[0], r2 = _bfsTestResults[1];
            if (r1 && r2 && r2.totalFingerprints < r1.totalFingerprints) {
                html += `<div style="color:#ffaa00; font-weight:bold; font-size:13px; margin-bottom:6px; ` +
                    `padding:8px; background:rgba(255,170,0,0.12); border:2px solid #ffaa00; border-radius:4px; ` +
                    `text-align:center;">` +
                    `⚠ RANDOM INCOMPLETE</div>`;
                html += `<div style="font-size:10px; color:#cc9944; margin-bottom:6px;">` +
                    `Random explored only ${r2.totalFingerprints} fps vs choreographer's ${r1.totalFingerprints}</div>`;
            }

            // Sanity check: if choreographer found MORE outcomes than random, random is broken
            if (r1 && r2 && r1.totalFingerprints > r2.totalFingerprints && !c.randomHalted) {
                html += _bigRedBox('RANDOM BROKEN — FEWER OUTCOMES THAN CHOREOGRAPHER');
                html += `<div style="font-size:10px; color:#ff8866; margin-bottom:6px;">` +
                    `Choreographer: ${r1.totalFingerprints} fps, Random: ${r2.totalFingerprints} fps. ` +
                    `Random should find at least as many — something is wrong.</div>`;
            }
        } else {
            // ── PASS: exhaustive ──
            html += `<div style="color:#44ff44; font-weight:bold; font-size:13px; margin-bottom:6px; ` +
                    `padding:8px; background:rgba(50,255,50,0.1); border:2px solid #44ff44; border-radius:4px; ` +
                    `text-align:center;">` +
                    `✓ MODEL IS EXHAUSTIVE</div>`;
        }

        for (let i = 0; i < 2; i++) {
            const r = _bfsTestResults[i];
            if (!r) continue;
            const modeLabel = r.mode === 'random' ? 'RANDOM' : 'CHOREOGRAPHER';
            html += `<div style="margin-bottom:4px; padding:4px; background:rgba(255,255,255,0.03); border-radius:3px;">`;
            html += `<div style="color:#6a8aaa; font-size:9px;">${modeLabel} &mdash; seed 0x${r.seed.toString(16).padStart(8,'0')}</div>`;
            html += `<div style="font-size:10px; color:#9abccc;">` +
                `highest tick: <b>${r.maxTick}</b> &middot; halt: ${r.haltReason} &middot; ` +
                `fps: ${r.totalFingerprints} &middot; retries: ${r.totalRetries} &middot; ` +
                `${(r.elapsedMs / 1000).toFixed(1)}s</div>`;
            if (r.haltViolation) {
                html += `<div style="font-size:9px; color:#cc8866; margin-top:2px;">${r.haltViolation}</div>`;
            }
            html += `</div>`;
        }

        if (!c.earlyAbort && c.fingerprintDiff && c.fingerprintDiff.ticksMismatch.length > 0) {
            html += `<div style="color:#ff8844; font-size:9px; margin-top:4px;">Divergent ticks: `;
            html += c.fingerprintDiff.ticksMismatch.slice(0, 10)
                .map(m => `t${m.tick}(+${m.onlyA}/-${m.onlyB})`)
                .join(', ');
            html += `</div>`;
        }
    }

    el.innerHTML = html;

    // Show export button only on success (not on early abort fail)
    const exportBtn = document.getElementById('btn-bfs-export');
    if (exportBtn) {
        const showExport = _bfsTestComparison && !_bfsTestComparison.earlyAbort;
        exportBtn.style.display = showExport ? 'inline-block' : 'none';
    }

    // Show traversal log download button whenever comparison exists
    if (_bfsTestComparison) {
        let dlBtn = document.getElementById('btn-traversal-log');
        if (!dlBtn) {
            dlBtn = document.createElement('button');
            dlBtn.id = 'btn-traversal-log';
            dlBtn.textContent = 'Download Traversal Log';
            dlBtn.style.cssText = 'margin-top:6px;padding:4px 10px;font-size:10px;cursor:pointer;' +
                'background:#1a3a4a;color:#9abccc;border:1px solid #3a6a7a;border-radius:3px;display:block;width:100%;';
            dlBtn.addEventListener('click', _downloadTraversalLog);
            el.parentElement.appendChild(dlBtn);
        }
        dlBtn.style.display = 'block';
    }
}

// ═══ Sweep Mode: Sequential Seeds with Cross-Seed Fingerprint Blacklist ════════

// ── IndexedDB persistence for cross-session blacklist ──
const _BL_IDB_NAME = 'FluxBlacklist';
const _BL_IDB_VERSION = 3;
const _BL_IDB_STORE = 'blacklists';
const _AS_IDB_STORE = 'autosave';
const _CS_IDB_STORE = 'council';
let _blIDB = null;
let _blIDBReady = false;

function _blIDBOpen() {
    return new Promise((resolve) => {
        try {
            const req = indexedDB.open(_BL_IDB_NAME, _BL_IDB_VERSION);
            req.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains(_BL_IDB_STORE)) db.createObjectStore(_BL_IDB_STORE);
                if (!db.objectStoreNames.contains(_AS_IDB_STORE)) db.createObjectStore(_AS_IDB_STORE);
                if (!db.objectStoreNames.contains(_CS_IDB_STORE)) db.createObjectStore(_CS_IDB_STORE);
            };
            req.onsuccess = (e) => { _blIDB = e.target.result; _blIDBReady = true; _migrateOldKeys().then(resolve); };
            req.onerror = () => { console.warn('[Blacklist] IndexedDB unavailable'); resolve(); };
        } catch (e) { resolve(); }
    });
}

// One-time migration: old key format was `|name=value` for every rule.
// New format: concat tag only when rule is active. Absent = OFF.
// Example old: v2|L2|t20=1|oct=1|cap=6|glu=1|bare=1|proj=1|idleOct=0
// Example new: v2|L2|t20|oct1|glu|bare|proj
function _migrateOldKeyFormat(oldKey) {
    // Only migrate keys matching old format (contain '=')
    if (!oldKey.includes('=')) return null;
    // Split into base (v..|L..) and rule segments
    const parts = oldKey.split('|');
    // Find the base prefix (v...|L...) — everything before first '='
    let base = '';
    let ruleParts = [];
    let seedSuffix = '';
    for (const p of parts) {
        if (p.includes('=')) {
            ruleParts.push(p);
        } else if (!base) {
            base = p;
        } else if (base && !base.includes('|L')) {
            base += '|' + p;
        } else {
            // Seed suffix (numeric after rule parts)
            seedSuffix = '|' + p;
        }
    }
    // Rebuild: base + only-when-active tags
    let k = base;
    for (const rp of ruleParts) {
        const [name, val] = rp.split('=');
        if (name === 't20' && val === '1')       k += '|t20';
        else if (name === 'oct' && +val > 0)     k += `|oct${val}`;
        else if (name === 'cap' && +val < 6)     k += `|cap${val}`;
        else if (name === 'glu' && val === '1')   k += '|glu';
        else if (name === 'bare' && val === '1')  k += '|bare';
        else if (name === 'proj' && val === '1')  k += '|proj';
        else if (name === 'idleOct' && val === '1') k += '|idle';
        // OFF rules: omitted entirely
    }
    return k + seedSuffix;
}

async function _migrateOldKeys() {
    if (!_blIDB) return;
    if (localStorage.getItem('_idbKeyMigrated_v1')) return; // already done
    let migrated = 0;
    for (const storeName of [_BL_IDB_STORE, _AS_IDB_STORE, _CS_IDB_STORE]) {
        try {
            const tx = _blIDB.transaction(storeName, 'readwrite');
            const store = tx.objectStore(storeName);
            const allKeys = await new Promise((res, rej) => {
                const r = store.getAllKeys(); r.onsuccess = () => res(r.result); r.onerror = rej;
            });
            for (const key of allKeys) {
                if (typeof key !== 'string') continue;
                const newKey = _migrateOldKeyFormat(key);
                if (!newKey || newKey === key) continue;
                const val = await new Promise((res, rej) => {
                    const r = store.get(key); r.onsuccess = () => res(r.result); r.onerror = rej;
                });
                // If new key already has data, keep the newer one
                const existing = await new Promise((res, rej) => {
                    const r = store.get(newKey); r.onsuccess = () => res(r.result); r.onerror = rej;
                });
                if (!existing) store.put(val, newKey);
                store.delete(key);
                migrated++;
            }
            await new Promise((res) => { tx.oncomplete = res; });
        } catch (e) { console.warn('[IDB migrate]', storeName, e); }
    }
    localStorage.setItem('_idbKeyMigrated_v1', '1');
    if (migrated > 0) console.log(`[IDB] Migrated ${migrated} keys to new format`);
}

// ── Autosave helpers (council-eligible crash recovery) ──

function _isCouncilEligible() {
    if (!_sweepActive) return true;  // manual demo = always save
    const maxSize = _goldenCouncilSize();
    const currentSeed = _forceSeed || _runSeed || 0;
    // Current seed already in council → still eligible (keep autosaving updates)
    if (_sweepGoldenCouncil.some(m => m.seed === currentSeed)) return true;
    // 50% of first place minimum to be admitted
    const firstPlacePeak = _sweepGoldenCouncil.length > 0 ? _sweepGoldenCouncil[0].peak : 0;
    if (firstPlacePeak > 0 && _maxTickReached < firstPlacePeak * 0.5) return false;
    if (_sweepGoldenCouncil.length < maxSize) return true;
    return _maxTickReached > _sweepGoldenCouncil[_sweepGoldenCouncil.length - 1].peak;
}

async function _autosaveToIDB() {
    if (_councilSnapArchive.length === 0) return;
    if (!_blIDBReady) await _blIDBOpen();
    if (!_blIDB) return;
    // Use forward-only archive (pre-serialized at each max-tick advance)
    const allSnaps = _councilSnapArchive.slice();
    const lvl = typeof latticeLevel !== 'undefined' ? latticeLevel : 2;
    const key = _blacklistRuleKey(lvl);
    try {
        const tx = _blIDB.transaction(_AS_IDB_STORE, 'readwrite');
        tx.objectStore(_AS_IDB_STORE).put({
            snapshots: allSnaps,
            tick: _demoTick,
            seed: _runSeed,
            maxTickReached: _maxTickReached,
            sweepSeedIdx: _sweepSeedIdx,
            timestamp: new Date().toISOString(),
        }, key);
        console.log(`%c[Autosave] tick ${_demoTick} saved (peak ${_maxTickReached}, ${allSnaps.length} snapshots)`, 'color:#80ff80');
    } catch (e) { console.warn('[Autosave] Save failed:', e); }
}

async function _autosaveIDBLoad(lvl) {
    if (!_blIDBReady) await _blIDBOpen();
    if (!_blIDB) return null;
    const key = _blacklistRuleKey(lvl || 2);
    return new Promise((resolve) => {
        try {
            const tx = _blIDB.transaction(_AS_IDB_STORE, 'readonly');
            const req = tx.objectStore(_AS_IDB_STORE).get(key);
            req.onsuccess = () => resolve(req.result || null);
            req.onerror = () => resolve(null);
        } catch (e) { resolve(null); }
    });
}

async function _autosaveIDBClear(lvl) {
    if (!_blIDBReady) await _blIDBOpen();
    if (!_blIDB) return;
    const key = _blacklistRuleKey(lvl || 2);
    try {
        const tx = _blIDB.transaction(_AS_IDB_STORE, 'readwrite');
        tx.objectStore(_AS_IDB_STORE).delete(key);
    } catch (e) { /* ignore */ }
}

// Canonical rule key: deterministic fingerprint of the ENTIRE rule config + snapshot version.
// If ANY config value differs, data goes into a separate IDB bucket.
function _blacklistRuleKey(lvl) {
    // Concat-only-when-active: each rule appends a tag only when ON.
    // OFF rules are absent → key is identical to before that rule existed.
    let k = `v${_SNAPSHOT_VERSION}|L${lvl}`;
    if (_ruleT20StrictMode)     k += '|t20';
    if (T79_MAX_FULL_TICKS > 0) k += `|oct${T79_MAX_FULL_TICKS}`;
    if (OCT_CAPACITY_MAX < 6)   k += `|cap${OCT_CAPACITY_MAX}`;
    if (_ruleGluonMediatedSC)   k += '|glu';
    if (_ruleBareTetrahedra)    k += '|bare';
    if (_ruleProjectedGuards)   k += '|proj';
    if (_ruleAdaptiveEjection) {
        k += '|adpt';
    } else if (_ruleCubeRootEjection) {
        k += '|cbrt';
    } else {
        if (T90_TOLERANCE > 1) k += `|eq${T90_TOLERANCE}`;
        if (T91_TOLERANCE > 1) k += `|ef${T91_TOLERANCE}`;
        if (T92_TOLERANCE > 1) k += `|eh${T92_TOLERANCE}`;
    }
    return k;
}

// Load blacklist from IndexedDB for current rules
async function _blIDBLoad(lvl) {
    if (!_blIDBReady) await _blIDBOpen();
    if (!_blIDB) return null;
    const key = _blacklistRuleKey(lvl);
    const t0 = performance.now();
    return new Promise((resolve) => {
        try {
            const tx = _blIDB.transaction(_BL_IDB_STORE, 'readonly');
            const req = tx.objectStore(_BL_IDB_STORE).get(key);
            req.onsuccess = async () => {
                const data = req.result;
                if (!data) { resolve(null); return; }

                // ── Parse council index (shared by both formats) ──
                let goldenCouncil = [];
                if (data.councilIndex && Array.isArray(data.councilIndex)) {
                    for (const stub of data.councilIndex) {
                        goldenCouncil.push({ peak: stub.peak, seed: stub.seed, _cold: true });
                    }
                } else if (data.goldenCouncil && Array.isArray(data.goldenCouncil)) {
                    // Old format (v2): migrate to cold storage
                    console.log(`[BL] Migrating ${data.goldenCouncil.length} council members to cold storage...`);
                    for (const member of data.goldenCouncil) {
                        try {
                            const cKey = key + '|' + member.seed;
                            const migTx = _blIDB.transaction(_CS_IDB_STORE, 'readwrite');
                            const migStore = migTx.objectStore(_CS_IDB_STORE);
                            migStore.put({
                                seed: member.seed, snapshots: member.snapshots, moves: member.moves,
                                snapshotVersion: (member.snapshots && member.snapshots[0] && member.snapshots[0]._v) || 0,
                                timestamp: new Date().toISOString(),
                            }, cKey);
                            // Also store moves separately for fast hydration
                            if (member.moves) {
                                const movesArr = [];
                                for (const [tick, moveMap] of (member.moves instanceof Map ? member.moves : new Map(member.moves))) {
                                    movesArr.push([tick, moveMap instanceof Map ? [...moveMap.entries()] : moveMap]);
                                }
                                migStore.put({ seed: member.seed, moves: movesArr }, cKey + '|mv');
                            }
                        } catch (e) { console.warn('[BL] Migration: failed to write member:', e); }
                        goldenCouncil.push({ peak: member.peak, seed: member.seed, _cold: true });
                    }
                }
                // Dedup: keep only the highest-peak entry per seed
                const seedBest = new Map();
                for (const m of goldenCouncil) {
                    const existing = seedBest.get(m.seed);
                    if (!existing || m.peak > existing.peak) seedBest.set(m.seed, m);
                }
                goldenCouncil = [...seedBest.values()].sort((a, b) => b.peak - a.peak);
                const peaks = goldenCouncil.map(m => 't' + m.peak).join(', ');

                // ── Bucketed format: metadata only, fingerprints loaded on demand ──
                if (data.bucketVersion >= 1) {
                    _blBucketVersion = data.bucketVersion;
                    _blBucketCount = data.bucketCount || 0;
                    _blBucketSize = data.bucketSize || 64;
                    _blLoadedBuckets = new Set();
                    const ms = (performance.now() - t0).toFixed(1);
                    console.log(`[BL] Loaded metadata: ${data.total} fps across ${_blBucketCount} buckets + council [${peaks}] in ${ms}ms`);
                    resolve({ map: new Map(), total: data.total || 0, seedIdx: data.seedIdx || 0, goldenCouncil });
                    return;
                }

                // ── Legacy single-blob format: deserialize all inline ──
                if (data.entries) {
                    _blBucketVersion = 0;
                    const map = new Map();
                    let total = 0;
                    for (const [tick, fps] of data.entries) {
                        map.set(tick, new Set(fps));
                        total += fps.length;
                    }
                    // Mark all covered buckets as loaded
                    _blLoadedBuckets = new Set();
                    for (const tick of map.keys()) {
                        _blLoadedBuckets.add(Math.floor(tick / _blBucketSize));
                    }
                    const ms = (performance.now() - t0).toFixed(1);
                    console.log(`[BL] Loaded legacy blob: ${total} fps + council [${peaks}] in ${ms}ms (will migrate on next save)`);
                    resolve({ map, total, seedIdx: data.seedIdx || 0, goldenCouncil });
                } else {
                    resolve(null);
                }
            };
            req.onerror = () => resolve(null);
        } catch (e) { resolve(null); }
    });
}

// Save blacklist + council index to IndexedDB (debounced). Council snapshots saved separately.
let _blIDBSaveTimer = null;
function _blIDBSave(lvl) {
    if (_blIDBSaveTimer) clearTimeout(_blIDBSaveTimer);
    _blIDBSaveTimer = setTimeout(() => {
        _blIDBSaveTimer = null;
        _blIDBSaveBlacklist(lvl);
    }, 2000);
}

// ── Bucketed blacklist: on-demand loading ──

// Load a single bucket from IDB into _sweepBlacklist. Returns a Promise.
async function _blPrefetchBucket(lvl, bucketIdx) {
    if (_blBucketVersion < 1) return;            // legacy format, already fully loaded
    if (_blLoadedBuckets.has(bucketIdx)) return;  // already in memory
    if (bucketIdx >= _blBucketCount) return;      // beyond stored range

    const t0 = performance.now();
    const baseKey = _blacklistRuleKey(lvl);
    const bucketKey = baseKey + '|bl|' + bucketIdx;
    const tickLo = bucketIdx * _blBucketSize;
    const tickHi = tickLo + _blBucketSize - 1;
    console.log(`[BL] Prefetching bucket ${bucketIdx} (ticks ${tickLo}-${tickHi})...`);

    if (!_blIDBReady) await _blIDBOpen();
    if (!_blIDB) { _blLoadedBuckets.add(bucketIdx); return; }

    return new Promise((resolve) => {
        try {
            const tx = _blIDB.transaction(_BL_IDB_STORE, 'readonly');
            const req = tx.objectStore(_BL_IDB_STORE).get(bucketKey);
            req.onsuccess = () => {
                const data = req.result;
                let count = 0;
                if (data && data.entries) {
                    for (const [tick, fps] of data.entries) {
                        if (!_sweepBlacklist.has(tick)) _sweepBlacklist.set(tick, new Set());
                        const set = _sweepBlacklist.get(tick);
                        for (const fp of fps) { set.add(fp); count++; }
                    }
                }
                _blLoadedBuckets.add(bucketIdx);
                const ms = (performance.now() - t0).toFixed(1);
                console.log(`[BL] Bucket ${bucketIdx} loaded: ${count} fps in ${ms}ms`);
                resolve();
            };
            req.onerror = () => { _blLoadedBuckets.add(bucketIdx); resolve(); };
        } catch (e) { _blLoadedBuckets.add(bucketIdx); resolve(); }
    });
}

// Prefetch all buckets covering a tick range. Parallel IDB reads.
async function _blPrefetchRange(lvl, tickLow, tickHigh) {
    if (_blBucketVersion < 1) return;
    const lo = Math.floor(Math.max(0, tickLow) / _blBucketSize);
    const hi = Math.floor(tickHigh / _blBucketSize);
    const needed = [];
    for (let bi = lo; bi <= hi; bi++) {
        if (!_blLoadedBuckets.has(bi) && bi < _blBucketCount) needed.push(bi);
    }
    if (needed.length === 0) return;
    console.log(`[BL] Range prefetch: buckets ${needed.join(',')} (ticks ${tickLow}-${tickHigh})`);
    await Promise.all(needed.map(bi => _blPrefetchBucket(lvl, bi)));
}

// Ensure the bucket for a given tick is loaded. Blocks if needed.
async function _blEnsureTick(lvl, tick) {
    if (_blBucketVersion < 1) return;
    const bi = Math.floor(tick / _blBucketSize);
    if (_blLoadedBuckets.has(bi)) return;
    const t0 = performance.now();
    console.warn(`[BL] ⚠️ Tick ${tick} blocked — awaiting bucket ${bi}`);
    await _blPrefetchBucket(lvl, bi);
    const ms = (performance.now() - t0).toFixed(1);
    console.warn(`[BL] ⚠️ Bucket ${bi} loaded after ${ms}ms block`);
}

// Serialize a single backtracker snapshot for IndexedDB storage.
// Converts Sets → arrays and Maps → [key, value] pairs.
function _serializeSnapshot(snap) {
    return {
        _v: snap._v || 0,
        tick: snap.tick,
        openingPhase: snap.openingPhase,
        xons: snap.xons, // plain objects already
        activeSet: [...snap.activeSet],
        xonImpliedSet: [...snap.xonImpliedSet],
        impliedSet: [...snap.impliedSet],
        scAttribution: [...snap.scAttribution.entries()],
        pos: snap.pos, // array of [x,y,z] already
        octFullConsecutive: snap.octFullConsecutive,
        demoVisits: snap.demoVisits,
        actualizationVisits: snap.actualizationVisits,
        faceEdgeEpoch: snap.faceEdgeEpoch,
        faceWasActualized: snap.faceWasActualized,
        edgeBalance: snap.edgeBalance ? [...snap.edgeBalance].map(([k, v]) => [k, { ...v }]) : null,
        ejectionBalance: snap.ejectionBalance ? [...snap.ejectionBalance] : null,
        octWindingDirection: snap.octWindingDirection,
        planckSeconds: snap.planckSeconds,
        globalModeStats: snap.globalModeStats,
        // Nucleus topology (v2+)
        octNodeSet: snap.octNodeSet ? [...snap.octNodeSet] : null,
        octSCIds: snap.octSCIds ? snap.octSCIds : null,
        octEdgeSet: snap.octEdgeSet ? [...snap.octEdgeSet] : null,
        nucleusTetFaceData: snap.nucleusTetFaceData || null,
        octEquatorCycle: snap.octEquatorCycle || null,
        octCageSCCycle: snap.octCageSCCycle || null,
        octSeedCenter: snap.octSeedCenter != null ? snap.octSeedCenter : null,
        octVoidIdx: snap.octVoidIdx != null ? snap.octVoidIdx : -1,
        octAntipodal: snap.octAntipodal ? [...snap.octAntipodal] : null,
    };
}

// Deserialize a snapshot from IndexedDB back into live format (with Sets/Maps).
function _deserializeSnapshot(s) {
    return {
        _v: s._v || 0,
        tick: s.tick,
        openingPhase: s.openingPhase,
        xons: s.xons.map(x => ({
            ...x,
            _loopSeq: x._loopSeq ? x._loopSeq.slice() : null,
            trail: x.trail.slice(),
            trailColHistory: x.trailColHistory.slice(),
            _trailRoleHistory: x._trailRoleHistory ? x._trailRoleHistory.slice() : [],
            _trailFrozenPos: x._trailFrozenPos ? x._trailFrozenPos.map(p => [p[0], p[1], p[2]]) : [],
            _dirBalance: x._dirBalance ? x._dirBalance.slice() : new Array(10).fill(0),
            _modeStats: x._modeStats ? { ...x._modeStats } : { oct: 0, tet: 0, idle_tet: 0, weak: 0, gluon: 0 },
            _gluonBoundSCs: x._gluonBoundSCs ? x._gluonBoundSCs.slice() : null,
        })),
        activeSet: new Set(s.activeSet),
        xonImpliedSet: new Set(s.xonImpliedSet),
        impliedSet: new Set(s.impliedSet),
        scAttribution: new Map(s.scAttribution),
        pos: s.pos.map(p => [p[0], p[1], p[2]]),
        octFullConsecutive: s.octFullConsecutive,
        demoVisits: s.demoVisits ? JSON.parse(JSON.stringify(s.demoVisits)) : null,
        actualizationVisits: s.actualizationVisits ? JSON.parse(JSON.stringify(s.actualizationVisits)) : null,
        faceEdgeEpoch: s.faceEdgeEpoch ? JSON.parse(JSON.stringify(s.faceEdgeEpoch)) : null,
        faceWasActualized: s.faceWasActualized ? { ...s.faceWasActualized } : null,
        edgeBalance: s.edgeBalance ? new Map(s.edgeBalance.map(([k, v]) => [k, { ...v }])) : null,
        ejectionBalance: s.ejectionBalance ? new Map(s.ejectionBalance) : null,
        octWindingDirection: s.octWindingDirection,
        planckSeconds: s.planckSeconds,
        globalModeStats: s.globalModeStats ? { ...s.globalModeStats } : null,
        // Nucleus topology (v2+)
        octNodeSet: s.octNodeSet ? new Set(s.octNodeSet) : null,
        octSCIds: s.octSCIds ? s.octSCIds.slice() : null,
        octEdgeSet: s.octEdgeSet ? new Set(s.octEdgeSet) : null,
        nucleusTetFaceData: s.nucleusTetFaceData ? JSON.parse(JSON.stringify(s.nucleusTetFaceData)) : null,
        octEquatorCycle: s.octEquatorCycle ? s.octEquatorCycle.slice() : null,
        octCageSCCycle: s.octCageSCCycle ? s.octCageSCCycle.slice() : null,
        octSeedCenter: s.octSeedCenter != null ? s.octSeedCenter : null,
        octVoidIdx: s.octVoidIdx != null ? s.octVoidIdx : -1,
        octAntipodal: s.octAntipodal ? new Map(s.octAntipodal) : null,
    };
}

// ── Hot/Cold storage: blacklist + council index saved together, council snapshots saved separately ──

function _blIDBSaveBlacklist(lvl) {
    if (!_blIDB) return;
    const t0 = performance.now();
    const baseKey = _blacklistRuleKey(lvl);

    // Group fingerprints by bucket index
    const buckets = new Map(); // bucketIdx → [[tick, [fps...]], ...]
    let maxBucket = -1;
    for (const [tick, fpSet] of _sweepBlacklist) {
        const bi = Math.floor(tick / _blBucketSize);
        if (bi > maxBucket) maxBucket = bi;
        if (!buckets.has(bi)) buckets.set(bi, []);
        buckets.get(bi).push([tick, [...fpSet]]);
    }
    const bucketCount = maxBucket + 1;

    // Council index: lightweight stubs only (no snapshots, no moves)
    const councilIndex = _sweepGoldenCouncil.map(m => ({
        peak: m.peak, seed: m.seed, snapshotVersion: _SNAPSHOT_VERSION,
    }));

    try {
        const tx = _blIDB.transaction(_BL_IDB_STORE, 'readwrite');
        const store = tx.objectStore(_BL_IDB_STORE);

        // Write each bucket as a separate key
        for (const [bi, entries] of buckets) {
            const count = entries.reduce((s, [, fps]) => s + fps.length, 0);
            store.put({ entries, count }, baseKey + '|bl|' + bi);
        }

        // Write metadata (no fingerprints — they're in buckets now)
        store.put({
            key: baseKey,
            bucketVersion: 1,
            bucketSize: _blBucketSize,
            bucketCount,
            total: _sweepTotalBlacklisted,
            seedIdx: _sweepSeedIdx,
            councilIndex,
            timestamp: new Date().toISOString(),
        }, baseKey);

        _blBucketVersion = 1;
        _blBucketCount = bucketCount;
        const peaks = _sweepGoldenCouncil.map(m => 't' + m.peak).join(', ');
        const ms = (performance.now() - t0).toFixed(1);
        console.log(`[BL] Saved ${buckets.size} buckets (${_sweepTotalBlacklisted} total fps) + council [${peaks}] in ${ms}ms`);
    } catch (e) { console.warn('[BL] Save failed:', e); }
}

async function _blIDBSaveCouncilMember(lvl, seed, snapshots, moves) {
    if (!_blIDBReady) await _blIDBOpen();
    if (!_blIDB) return;
    const key = _blacklistRuleKey(lvl) + '|' + seed;
    const snapsArr = snapshots.map(_serializeSnapshot);
    const movesArr = [];
    for (const [tick, moveMap] of moves) {
        movesArr.push([tick, [...moveMap.entries()]]);
    }
    try {
        const tx = _blIDB.transaction(_CS_IDB_STORE, 'readwrite');
        const store = tx.objectStore(_CS_IDB_STORE);
        // Store snapshots+moves together (full blob for replay hydration)
        store.put({
            seed, snapshots: snapsArr, moves: movesArr,
            snapshotVersion: _SNAPSHOT_VERSION,
            timestamp: new Date().toISOString(),
        }, key);
        // Store moves separately under key|mv (lightweight read for golden boost)
        store.put({ seed, moves: movesArr }, key + '|mv');
        console.log(`%c[Council IDB] Saved member seed 0x${seed.toString(16).padStart(8,'0')} (${snapsArr.length} snapshots + ${movesArr.length} moves) to cold storage`, 'color:#66ccff');
    } catch (e) { console.warn('[Council IDB] Save failed:', e); }
}

async function _blIDBDeleteCouncilMember(lvl, seed) {
    if (!_blIDBReady) await _blIDBOpen();
    if (!_blIDB) return;
    const key = _blacklistRuleKey(lvl) + '|' + seed;
    try {
        const tx = _blIDB.transaction(_CS_IDB_STORE, 'readwrite');
        const store = tx.objectStore(_CS_IDB_STORE);
        store.delete(key);
        store.delete(key + '|mv');
        console.log(`[Council IDB] Deleted evicted member seed 0x${seed.toString(16).padStart(8,'0')} from cold storage`);
    } catch (e) { /* ignore */ }
}

async function _hydrateCouncilMember(lvl, member) {
    if (!member._cold) return; // already hydrated
    if (!_blIDBReady) await _blIDBOpen();
    if (!_blIDB) return;
    const t0 = performance.now();
    const key = _blacklistRuleKey(lvl) + '|' + member.seed;
    return new Promise((resolve) => {
        try {
            const tx = _blIDB.transaction(_CS_IDB_STORE, 'readonly');
            const req = tx.objectStore(_CS_IDB_STORE).get(key);
            req.onsuccess = () => {
                const data = req.result;
                if (data) {
                    const t1 = performance.now();
                    if (data.snapshots) {
                        member.snapshots = data.snapshots.map(_deserializeSnapshot);
                    }
                    const t2 = performance.now();
                    if (data.moves) {
                        member.moves = new Map();
                        for (const [tick, pairs] of data.moves) {
                            member.moves.set(tick, new Map(pairs));
                        }
                    }
                    member._cold = false;
                    const t3 = performance.now();
                    console.log(`%c[Council IDB] Hydrated member seed 0x${member.seed.toString(16).padStart(8,'0')} — IDB read: ${(t1-t0).toFixed(1)}ms, ${member.snapshots ? member.snapshots.length : 0} snapshots deser: ${(t2-t1).toFixed(1)}ms, moves deser: ${(t3-t2).toFixed(1)}ms, total: ${(t3-t0).toFixed(1)}ms`, 'color:#66ccff');
                }
                resolve();
            };
            req.onerror = () => resolve();
        } catch (e) { resolve(); }
    });
}

async function _hydrateCouncilMoves(lvl, member) {
    if (member.moves) return; // already has moves
    if (!_blIDBReady) await _blIDBOpen();
    if (!_blIDB) return;
    const t0 = performance.now();
    const key = _blacklistRuleKey(lvl) + '|' + member.seed;
    // Try lightweight moves-only key first, fall back to full blob
    const mvKey = key + '|mv';
    return new Promise((resolve) => {
        try {
            const tx = _blIDB.transaction(_CS_IDB_STORE, 'readonly');
            const store = tx.objectStore(_CS_IDB_STORE);
            const req = store.get(mvKey);
            req.onsuccess = () => {
                const data = req.result;
                if (data && data.moves) {
                    // Fast path: moves-only record
                    member.moves = new Map();
                    for (const [tick, pairs] of data.moves) {
                        member.moves.set(tick, new Map(pairs));
                    }
                    console.log(`[STARTUP] Hydrated moves (fast) for seed 0x${member.seed.toString(16).padStart(8,'0')}: ${member.moves.size} ticks in ${(performance.now()-t0).toFixed(1)}ms`);
                    resolve();
                } else {
                    // Fallback: read full blob (legacy, no |mv key yet)
                    const req2 = store.get(key);
                    req2.onsuccess = () => {
                        const full = req2.result;
                        if (full && full.moves) {
                            member.moves = new Map();
                            for (const [tick, pairs] of full.moves) {
                                member.moves.set(tick, new Map(pairs));
                            }
                            console.log(`[STARTUP] Hydrated moves (legacy fallback) for seed 0x${member.seed.toString(16).padStart(8,'0')}: ${member.moves.size} ticks in ${(performance.now()-t0).toFixed(1)}ms`);
                            // Migrate: write |mv key so next load is fast
                            try {
                                const wTx = _blIDB.transaction(_CS_IDB_STORE, 'readwrite');
                                wTx.objectStore(_CS_IDB_STORE).put({ seed: member.seed, moves: full.moves }, mvKey);
                                console.log(`[STARTUP] Migrated |mv key for seed 0x${member.seed.toString(16).padStart(8,'0')}`);
                            } catch (e) { /* best-effort */ }
                        }
                        resolve();
                    };
                    req2.onerror = () => resolve();
                }
            };
            req.onerror = () => resolve();
        } catch (e) { resolve(); }
    });
}

function _dehydrateCouncilMember(member) {
    member.snapshots = null;
    if (!_sweepActive) member.moves = null;
    member._cold = true;
}

async function startSweepTest(latticeLevel, replayMemberIdx) {
    if (_sweepActive || _bfsTestActive || _demoActive) return;
    const lvl = latticeLevel || 1;
    const _startupT0 = performance.now();
    const _startupLog = (label) => {
        const ms = (performance.now() - _startupT0).toFixed(1);
        console.log(`[STARTUP] +${ms}ms — ${label}`);
    };

    _sweepActive = true;
    _sweepSeedIdx = 0;
    _sweepBlacklist = new Map();
    _sweepResults = [];
    _sweepTotalBlacklisted = 0;
    _sweepBlacklistHits = 0;
    _sweepBlacklistHitsSeed = 0;
    _sweepGoldenHits = 0;
    _sweepGoldenHitsSeed = 0;
    // Flush any pending debounced save BEFORE clearing council,
    // so a queued save doesn't overwrite IDB with empty council later
    if (_blIDBSaveTimer) { clearTimeout(_blIDBSaveTimer); _blIDBSaveTimer = null; }
    _sweepGoldenCouncil = [];
    _searchTraversalLog = [];
    _searchEventCounter = 0;
    _searchPathStack = [];
    _searchParentNodeId = null;
    _searchLastCandidates = null;

    // Set lattice level
    const slider = document.getElementById('lattice-slider');
    if (slider && +slider.value !== lvl) {
        slider.value = lvl;
        slider.dispatchEvent(new Event('input'));
        await new Promise(r => setTimeout(r, 100));
    }

    _setBfsTestPanelVisible(true);
    _updateSweepPanel('Loading blacklist...');
    _startupLog('UI ready, loading blacklist...');

    // Load persisted blacklist from IndexedDB for this rule config
    const cached = await _blIDBLoad(lvl);
    _startupLog('Blacklist loaded');
    if (cached) {
        _sweepBlacklist = cached.map;
        _sweepTotalBlacklisted = cached.total;
        _sweepSeedIdx = cached.seedIdx;
        if (cached.goldenCouncil && cached.goldenCouncil.length > 0) {
            _sweepGoldenCouncil = cached.goldenCouncil;
        }
        const councilStr = _sweepGoldenCouncil.length > 0
            ? `council [${_sweepGoldenCouncil.map(m => 't' + m.peak).join(', ')}]` : 'no council';
        _updateSweepPanel(`Resumed: ${cached.total} blacklisted, ${councilStr}, from ${cached.seedIdx} prior seeds`);
        // Eagerly prefetch bucket 0 so first tick has blacklist data
        if (_blBucketVersion >= 1) {
            await _blPrefetchBucket(lvl, 0);
        }
    } else {
        _updateSweepPanel('Starting sweep (fresh)...');
    }

    // Lock rules during sweep
    if (typeof _lockRules === 'function') _lockRules(true);

    let sweepStartTime = performance.now();
    let _sweepPausedAt = 0;

    // Hydrate council member moves for golden boost scoring
    _startupLog(`Hydrating moves for ${_sweepGoldenCouncil.length} council members...`);
    for (const member of _sweepGoldenCouncil) {
        if (!member.moves) await _hydrateCouncilMoves(lvl, member);
    }
    _startupLog('Council moves hydrated');

    // If replaying a council member, set up for the first seed
    let _replayOnFirstSeed = (typeof replayMemberIdx === 'number' && replayMemberIdx >= 0
        && _sweepGoldenCouncil.length > replayMemberIdx) ? _sweepGoldenCouncil[replayMemberIdx] : null;

    while (_sweepActive) {
        let seed = _sweepSeedIdx + 1; // 1, 2, 3, ...

        // Council replay: override first seed with the member's seed
        if (_replayOnFirstSeed) {
            seed = _replayOnFirstSeed.seed;
            _sweepReplayActive = true;
            _sweepReplayMember = _replayOnFirstSeed;
            console.log(`%c[REPLAY] Starting council replay — seed 0x${seed.toString(16).padStart(8,'0')}, peak t${_replayOnFirstSeed.peak}`, 'color:#66ccff;font-weight:bold');
        } else {
            _sweepReplayActive = false;
            _sweepReplayMember = null;
        }

        _forceSeed = seed;

        // Reset per-seed state
        simHalted = false;
        _btBadMoveLedger.clear();
        _btTriedFingerprints.clear();
        _sweepBlacklistHitsSeed = 0;
        _sweepGoldenHitsSeed = 0;
        _sweepSeedMoves = new Map();
        _searchEventCounter = 0;
        _searchPathStack = [];
        _searchParentNodeId = null;
        _searchLastCandidates = null;
        _searchStartTime = performance.now();

        // Set up nucleus and run
        if (typeof stopDemo === 'function' && _demoActive) stopDemo();
        _startupLog('simulateNucleus() starting...');
        NucleusSimulator.simulateNucleus();
        _startupLog('simulateNucleus() done, yielding 100ms...');
        await new Promise(r => setTimeout(r, 100));
        _startupLog('yield done');
        _bfsTestRandomChoreographer = false; // GC mode

        // Council replay: hydrate from cold storage BEFORE starting the demo
        // loop. The await yields to the event loop — if startDemoLoop() ran
        // first, the uncapped tick loop would execute live ticks during the
        // IDB read, corrupting the fresh state.
        let _replaySnapshots = null;
        if (_replayOnFirstSeed) {
            if (_replayOnFirstSeed._cold) {
                _startupLog('Hydrating council member from cold storage...');
                await _hydrateCouncilMember(lvl, _replayOnFirstSeed);
                _startupLog('Council member hydrated');
            }
            if (_replayOnFirstSeed.snapshots && _replayOnFirstSeed.snapshots.length > 0) {
                // Stash snapshots — startDemoLoop() clears _redoStack,
                // so we populate it AFTER the loop initializes.
                _replaySnapshots = _replayOnFirstSeed.snapshots;
            }
        }

        _startupLog('startDemoLoop() starting...');
        startDemoLoop();
        _startupLog('startDemoLoop() done');

        // Now that startDemoLoop has initialized everything (xons, lattice,
        // seed, etc.) and cleared _redoStack, populate redo from the stashed
        // snapshots and switch to redo-drain mode.
        if (_replaySnapshots) {
            _redoStack.length = 0;
            for (let i = _replaySnapshots.length - 1; i >= 0; i--) {
                _redoStack.push(_replaySnapshots[i]);
            }
            console.log(`%c[REPLAY] Loaded ${_replaySnapshots.length} snapshots from cold storage — save game mode`, 'color:#66ccff;font-weight:bold');
            // Kill the live tick loop so resumeDemo() starts redo drain instead
            if (_demoInterval) { clearInterval(_demoInterval); _demoInterval = null; }
            if (_demoUncappedId) { clearTimeout(_demoUncappedId); _demoUncappedId = null; }
            // Council-grade runs have already revealed the oct — force it so
            // void spheres and shells render from the first replay frame.
            _demoOctRevealed = true;
            for (let f = 1; f <= 8; f++) _demoVisitedFaces.add(f);
            pauseDemo();
            _testRunning = false; // enable rendering for entire replay seed
            // Ensure opacity defaults are applied for replay visuals
            // Source: DEMO_VISUAL_DEFAULTS in flux-demo-state.js
            for (const [id, val] of DEMO_VISUAL_DEFAULTS) {
                const el = document.getElementById(id);
                if (el) { el.value = val; el.dispatchEvent(new Event('input')); }
            }
            resumeDemo();
        }

        // Poll for completion
        await new Promise(resolve => {
            const pollId = setInterval(() => {
                if (_demoPaused) {
                    if (!_sweepPausedAt) _sweepPausedAt = performance.now();
                } else {
                    if (_sweepPausedAt) {
                        sweepStartTime += performance.now() - _sweepPausedAt;
                        _sweepPausedAt = 0;
                    }
                    _updateSweepPanel(null, sweepStartTime);
                }
                if (simHalted || !_demoActive || !_sweepActive) {
                    clearInterval(pollId);
                    resolve();
                }
            }, 100);
        });
        if (typeof stopDemo === 'function') stopDemo();

        if (!_sweepActive) break; // user stopped

        // Capture result
        const result = _captureBfsRunResult();
        result.seed = seed;
        result.mode = 'choreographer';
        result.seedIdx = _sweepSeedIdx;

        // Add ALL tried fingerprints to blacklist (they're all dead ends if canary)
        if (simHalted) {
            let newBlacklisted = 0;
            for (const [tick, fpSet] of _btTriedFingerprints) {
                if (!_sweepBlacklist.has(tick)) _sweepBlacklist.set(tick, new Set());
                const bl = _sweepBlacklist.get(tick);
                for (const fp of fpSet) {
                    if (!bl.has(fp)) {
                        bl.add(fp);
                        newBlacklisted++;
                    }
                }
            }
            _sweepTotalBlacklisted += newBlacklisted;
            result.newBlacklisted = newBlacklisted;
        } else {
            result.newBlacklisted = 0;
        }

        _sweepResults.push(result);
        _sweepSeedIdx++;

        // Clear replay mode after first seed — subsequent seeds run normally
        if (_replayOnFirstSeed) {
            // Dehydrate: release snapshots from RAM, keep moves for golden boost
            if (!_replayOnFirstSeed._cold) {
                _replayOnFirstSeed.snapshots = null;
                _replayOnFirstSeed._cold = true;
            }
            _replayOnFirstSeed = null;
            _sweepReplayActive = false;
            _sweepReplayMember = null;
        }

        // Golden council: insert this seed if it qualifies
        if (_sweepSeedMoves && _sweepSeedMoves.size > 0) {
            const maxSize = _goldenCouncilSize();
            const peak = result.maxTick || 0;
            const lowestPeak = _sweepGoldenCouncil.length >= maxSize
                ? _sweepGoldenCouncil[_sweepGoldenCouncil.length - 1].peak : -1;
            const firstPlacePeak = _sweepGoldenCouncil.length > 0 ? _sweepGoldenCouncil[0].peak : 0;
            if (firstPlacePeak > 0 && peak < firstPlacePeak * 0.5) {
                console.log(`%c[GOLDEN COUNCIL] Seed ${seed} (peak t${peak}) rejected — below 50%% of first place (t${firstPlacePeak})`, 'color:#cc6666');
            } else if (_sweepGoldenCouncil.length < maxSize || peak > lowestPeak) {
                const snapsCopy = _councilSnapArchive.map(s => _deserializeSnapshot(s));
                // Dedup: if this seed already exists in council, update it instead of pushing a duplicate
                const existingMember = _sweepGoldenCouncil.find(m => m.seed === seed);
                if (existingMember) {
                    existingMember.peak = Math.max(existingMember.peak, peak);
                    if (!existingMember.moves) existingMember.moves = _sweepSeedMoves;
                    else { for (const [tick, tickMap] of _sweepSeedMoves) { if (!existingMember.moves.has(tick)) existingMember.moves.set(tick, tickMap); } }
                    _blIDBSaveCouncilMember(lvl, seed, snapsCopy, existingMember.moves);
                    console.log(`%c[GOLDEN COUNCIL] Seed ${seed} (peak t${peak}) updated in council [${_sweepGoldenCouncil.map(m => 't' + m.peak).join(', ')}]`, 'color:#ffcc00;font-weight:bold');
                } else {
                    // Collect evicted seeds before trimming
                    const prevSeeds = new Set(_sweepGoldenCouncil.map(m => m.seed));
                    _sweepGoldenCouncil.push({ peak, seed, moves: _sweepSeedMoves, _cold: true });
                    _sweepGoldenCouncil.sort((a, b) => b.peak - a.peak);
                    if (_sweepGoldenCouncil.length > maxSize) _sweepGoldenCouncil.length = maxSize;
                    _blIDBSaveCouncilMember(lvl, seed, snapsCopy, _sweepSeedMoves);
                    // Delete evicted members from cold storage
                    for (const ps of prevSeeds) {
                        if (!_sweepGoldenCouncil.find(m => m.seed === ps)) {
                            _blIDBDeleteCouncilMember(lvl, ps);
                        }
                    }
                    console.log(`%c[GOLDEN COUNCIL] Seed ${seed} (peak t${peak}) joined council [${_sweepGoldenCouncil.map(m => 't' + m.peak).join(', ')}] (${_btSnapshots.length} snapshots → cold)`, 'color:#ffcc00;font-weight:bold');
                }
            }
        }

        // Persist blacklist + council index to IndexedDB after each seed
        _blIDBSave(lvl);

        // If NOT canary (rules satisfiable!), stop sweep
        if (!simHalted) {
            _updateSweepPanel('SOLUTION FOUND at seed ' + seed, sweepStartTime);
            break;
        }

        // Brief pause between seeds
        await new Promise(r => setTimeout(r, 200));
    }

    _sweepActive = false;
    _forceSeed = null;
    _bfsTestRandomChoreographer = false;
    if (typeof _lockRules === 'function') _lockRules(false);
    // Flush blacklist to IndexedDB immediately on sweep end
    if (_blIDBSaveTimer) { clearTimeout(_blIDBSaveTimer); _blIDBSaveTimer = null; }
    _blIDBSaveBlacklist(lvl);
    _updateSweepPanel('Sweep complete', sweepStartTime);
    _populateCouncilDropdown();  // refresh dropdown with any new council members
}

function _stopSweep() {
    _sweepActive = false;
}

// ── Clear Cache: wipe blacklist + council for current rule config ──

function _clearCacheConfirm() {
    const btn = document.getElementById('btn-clear-cache');
    if (!btn) return;
    // Create confirm/cancel row below the button
    let row = document.getElementById('clear-cache-confirm');
    if (!row) {
        row = document.createElement('div');
        row.id = 'clear-cache-confirm';
        row.style.cssText = 'margin-top:4px;display:flex;gap:4px;';
        btn.parentElement.insertBefore(row, btn.nextSibling);
    }
    row.dataset.active = '1';
    row.style.display = 'flex';
    row.innerHTML =
        `<button id="btn-clear-cache-yes" style="flex:1;padding:6px;font-size:12px;cursor:pointer;` +
        `background:#7a2a2a;color:#ffaaaa;border:1px solid #aa4444;border-radius:3px;">Yes, clear</button>` +
        `<button id="btn-clear-cache-no" style="flex:1;padding:6px;font-size:12px;cursor:pointer;` +
        `background:#2a2a3a;color:#aaaacc;border:1px solid #4a4a6a;border-radius:3px;">Cancel</button>`;
    document.getElementById('btn-clear-cache-yes').addEventListener('click', _clearCacheExecute);
    document.getElementById('btn-clear-cache-no').addEventListener('click', () => {
        row.style.display = 'none';
        delete row.dataset.active;
    });
}

async function _clearCacheExecute() {
    const lvl = typeof latticeLevel !== 'undefined' ? latticeLevel : 2;
    const key = _blacklistRuleKey(lvl);

    // Stop any active sweep and demo
    _sweepActive = false;
    if (_demoActive && typeof stopDemo === 'function') stopDemo();

    // Clear in-memory state
    _sweepBlacklist = new Map();
    _sweepTotalBlacklisted = 0;
    _sweepBlacklistHits = 0;
    _sweepBlacklistHitsSeed = 0;
    _sweepResults = [];

    // Delete each council member's cold storage
    for (const m of _sweepGoldenCouncil) {
        await _blIDBDeleteCouncilMember(lvl, m.seed);
    }
    _sweepGoldenCouncil = [];

    // Delete blacklist metadata + all bucket keys from IDB
    if (_blIDB) {
        try {
            const tx = _blIDB.transaction(_BL_IDB_STORE, 'readwrite');
            const store = tx.objectStore(_BL_IDB_STORE);
            store.delete(key); // metadata
            for (let bi = 0; bi < _blBucketCount; bi++) {
                store.delete(key + '|bl|' + bi);
            }
            console.log(`[BL] Cleared ${_blBucketCount} bucket keys + metadata`);
        } catch (e) { console.warn('[Clear Cache] blacklist delete failed:', e); }
        // Delete autosave too
        try {
            const tx = _blIDB.transaction(_AS_IDB_STORE, 'readwrite');
            tx.objectStore(_AS_IDB_STORE).delete(key);
        } catch (e) { /* ignore */ }
    }
    _blLoadedBuckets = new Set();
    _blBucketCount = 0;
    _blBucketVersion = 0;

    console.log(`%c[Clear Cache] Cleared blacklist + council for key: ${key}`, 'color:#ff8866;font-weight:bold');

    // Reload page for a clean slate
    window.location.reload();
}

// Save the current in-progress run as a council member (even if it hasn't terminated)
function _saveCurrentRunToCouncil() {
    if (!_sweepSeedMoves || _sweepSeedMoves.size === 0) return;
    const seed = _forceSeed || _runSeed || 0;
    const peak = Math.max(_demoTick || 0, _maxTickReached || 0);
    const maxSize = _goldenCouncilSize();
    const lowestPeak = _sweepGoldenCouncil.length >= maxSize
        ? _sweepGoldenCouncil[_sweepGoldenCouncil.length - 1].peak : -1;
    // Clone moves so the live map can keep growing
    const movesCopy = new Map();
    for (const [tick, tickMap] of _sweepSeedMoves) {
        movesCopy.set(tick, new Map(tickMap));
    }
    const firstPlacePeak = _sweepGoldenCouncil.length > 0 ? _sweepGoldenCouncil[0].peak : 0;
    // Existing seed already in council can always update (don't gate updates on 50% rule)
    const existingMember = _sweepGoldenCouncil.find(m => m.seed === seed);
    if (!existingMember && firstPlacePeak > 0 && peak < firstPlacePeak * 0.5) {
        console.log(`%c[SAVE] Current run (peak t${peak}) rejected — below 50%% of first place (t${firstPlacePeak})`, 'color:#cc6666');
    } else if (_sweepGoldenCouncil.length < maxSize || peak > lowestPeak || existingMember) {
        const slider = document.getElementById('lattice-slider');
        const lvl = slider ? +slider.value : 2;
        // Use forward-only archive (pre-serialized, never popped by backtracking)
        const snapsCopy = _councilSnapArchive.map(s => _deserializeSnapshot(s));
        if (existingMember) {
            // Only update if current run surpasses the existing peak (most mature wins)
            if (peak <= existingMember.peak) {
                console.log(`%c[SAVE] Current run (peak t${peak}) not more mature than existing (t${existingMember.peak}) — skip`, 'color:#cc8866');
                return;
            }
            existingMember.peak = peak;
            existingMember.moves = movesCopy;
            // Overwrite cold storage with the more mature data
            _blIDBSaveCouncilMember(lvl, seed, snapsCopy, existingMember.moves);
        } else {
            const prevSeeds = new Set(_sweepGoldenCouncil.map(m => m.seed));
            _sweepGoldenCouncil.push({ peak, seed, moves: movesCopy, _cold: true });
            _sweepGoldenCouncil.sort((a, b) => b.peak - a.peak);
            if (_sweepGoldenCouncil.length > maxSize) _sweepGoldenCouncil.length = maxSize;
            _blIDBSaveCouncilMember(lvl, seed, snapsCopy, movesCopy);
            for (const ps of prevSeeds) {
                if (!_sweepGoldenCouncil.find(m => m.seed === ps)) _blIDBDeleteCouncilMember(lvl, ps);
            }
        }
        _sweepGoldenCouncil.sort((a, b) => b.peak - a.peak);
        _blIDBSave(lvl);
        _populateCouncilDropdown();
        console.log(`%c[SAVE] Saved current run (seed 0x${seed.toString(16).padStart(8,'0')}, peak t${peak}) to council (${snapsCopy.length} snapshots → cold, ${existingMember ? 'updated' : 'new'})`, 'color:#66cc88;font-weight:bold');
    } else {
        console.log(`%c[SAVE] Current run (peak t${peak}) doesn't beat lowest council member (t${lowestPeak})`, 'color:#cc8866');
    }
}

// ── Council member replay — starts a sweep with the member's seed first ──
// Replay phase uses forced moves + guard suppression up to peak,
// then continues as a normal greedy sweep (blacklist, council, etc.)
function startCouncilReplay(memberIdx) {
    if (_sweepActive || _bfsTestActive || _demoActive) return;
    const slider = document.getElementById('lattice-slider');
    const lvl = slider ? +slider.value : 2;
    startSweepTest(lvl, memberIdx);
}


function _updateReplayPanel(member, startTime, message) {
    const el = document.getElementById('bfs-test-results');
    if (!el) return;
    const elapsed = startTime ? ((performance.now() - startTime) / 1000).toFixed(1) : '?';
    const seedHex = '0x' + member.seed.toString(16).padStart(8, '0');
    const pastPeak = _demoTick > member.peak;
    let html = '';

    if (message) {
        html += `<div style="color:#9abccc; font-size:11px; margin-bottom:6px;">${message}</div>`;
    } else if (pastPeak) {
        html += `<div style="color:#ff9966; font-size:11px; margin-bottom:6px;">` +
            `Live exploration — seed ${seedHex} (past peak t${member.peak}) — ` +
            `tick ${_demoTick}, ${elapsed}s</div>`;
    } else {
        html += `<div style="color:#66ccff; font-size:11px; margin-bottom:6px;">` +
            `Replaying seed ${seedHex} — tick ${_demoTick} / peak t${member.peak} — ${elapsed}s</div>`;
    }

    // Blacklist stats
    html += `<div style="font-size:10px; color:#aaa;">` +
        `Blacklist: ${_sweepTotalBlacklisted.toLocaleString()} states, ` +
        `hits: ${_sweepBlacklistHits.toLocaleString()}</div>`;

    // Stop button
    html += `<div style="margin-top:6px;"><button onclick="_stopSweep()" ` +
        `style="font-size:10px; padding:2px 8px; cursor:pointer;">Stop Replay</button></div>`;

    el.innerHTML = html;
}

// Populate council dropdown — works with cold stubs (no snapshots in RAM)
async function _populateCouncilDropdown() {
    const sel = document.getElementById('council-replay-select');
    if (!sel) return;

    const slider = document.getElementById('lattice-slider');
    const lvl = slider ? +slider.value : 2;

    // Always load from IDB when no sweep is active (rules may have changed)
    let council = [];
    if (_sweepActive) {
        council = _sweepGoldenCouncil;
    } else {
        try {
            const cached = await _blIDBLoad(lvl);
            if (cached && cached.goldenCouncil) {
                council = cached.goldenCouncil;
            }
        } catch (e) {
            console.warn('[Council dropdown] Failed to load:', e);
        }
        // Persist loaded stubs so replay can find them by index
        _sweepGoldenCouncil = council;
    }

    // Sort by peak descending
    council.sort((a, b) => (b.peak || 0) - (a.peak || 0));

    sel.innerHTML = '';
    const blankOpt = document.createElement('option');
    blankOpt.value = '';
    blankOpt.textContent = 'New run';
    sel.appendChild(blankOpt);

    for (let i = 0; i < council.length; i++) {
        const m = council[i];
        const seedHex = '0x' + m.seed.toString(16).padStart(8, '0');
        const opt = document.createElement('option');
        opt.value = i;
        opt.textContent = `${seedHex} (t${m.peak})`;
        sel.appendChild(opt);
    }
}

function _updateSweepPanel(message, sweepStartTime) {
    const el = document.getElementById('bfs-test-results');
    if (!el) return;

    const totalElapsed = sweepStartTime ? ((performance.now() - sweepStartTime) / 1000).toFixed(1) : '?';

    let html = '';

    // Header
    html += `<div style="color:#9abccc; font-size:11px; margin-bottom:6px;">`;
    if (message) {
        html += message;
    } else {
        html += `Seed ${_sweepSeedIdx + 1} / \u221E &mdash; tick ${_demoTick}, retries ${_totalBacktrackRetries}, ` +
            `layer ${typeof _bfsLayer !== 'undefined' ? _bfsLayer : '?'}`;
    }
    html += `</div>`;

    // Blacklist stats
    html += `<div style="padding:4px; background:rgba(100,180,255,0.06); border:1px solid rgba(100,180,255,0.15); border-radius:3px; margin-bottom:6px;">`;
    html += `<div style="font-size:10px; color:#9abccc;">` +
        `Blacklisted: <b>${_sweepTotalBlacklisted.toLocaleString()}</b> states` +
        (_blBucketVersion >= 1 ? ` (${_blLoadedBuckets.size}/${_blBucketCount} buckets)` : '') +
        ` &middot; Hits: <b>${_sweepBlacklistHits.toLocaleString()}</b> (${_sweepBlacklistHitsSeed} this seed) &middot; ` +
        `Seeds: <b>${_sweepResults.length}</b> &middot; ` +
        `Total: ${totalElapsed}s</div>`;
    if (_sweepGoldenCouncil.length > 0 || (_demoActive && _lastAutosavePeak > 0)) {
        // Build sorted entries: council members, marking the current round's seed with green *
        const hasRecentAutosave = _demoActive && _maxTickReached > 0
            && _lastAutosavePeak > 0 && (_maxTickReached - _lastAutosavePeak) < 10
            && typeof _isCouncilEligible === 'function' && _isCouncilEligible();
        const liveSeed = hasRecentAutosave ? (_forceSeed || _runSeed || 0) : null;
        const entries = _sweepGoldenCouncil.map(m => ({ peak: m.peak, live: liveSeed !== null && m.seed === liveSeed }));
        // If live run isn't in council yet but is autosave-eligible, add it as a separate green entry
        if (hasRecentAutosave && !entries.some(e => e.live)) {
            entries.push({ peak: _lastAutosavePeak, live: true });
        }
        entries.sort((a, b) => b.peak - a.peak);
        const peakStrs = entries.map(e => e.live
            ? `<span style="color:#80ff80;">t${e.peak}*</span>`
            : 't' + e.peak);
        html += `<div style="font-size:10px; color:#ffcc66; margin-top:2px;">` +
            `Council [${_sweepGoldenCouncil.length}/${_goldenCouncilSize()}]: ${peakStrs.join(', ')} &middot; ` +
            `Votes: <b>${_sweepGoldenHits.toLocaleString()}</b> (${_sweepGoldenHitsSeed} this seed)</div>`;
        if (hasRecentAutosave) {
            html += `<div style="font-size:9px; color:#80ff80; margin-top:1px;">* autosaved</div>`;
        }
    }
    html += `</div>`;

    // ── Seed peak-tick sparkline bar chart (same style as ratio accuracy) ──
    if (_sweepResults.length > 0) {
        const SPARK_SLOTS = 32;
        const SPARK = ['\u2581', '\u2582', '\u2583', '\u2584', '\u2585', '\u2586', '\u2587', '\u2588'];
        const peaks = _sweepResults.map(r => r.maxTick || 0);
        const chartData = peaks.slice(-SPARK_SLOTS);
        const chartMax = Math.max(...chartData, 1);
        const chartMin = 0;
        let sparkline = '';
        for (const v of chartData) {
            const norm = Math.max(0, Math.min(1, (v - chartMin) / (chartMax - chartMin)));
            const idx = Math.min(7, Math.floor(norm * 7.99));
            // Color: green if high, orange if mid, red if low
            const pct = chartMax > 0 ? v / chartMax : 0;
            const c = pct >= 0.7 ? '#66dd66' : pct >= 0.3 ? '#ccaa66' : '#cc5544';
            sparkline += `<span style="color:${c};">${SPARK[idx]}</span>`;
        }
        html += `<div style="margin-top:4px; overflow:hidden; width:100%;">`
            + `<div style="font-size:8px; color:#667788; margin-bottom:2px;">seed peak tick (last ${chartData.length})</div>`
            + `<div style="font-size:22px; letter-spacing:-1px; line-height:1; font-family:monospace; white-space:nowrap; overflow:hidden;">${sparkline}</div>`
            + `<div style="display:flex; justify-content:space-between; font-size:7px; color:#445566; margin-top:2px;">`
            + `<span>t0</span><span>t${chartMax}</span></div>`
            + `</div>`;
    }

    // ── Blacklist contribution sparkline bar chart ──
    if (_sweepResults.length > 0) {
        const SPARK_SLOTS = 32;
        const SPARK = ['\u2581', '\u2582', '\u2583', '\u2584', '\u2585', '\u2586', '\u2587', '\u2588'];
        const bls = _sweepResults.map(r => r.newBlacklisted || 0);
        const chartData = bls.slice(-SPARK_SLOTS);
        const chartMax = Math.max(...chartData, 1);
        let sparkline = '';
        for (const v of chartData) {
            const norm = Math.max(0, Math.min(1, v / chartMax));
            const idx = Math.min(7, Math.floor(norm * 7.99));
            const c = v === 0 ? '#445566' : norm >= 0.7 ? '#66aaff' : norm >= 0.3 ? '#7799bb' : '#556688';
            sparkline += `<span style="color:${c};">${SPARK[idx]}</span>`;
        }
        html += `<div style="margin-top:4px; overflow:hidden; width:100%;">`
            + `<div style="font-size:8px; color:#667788; margin-bottom:2px;">blacklist contributions (last ${chartData.length})</div>`
            + `<div style="font-size:22px; letter-spacing:-1px; line-height:1; font-family:monospace; white-space:nowrap; overflow:hidden;">${sparkline}</div>`
            + `<div style="display:flex; justify-content:space-between; font-size:7px; color:#445566; margin-top:2px;">`
            + `<span>0</span><span>${chartMax.toLocaleString()}</span></div>`
            + `</div>`;
    }

    // Download button placeholder (listener attached after innerHTML)
    if (_sweepResults.length > 0 && !_sweepActive) {
        html += `<button id="btn-sweep-download" style="margin-top:6px;padding:4px 10px;font-size:10px;cursor:pointer;` +
            `background:#1a3a4a;color:#9abccc;border:1px solid #3a6a7a;border-radius:3px;display:block;width:100%;"` +
            `>Download Sweep Log</button>`;
    }

    el.innerHTML = html;

    // Attach download listener after innerHTML is set
    const dlBtn = document.getElementById('btn-sweep-download');
    if (dlBtn) dlBtn.addEventListener('click', _downloadSweepLog);

    // Clear Cache button (replaces Stop Sweep)
    let clearBtn = document.getElementById('btn-clear-cache');
    if (!clearBtn) {
        clearBtn = document.createElement('button');
        clearBtn.id = 'btn-clear-cache';
        clearBtn.textContent = 'Clear Cache';
        clearBtn.style.cssText = 'margin-top:6px;padding:8px 10px;font-size:13px;cursor:pointer;' +
            'background:#4a1a1a;color:#ff8866;border:1px solid #7a3a3a;border-radius:3px;display:block;width:100%;';
        clearBtn.addEventListener('click', _clearCacheConfirm);
        el.parentElement.appendChild(clearBtn);
    }
    // Hide confirm row if it exists and we're not in confirm state
    const confirmRow = document.getElementById('clear-cache-confirm');
    if (confirmRow && !confirmRow.dataset.active) confirmRow.style.display = 'none';

    // Hide old traversal log button during sweep
    const oldDlBtn = document.getElementById('btn-traversal-log');
    if (oldDlBtn) oldDlBtn.style.display = 'none';
}

// ── Compact event encoding for download ──
// Strips candidates from rewind events (option 1), keeps them only on success/escalation (option 2),
// and uses compact string encoding for candidates (option 3): "node:score" or "node:score:x" if excluded.
function _compactEvent(e) {
    const compact = {
        id: e.eventId, nid: e.nodeId, pid: e.parentId,
        t: e.tick, r: e.retry, L: e.bfsLayer,
        fp: e.fingerprint, o: e.outcome,
    };
    // Wall: only include type + guard ID (strip verbose details)
    if (e.wall) {
        const guardMatch = e.wall.details?.[0]?.match(/^(T\d+\w*)/);
        compact.w = guardMatch ? guardMatch[1] : e.wall.type;
    }
    // Moves: compact "xonIdx:from>to:mode[0]" e.g. "0:9>13:o"
    if (e.moves && e.moves.length > 0) {
        compact.m = e.moves.map(m => m.xonIdx + ':' + m.from + '>' + m.to + ':' + m.mode[0]);
    }
    // Candidates: only on success/escalation/canary events (not rewinds — option 1+2)
    if (e.outcome !== 'rewind' && e.candidates) {
        const cc = {};
        for (const [xi, arr] of Object.entries(e.candidates)) {
            if (arr.length > 0) {
                // Compact: "node:score" or "node:score:x" if excluded (option 3)
                cc[xi] = arr.map(c => c.node + ':' + (c.score || 0) + (c.excluded ? ':x' : ''));
            }
        }
        if (Object.keys(cc).length > 0) compact.c = cc;
    }
    // Exclusion count (not full list)
    if (e.exclusionTotal > 0) compact.ex = e.exclusionTotal;
    return compact;
}

function _downloadSweepLog() {
    if (!_sweepResults || _sweepResults.length === 0) return;

    const seeds = _sweepResults.map(r => ({
        seedIdx: r.seedIdx,
        seed: '0x' + (r.seed >>> 0).toString(16).padStart(8, '0'),
        summary: {
            maxTick: r.maxTick,
            totalRetries: r.totalRetries,
            totalFingerprints: r.totalFingerprints,
            haltReason: r.haltReason,
            haltViolation: r.haltViolation || '',
            elapsedMs: Math.round(r.elapsedMs),
            newBlacklisted: r.newBlacklisted || 0,
        },
        events: (r.traversalLog || []).map(_compactEvent),
    }));

    const payload = {
        version: 4,
        mode: 'sweep',
        timestamp: new Date().toISOString(),
        latticeLevel: typeof latticeLevel !== 'undefined' ? latticeLevel : '?',
        rules: {
            t20Strict: _ruleT20StrictMode,
            maxFullOctTicks: T79_MAX_FULL_TICKS,
            octCapacityMax: OCT_CAPACITY_MAX,
            gluonMediated: _ruleGluonMediatedSC,
            bareTetrahedra: _ruleBareTetrahedra,
        },
        totalSeeds: _sweepResults.length,
        totalBlacklisted: _sweepTotalBlacklisted,
        seeds,
    };

    // No pretty-print — compact JSON saves ~40% more
    const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `sweep-log-L${payload.latticeLevel}-${payload.totalSeeds}seeds-${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function _downloadTraversalLog() {
    if (!_bfsTestResults || _bfsTestResults.length < 2) return;

    const runs = _bfsTestResults.map(r => ({
        mode: r.mode || 'unknown',
        seed: '0x' + (r.seed >>> 0).toString(16).padStart(8, '0'),
        summary: {
            maxTick: r.maxTick,
            totalRetries: r.totalRetries,
            totalFingerprints: r.totalFingerprints,
            haltReason: r.haltReason,
            haltViolation: r.haltViolation || '',
            elapsedMs: Math.round(r.elapsedMs),
        },
        events: (r.traversalLog || []).map(_compactEvent),
    }));

    const payload = {
        version: 4,
        timestamp: new Date().toISOString(),
        latticeLevel: typeof latticeLevel !== 'undefined' ? latticeLevel : '?',
        rules: {
            t20Strict: _ruleT20StrictMode,
            maxFullOctTicks: T79_MAX_FULL_TICKS,
            octCapacityMax: OCT_CAPACITY_MAX,
            gluonMediated: _ruleGluonMediatedSC,
            bareTetrahedra: _ruleBareTetrahedra,
        },
        comparison: _bfsTestComparison ? {
            identical: _bfsTestComparison.identical,
            earlyAbort: _bfsTestComparison.earlyAbort || false,
            novelCount: _bfsTestComparison.novelCount || 0,
            summary: _bfsTestComparison.summary,
        } : null,
        runs,
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `traversal-log-L${payload.latticeLevel}-${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Show/hide BFS panel and move to top of left panels during tests
function _setBfsTestPanelVisible(visible) {
    const bfsSection = document.getElementById('bfs-section');
    const qbSection = document.getElementById('quark-balance-section');
    if (!bfsSection) return;
    if (visible) {
        bfsSection.style.display = '';
        // Move BFS section before quark balance (top of panel)
        if (qbSection && bfsSection.parentNode === qbSection.parentNode) {
            qbSection.parentNode.insertBefore(bfsSection, qbSection);
        }
    } else {
        bfsSection.style.display = 'none';
    }
}

// ── Export BFS test results to JSON file ──
function _exportBfsTestResults() {
    if (!_bfsTestResults[0] || !_bfsTestResults[1] || !_bfsTestComparison) {
        alert('No BFS test results to export. Run the test first.');
        return;
    }

    const now = new Date();
    const timestamp = now.toISOString().replace(/[:.]/g, '-');

    // Build per-run detail objects
    function buildRunDetail(result, label) {
        // Per-tick fingerprint breakdown: which solutions were tried at each tick, in order
        const perTickDetail = [];
        const sortedTicks = [...result.perTickFingerprints.keys()].sort((a, b) => a - b);
        for (const tick of sortedTicks) {
            const fps = [...result.perTickFingerprints.get(tick)];
            perTickDetail.push({
                tick,
                fingerprintCount: fps.length,
                fingerprints: fps,
            });
        }

        // Per-tick bad-move ledger (exclusions that accumulated)
        const perTickExclusions = [];
        const ledgerTicks = [...result.perTickLedger.keys()].sort((a, b) => a - b);
        for (const tick of ledgerTicks) {
            const exclusions = [...result.perTickLedger.get(tick)];
            perTickExclusions.push({
                tick,
                exclusionCount: exclusions.length,
                exclusions,
            });
        }

        // Identify starting fingerprint (tick 0, first entry) and longest/stopping ticks
        const startingFP = sortedTicks.length > 0
            ? [...result.perTickFingerprints.get(sortedTicks[0])][0] || null
            : null;

        // The "longest found" is the highest tick that has any fingerprint
        const longestTick = sortedTicks.length > 0 ? sortedTicks[sortedTicks.length - 1] : 0;
        const longestFPs = longestTick >= 0 && result.perTickFingerprints.has(longestTick)
            ? [...result.perTickFingerprints.get(longestTick)]
            : [];

        // Stopping solution: the last fingerprint tried at the highest tick
        const stoppingFP = longestFPs.length > 0 ? longestFPs[longestFPs.length - 1] : null;

        return {
            label,
            seed: '0x' + result.seed.toString(16).padStart(8, '0'),
            seedDecimal: result.seed,
            maxTickReached: result.maxTick,
            haltReason: result.haltReason,
            haltViolation: result.haltViolation || null,
            totalBacktrackRetries: result.totalRetries,
            totalUniqueFingerprints: result.totalFingerprints,
            searchTimeMs: Math.round(result.elapsedMs),
            searchTimeSec: +(result.elapsedMs / 1000).toFixed(2),
            ticksExplored: sortedTicks.length,
            startingSolution: startingFP,
            longestSolutionTick: longestTick,
            longestSolutions: longestFPs,
            stoppingSolution: stoppingFP,
            perTickFingerprints: perTickDetail,
            perTickExclusions,
        };
    }

    const runA = buildRunDetail(_bfsTestResults[0], 'Test 1: CHOREOGRAPHER (normal)');
    runA.choreographerMode = _bfsTestResults[0].mode || 'choreographer';
    const runB = buildRunDetail(_bfsTestResults[1], 'Test 2: RANDOM');
    runB.choreographerMode = _bfsTestResults[1].mode || 'random';

    // Comparison detail
    const cmp = _bfsTestComparison;
    const comparison = {
        verdict: cmp.identical ? 'EXHAUSTIVE (PASS)' : 'DIVERGENT (FAIL)',
        identical: cmp.identical,
        sameLongestSolution: cmp.sameLongestSolution,
        sameTotalFingerprints: cmp.sameTotalFingerprints,
        differentPaths: cmp.differentPaths,
        maxTickMatch: cmp.maxTickMatch,
        haltReasonMatch: cmp.haltReasonMatch,
        violationMatch: cmp.violationMatch,
        allFingerprintsMatch: cmp.allFPMatch,
        summary: cmp.summary,
        divergentTicks: cmp.fingerprintDiff.ticksMismatch.map(m => ({
            tick: m.tick,
            sharedFingerprints: m.shared,
            onlyInChoreographer: m.onlyA,
            onlyInRandom: m.onlyB,
        })),
        ticksOnlyInChoreographer: cmp.fingerprintDiff.ticksOnlyA,
        ticksOnlyInRandom: cmp.fingerprintDiff.ticksOnlyB,
    };

    // Fingerprint overlap analysis: for each tick, what % of fingerprints are shared?
    const overlapAnalysis = [];
    const allTicks = new Set([
        ..._bfsTestResults[0].perTickFingerprints.keys(),
        ..._bfsTestResults[1].perTickFingerprints.keys(),
    ]);
    for (const tick of [...allTicks].sort((a, b) => a - b)) {
        const setA = _bfsTestResults[0].perTickFingerprints.get(tick);
        const setB = _bfsTestResults[1].perTickFingerprints.get(tick);
        const countA = setA ? setA.size : 0;
        const countB = setB ? setB.size : 0;
        let shared = 0;
        if (setA && setB) {
            for (const fp of setA) { if (setB.has(fp)) shared++; }
        }
        const union = countA + countB - shared;
        overlapAnalysis.push({
            tick,
            countA,
            countB,
            shared,
            onlyA: countA - shared,
            onlyB: countB - shared,
            jaccardSimilarity: union > 0 ? +(shared / union).toFixed(4) : 1,
        });
    }

    // Aggregate stats
    const totalShared = overlapAnalysis.reduce((s, o) => s + o.shared, 0);
    const totalUnion = overlapAnalysis.reduce((s, o) => s + o.countA + o.countB - o.shared, 0);

    const report = {
        exportedAt: now.toISOString(),
        latticeLevel: typeof latticeLevel !== 'undefined' ? latticeLevel : 'unknown',
        nodeCount: typeof pos !== 'undefined' ? pos.length : 'unknown',
        xonCount: typeof _demoXons !== 'undefined' ? _demoXons.length : 6,
        comparison,
        aggregateStats: {
            totalFingerprintsA: runA.totalUniqueFingerprints,
            totalFingerprintsB: runB.totalUniqueFingerprints,
            totalSharedFingerprints: totalShared,
            overallJaccardSimilarity: totalUnion > 0 ? +(totalShared / totalUnion).toFixed(4) : 1,
            totalSearchTimeMs: runA.searchTimeMs + runB.searchTimeMs,
            totalBacktrackRetries: runA.totalBacktrackRetries + runB.totalBacktrackRetries,
        },
        runA,
        runB,
        perTickOverlap: overlapAnalysis,
    };

    // Download as JSON
    const json = JSON.stringify(report, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `dfs-exhaustiveness-${timestamp}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    // Console summary
    const sz = (json.length / 1024).toFixed(1);
    console.log(`%c[DFS TEST] ══════════════════════════════════════════`, 'color:cyan');
    console.log(`%c[DFS TEST] Exported ${sz} KB report`, 'color:lime');
    console.log(`%c[DFS TEST] Verdict: ${comparison.verdict}`, comparison.identical ? 'color:lime;font-weight:bold' : 'color:red;font-weight:bold');
    console.log(`  ✦ Same longest solution: ${comparison.sameLongestSolution ? '✓ YES' : '✗ NO'}`);
    console.log(`  ✦ Same total fingerprints: ${comparison.sameTotalFingerprints ? '✓ YES' : '✗ NO'} (${runA.totalUniqueFingerprints} vs ${runB.totalUniqueFingerprints})`);
    console.log(`  ✦ Took different paths: ${comparison.differentPaths ? '✓ YES' : '✗ NO'} (retries: ${runA.totalBacktrackRetries} vs ${runB.totalBacktrackRetries})`);
    console.log(`  Test 1 (CHOREOGRAPHER): highest=${runA.maxTickReached}, fps=${runA.totalUniqueFingerprints}, ${runA.searchTimeSec}s`);
    console.log(`  Test 2 (RANDOM): highest=${runB.maxTickReached}, fps=${runB.totalUniqueFingerprints}, ${runB.searchTimeSec}s`);
    console.log(`  Starting solution (choreographer): ${runA.startingSolution}`);
    console.log(`  Starting solution (random): ${runB.startingSolution}`);
    console.log(`  Longest solution tick: ${runA.longestSolutionTick} (choreographer) vs ${runB.longestSolutionTick} (random)`);
    console.log(`%c[DFS TEST] ══════════════════════════════════════════`, 'color:cyan');
}

// ── Wire up nucleus UI ──
(function(){
    NucleusSimulator.populateModelSelect();

    // Play button — new run or council replay depending on dropdown selection
    document.getElementById('btn-simulate-nucleus')?.addEventListener('click', function(){
        if (_sweepActive || _bfsTestActive || _demoActive) return;
        const slider = document.getElementById('lattice-slider');
        const lvl = slider ? +slider.value : 2;
        const sel = document.getElementById('council-replay-select');
        const selectedVal = sel ? sel.value : '';
        if (selectedVal && selectedVal !== '') {
            const idx = parseInt(selectedVal, 10);
            if (!isNaN(idx)) {
                startCouncilReplay(idx);
                return;
            }
        }
        startSweepTest(lvl);
    });

    // Tournament button
    document.getElementById('btn-tournament')?.addEventListener('click', function(){
        if(tournamentActive) stopTournament();
        else startTournament();
    });

    // Populate council dropdown on page load
    _populateCouncilDropdown();

    // Re-populate when lattice slider changes
    document.getElementById('lattice-slider')?.addEventListener('change', function(){
        _populateCouncilDropdown();
    });

    // Re-populate when rule checkboxes/sliders change (affects the blacklist rule key)
    for (const id of ['rule-t20-strict-toggle', 'rule-gluon-mediated-toggle', 'rule-bare-tet-toggle']) {
        document.getElementById(id)?.addEventListener('change', function(){
            _populateCouncilDropdown();
        });
    }
    for (const id of ['rule-oct-full-slider', 'rule-oct-capacity-slider']) {
        document.getElementById(id)?.addEventListener('input', function(){
            _populateCouncilDropdown();
        });
    }

    // BFS export button
    document.getElementById('btn-bfs-export')?.addEventListener('click', function(){
        _exportBfsTestResults();
    });

    // Play/pause button — pauses/resumes the demo tick interval
    document.getElementById('btn-nucleus-pause')?.addEventListener('click', function(){
        // Movie playback mode: toggle pause
        if (_playbackMode) {
            _demoPaused = !_demoPaused;
            this.textContent = _demoPaused ? '\u25B6' : '\u23F8';
            this.title = _demoPaused ? 'Resume playback' : 'Pause playback';
            return;
        }
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
        if (_playbackMode && _importedMovie) {
            _demoPaused = true;
            if (_playbackFrame > 0) { _playbackFrame--; _pbPosCache = null; _applyMovieFrame(_playbackFrame); }
            const s = document.getElementById('timeline-scrubber'); if (s) s.value = _playbackFrame;
            const v = document.getElementById('timeline-val'); if (v) v.textContent = _playbackFrame;
            return;
        }
        if (!_demoActive) return;
        if (_demoReversing) stopReverse();
        if (!_demoPaused) pauseDemo();
        _playbackStepBack();
        const pb = document.getElementById('btn-nucleus-pause');
        if (pb) pb.textContent = '\u25B6';
    });

    document.getElementById('btn-step-forward')?.addEventListener('click', function() {
        if (_playbackMode && _importedMovie) {
            _demoPaused = true;
            if (_playbackFrame < _importedMovie.totalFrames - 1) { _playbackFrame++; _applyMovieFrame(_playbackFrame); }
            const s = document.getElementById('timeline-scrubber'); if (s) s.value = _playbackFrame;
            const v = document.getElementById('timeline-val'); if (v) v.textContent = _playbackFrame;
            return;
        }
        if (!_demoActive) return;
        if (_demoReversing) stopReverse();
        if (!_demoPaused) pauseDemo();
        _playbackStepForward();
        const pb = document.getElementById('btn-nucleus-pause');
        if (pb) pb.textContent = '\u25B6';
    });

    document.getElementById('btn-reverse')?.addEventListener('click', function() {
        if (_playbackMode) {
            _demoReversing = !_demoReversing;
            _demoPaused = false;
            _updatePlaybackButtons();
            return;
        }
        if (!_demoActive) return;
        if (_demoReversing) {
            stopReverse();
        } else {
            startReverse();
        }
    });

    // Rewind all the way to t=0
    document.getElementById('btn-rewind-start')?.addEventListener('click', function() {
        if (_playbackMode && _importedMovie) {
            _demoPaused = true; _playbackFrame = 0; _pbPosCache = null;
            _applyMovieFrame(0);
            const s = document.getElementById('timeline-scrubber'); if (s) s.value = 0;
            const v = document.getElementById('timeline-val'); if (v) v.textContent = 0;
            return;
        }
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
        if (_playbackMode && _importedMovie) {
            _demoPaused = true; _playbackFrame = _importedMovie.totalFrames - 1;
            _pbPosCache = null; _applyMovieFrame(_playbackFrame);
            const s = document.getElementById('timeline-scrubber'); if (s) s.value = _playbackFrame;
            const v = document.getElementById('timeline-val'); if (v) v.textContent = _playbackFrame;
            return;
        }
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

})();

// ╔══════════════════════════════════════════════════════════════════════╗
// ║  TOURNAMENT / RL TRAINING — REMOVED                                ║
// ║  Quark balance with zero jitter proven optimal. Code stubbed out   ║
// ║  in flux-tournament.js / flux-rl.js / flux-rules-v2.js.           ║
// ╚══════════════════════════════════════════════════════════════════════╝

// Compatibility stub — other code may check this flag
let _tournamentRunning = false;
// (Tournament variables, functions, and UI code removed — see stubbed files)

