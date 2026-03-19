// flux-demo-ui-sweep.js — Playback controls, timeline, export/import, movie playback, sim UI state
// ══════════════════════════════════════════════════════════════════════════
// PLAYBACK CONTROLS — rewind, step, reverse, export
// ══════════════════════════════════════════════════════════════════════════

// Step back one tick by restoring the previous snapshot.
// Step back one tick. During replay: just decrement cursor. During live: truncate _btSnapshots.
function _playbackStepBack() {
    if (_tickInProgress) return false;
    if (_demoTick <= 0) return false;
    if (_replayCursor > 0) {
        // Replay mode: just move cursor backward
        _replayCursor--;
        _btRestoreSnapshot(_btSnapshots[_replayCursor], true);
        simHalted = false;
        _playbackUpdateDisplay();
        return true;
    }
    // Live mode: pop last snapshot, restore the one before it
    if (_btSnapshots.length < 2) return false;
    _btSnapshots.length--;  // discard current (last) snapshot
    const snap = _btSnapshots[_btSnapshots.length - 1];
    if (!snap) return false;
    _btRestoreSnapshot(snap, true); // reverse=true for fighterjet reverse animation
    simHalted = false;
    _bfsReset();
    _btReset();
    if (typeof _liveGuardResetForRewind === 'function') _liveGuardResetForRewind();
    while (_tickLog.length > 0 && _tickLog[_tickLog.length - 1].tick >= _demoTick) {
        _tickLog.pop();
    }
    _playbackUpdateDisplay();
    return true;
}

// Step forward one tick. During replay: advance cursor. During live: re-execute tick.
async function _playbackStepForward() {
    if (_tickInProgress || !_demoActive) return;
    simHalted = false;
    if (_replayCursor >= 0 && _replayCursor < _btSnapshots.length - 1) {
        // Replay mode: advance cursor
        _replayCursor++;
        _btRestoreSnapshot(_btSnapshots[_replayCursor]);
        simHalted = false;
        _playbackUpdateDisplay();
        return;
    }
    // Live mode or end of replay — re-execute tick
    _replayCursor = -1;
    const wasPaused = _demoPaused;
    _demoPaused = false;
    await demoTick();
    if (wasPaused) _demoPaused = true;
    _playbackUpdateDisplay();
}

// Begin reverse playback at current speed (responsive to slider changes).
function startReverse() {
    if (_demoReversing) return;
    pauseDemo();
    _demoReversing = true;
    let _lastVisualUpdate = 0;
    const VISUAL_INTERVAL = 33; // ~30fps
    function _reverseStep() {
        if (!_demoReversing) return;
        if (_demoTick <= 0) {
            stopReverse();
            return;
        }
        if (_replayCursor > 0) {
            // Replay mode: just decrement cursor
            _replayCursor--;
            _btRestoreSnapshot(_btSnapshots[_replayCursor], true);
        } else if (_btSnapshots.length >= 2) {
            // Live mode: truncate _btSnapshots
            _btSnapshots.length--;
            const snap = _btSnapshots[_btSnapshots.length - 1];
            if (!snap) { stopReverse(); return; }
            _btRestoreSnapshot(snap, true);
        } else {
            stopReverse();
            return;
        }
        simHalted = false;
        _bfsReset();
        _btReset();
        if (typeof _liveGuardResetForRewind === 'function') _liveGuardResetForRewind();
        while (_tickLog.length > 0 && _tickLog[_tickLog.length - 1].tick >= _demoTick) {
            _tickLog.pop();
        }
        const now = performance.now();
        if (now - _lastVisualUpdate >= VISUAL_INTERVAL) {
            _playbackUpdateDisplay();
            _lastVisualUpdate = now;
        }
        const nextMs = Math.max(4, _getDemoIntervalMs());
        _reverseInterval = setTimeout(_reverseStep, nextMs);
    }
    const intervalMs = Math.max(4, _getDemoIntervalMs());
    _reverseInterval = setTimeout(_reverseStep, intervalMs);
    _updatePlaybackButtons();
}

// Stop reverse playback.
function stopReverse() {
    _demoReversing = false;
    if (_reverseInterval) { clearTimeout(_reverseInterval); _reverseInterval = null; }
    _demoPaused = true;
    // Final visual sync so display matches current state
    _playbackUpdateDisplay();
    _updatePlaybackButtons();
}

// Refresh display after a manual step (back or forward).
// Snapshot already contains pos[], activeSet, impliedSet, xonImpliedSet —
// NO solver or detectImplied() needed. Just update 3D scene from restored state.
function _playbackUpdateDisplay() {
    // Tick counter
    const dpT = document.getElementById('dp-title');
    if (dpT) dpT.innerHTML = `${_planckSeconds} Flux Events<br><span style="font-size:0.7em; color:#8a9aaa; letter-spacing:0.05em;">${_demoTick} Planck Seconds</span><br><span style="font-size:0.55em; color:#556677; margin-top:-2px; display:block; line-height:1;">(Best ${_maxTickReached})</span>`;
    _updateTimelineScrubber();
    // Apply restored solver positions to the 3D scene (no re-solve needed)
    if (typeof applyPositions === 'function' && typeof pos !== 'undefined') applyPositions(pos);
    // Rebuild state + SC lines + void spheres from restored SC sets
    if (typeof bumpState === 'function') bumpState();
    if (typeof rebuildShortcutLines === 'function') rebuildShortcutLines();
    // Re-apply tet coloring from restored edge balance (no counting during replay)
    if (typeof _applyTetColoring === 'function') _applyTetColoring(false);
    if (typeof updateVoidSpheres === 'function') updateVoidSpheres();
    // Lazily build wavefunction + shells on first replay frame (cold-storage
    // replay may not have had valid REST positions during init).
    if (typeof _wfMesh !== 'undefined' && !_wfMesh && typeof buildWavefunction === 'function') buildWavefunction();
    if (typeof _braneShells !== 'undefined' && _braneShells.length === 0 && typeof buildBranes === 'function') buildBranes();
    if (typeof updateWavefunction === 'function') updateWavefunction();
    if (typeof updateSpheres === 'function') updateSpheres();
    if (typeof updateStatus === 'function') updateStatus();
    // Force-update bottom-stats even when simHalted (updateStatus bails early)
    _updateBottomStats();
    // Sync spark colors from role (snapshot col may be stale white from oct mode)
    for (let i = 0; i < _demoXons.length; i++) {
        const x = _demoXons[i];
        if (!x || !x.alive) continue;
        const role = typeof _xonRole === 'function' ? _xonRole(x) : 'oct';
        const roleCol = _xpRoleColor(role);
        if (x.col !== roleCol) { x.col = roleCol; if (x.sparkMat) x.sparkMat.color.setHex(roleCol); }
    }
    updateDemoPanel();
    _updateEdgeBalancePanel();
    updateXonPanel();
}

// ── Timeline Scrubber ──
// Total range = _btSnapshots.length. Current position = cursor or end.
function _updateTimelineScrubber() {
    const slider = document.getElementById('timeline-scrubber');
    const valEl = document.getElementById('timeline-val');
    if (!slider) return;
    const total = typeof _btSnapshots !== 'undefined' ? _btSnapshots.length : 0;
    const pos = _replayCursor >= 0 ? _replayCursor : total;
    slider.max = total;
    slider.value = pos;
    if (valEl) valEl.textContent = _demoTick;
}

// Seek to a specific position in the snapshot timeline.
// Direct index into _btSnapshots — no push/pop gymnastics.
function _timelineScrubTo(targetPos) {
    // Movie playback scrubbing
    if (_playbackMode && _importedMovie) {
        const idx = Math.max(0, Math.min(targetPos, _importedMovie.totalFrames - 1));
        _playbackFrame = idx;
        _pbPosCache = null; // invalidate cache for random seek
        _applyMovieFrame(idx);
        const val = document.getElementById('timeline-val');
        if (val) val.textContent = idx;
        return;
    }
    if (!_demoActive) return;
    if (_demoReversing && typeof stopReverse === 'function') stopReverse();
    if (!_demoPaused && typeof pauseDemo === 'function') pauseDemo();
    if (targetPos < 0 || targetPos >= _btSnapshots.length) return;
    const currentPos = _replayCursor >= 0 ? _replayCursor : _btSnapshots.length - 1;
    if (targetPos === currentPos) return;
    // Direct restore — just set cursor and restore snapshot
    _replayCursor = targetPos;
    _btRestoreSnapshot(_btSnapshots[targetPos]);
    if (typeof simHalted !== 'undefined') simHalted = false;
    _bfsReset(); _btReset();
    if (typeof _liveGuardResetForRewind === 'function') _liveGuardResetForRewind();
    _playbackUpdateDisplay();
    const pb = document.getElementById('btn-nucleus-pause');
    if (pb) pb.textContent = '\u25B6';
}

// Init scrubber event listener
(function _initTimelineScrubber() {
    function attach() {
        const slider = document.getElementById('timeline-scrubber');
        if (!slider) return;
        slider.addEventListener('input', function() {
            _timelineScrubTo(parseInt(this.value, 10));
        });
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', attach);
    } else {
        attach();
    }
})();

// Sync playback button visual states.
function _updatePlaybackButtons() {
    const revBtn = document.getElementById('btn-reverse');
    const pauseBtn = document.getElementById('btn-nucleus-pause');
    if (revBtn) {
        revBtn.style.borderColor = _demoReversing ? '#ff8844' : '#6688aa';
        revBtn.style.color = _demoReversing ? '#ffaa66' : '#88aacc';
    }
    if (pauseBtn) {
        pauseBtn.textContent = (_demoPaused && !_demoReversing) ? '\u25B6' : '\u23F8';
    }
}

// Export tick log or council replay snapshots as downloadable JSON.
function exportTickLog() {
    // During council replay or paused demo, export full snapshots
    if (_btSnapshots && _btSnapshots.length > 0 && typeof _serializeSnapshot === 'function') {
        // Serialize then reconstruct trails for export (modern snapshots use trailLen only)
        const snapData = _btSnapshots.map(s => _serializeSnapshot(s));
        // Reconstruct unified trail arrays so exported file is self-contained
        const numXons = snapData[0] && snapData[0].xons ? snapData[0].xons.length : 0;
        for (let si = 0; si < snapData.length; si++) {
            const snap = snapData[si];
            if (!snap.xons) continue;
            for (let xi = 0; xi < numXons && xi < snap.xons.length; xi++) {
                const xon = snap.xons[xi];
                const role = xon._role || 'oct';
                xon.col = (typeof _xpRoleColor === 'function') ? _xpRoleColor(role) : xon.col || 0xffffff;
                const p = snap.pos && snap.pos[xon.node];
                const entry = { node: xon.node, role, pos: p ? [p[0], p[1], p[2]] : [0, 0, 0] };
                if (si === 0) {
                    xon.trail = [entry];
                } else {
                    const prev = snapData[si - 1].xons[xi];
                    xon.trail = prev.trail.concat([entry]);
                }
            }
        }
        const seed = _forceSeed || _runSeed || 0;
        const peak = _btSnapshots[_btSnapshots.length - 1].tick || 0;
        const data = {
            version: 2, type: 'council-replay',
            exported: new Date().toISOString(),
            seed: seed, peak: peak,
            totalSnapshots: snapData.length,
            snapshots: snapData
        };
        const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const seedHex = seed.toString(16).padStart(8, '0');
        a.download = `flux-replay-0x${seedHex}-t${peak}-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        console.log(`[export] Downloaded ${snapData.length} snapshots (seed 0x${seedHex}, peak t${peak})`);
        return;
    }
    if (_tickLog.length === 0) {
        console.warn('[export] No data to export');
        return;
    }
    const data = {
        version: 2,
        exported: new Date().toISOString(),
        totalTicks: _tickLog.length,
        format: { guards: 'delta — tick 0 has full state, subsequent ticks have only changed keys (null = no change)' },
        params: JSON.parse(JSON.stringify(_choreoParams)),
        log: _tickLog
    };
    const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `flux-log-${_tickLog.length}ticks-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    console.log(`[export] Downloaded ${_tickLog.length} tick log entries`);
}

// Import a previously exported council replay and play it back.
function importReplay(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const data = JSON.parse(e.target.result);
            if (data.type !== 'council-replay' || !data.snapshots || data.snapshots.length === 0) {
                console.warn('[import] Not a valid council replay export');
                alert('Not a valid council replay file.');
                return;
            }
            // Deserialize snapshots
            const snapshots = data.snapshots.map(s => _deserializeSnapshot(s));
            // Inject as a temporary council member
            const member = {
                seed: data.seed || 1,
                peak: data.peak || snapshots[snapshots.length - 1].tick || 0,
                snapshots: snapshots,
                _cold: false,
                _imported: true,
            };
            // Add to front of council and trigger replay
            _sweepGoldenCouncil.unshift(member);
            console.log(`%c[import] Loaded replay: seed 0x${member.seed.toString(16).padStart(8,'0')}, peak t${member.peak}, ${snapshots.length} snapshots`, 'color:#66ccff;font-weight:bold');
            // Start replay on the newly inserted member (index 0)
            if (typeof startCouncilReplay === 'function') {
                startCouncilReplay(0);
            }
        } catch (err) {
            console.error('[import] Failed to parse replay file:', err);
            alert('Failed to parse replay file: ' + err.message);
        }
    };
    reader.readAsText(file);
}

// ── Movie export ──
function exportMovie() {
    if (_movieFrames.length === 0) {
        console.warn('[movie] No movie frames to export');
        return;
    }
    const data = {
        version: 3,
        type: 'movie',
        exported: new Date().toISOString(),
        latticeLevel: latticeLevel,
        nodeCount: N,
        totalFrames: _movieFrames.length,
        frames: _movieFrames
    };
    const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `flux-movie-${_movieFrames.length}f-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    console.log(`[movie] Downloaded ${_movieFrames.length} frames`);
}

// ── Movie import ──
function importMovie() {
    const input = document.getElementById('movie-file-input');
    if (input) input.click();
}

function _handleMovieFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function(ev) {
        try {
            const data = JSON.parse(ev.target.result);
            if (data.version < 3 || data.type !== 'movie') {
                console.error('[movie] Invalid movie file (need version ≥ 3, type=movie)');
                return;
            }
            if (!data.frames || data.frames.length === 0) {
                console.error('[movie] No frames in movie file');
                return;
            }
            _importedMovie = data;
            _startMoviePlayback();
        } catch (err) {
            console.error('[movie] Failed to parse movie file:', err);
        }
    };
    reader.readAsText(file);
    e.target.value = ''; // reset so same file can be re-imported
}

// ── Movie playback engine ──
// Create a bare xon with Three.js visuals for movie playback (no nucleus data needed).
function _spawnPlaybackXon(startNode) {
    const col = 0xffffff;
    const sparkMat = new THREE.SpriteMaterial({
        color: col, map: _sparkTex, transparent: true, opacity: 1.0,
        blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
    });
    const spark = new THREE.Sprite(sparkMat);
    spark.scale.set(0.28, 0.28, 1);
    spark.renderOrder = 22;
    const group = new THREE.Group();
    group.add(spark);
    if (pos[startNode]) group.position.set(pos[startNode][0], pos[startNode][1], pos[startNode][2]);
    scene.add(group);

    const trailGeo = new THREE.BufferGeometry();
    const trailPos = new Float32Array((XON_TRAIL_VERTS + 1) * 3);
    const trailCol = new Float32Array((XON_TRAIL_VERTS + 1) * 3);
    trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
    trailGeo.setAttribute('color', new THREE.BufferAttribute(trailCol, 3));
    const trailMat = new THREE.LineBasicMaterial({
        vertexColors: true, transparent: true, opacity: 1.0,
        depthTest: false, blending: THREE.AdditiveBlending,
    });
    const trailLine = new THREE.Line(trailGeo, trailMat);
    trailLine.renderOrder = 20;
    scene.add(trailLine);

    const xon = {
        node: startNode, prevNode: startNode, sign: 1,
        _loopType: null, _loopSeq: null, _loopStep: 0,
        _assignedFace: null, _quarkType: null,
        _mode: 'oct', _lastDir: null, _dirHistory: [],
        _dirBalance: new Array(10).fill(0),
        _modeStats: { oct: 0, tet: 0, idle_tet: 0, weak: 0, gluon: 0 },
        col, group, spark, sparkMat,
        trailLine, trailGeo, trailPos, trailCol,
        trail: [{ node: startNode, role: 'oct', pos: pos[startNode] ? [pos[startNode][0], pos[startNode][1], pos[startNode][2]] : [0,0,0] }],
        tweenT: 1, flashT: 1.0, _highlightT: 0, alive: true,
    };
    _demoXons.push(xon);
    return xon;
}

function _startMoviePlayback() {
    if (!_importedMovie) return;
    // Stop any running demo
    if (_demoActive) {
        if (_demoInterval) { clearInterval(_demoInterval); _demoInterval = null; }
        _demoActive = false;
    }
    // Build lattice if not already built
    if (!pos || pos.length === 0) {
        buildLattice(_importedMovie.latticeLevel || 2);
    }
    // Apply frame 0 positions so lattice is correct before spawning
    const f0 = _importedMovie.frames[0];
    if (f0 && f0.pos) {
        for (let i = 0; i < f0.pos.length && i < pos.length; i++) {
            pos[i][0] = f0.pos[i][0]; pos[i][1] = f0.pos[i][1]; pos[i][2] = f0.pos[i][2];
        }
    }
    // Spawn playback xons (clean slate)
    for (const x of _demoXons) {
        if (x.group) scene.remove(x.group);
        if (x.trailLine) scene.remove(x.trailLine);
    }
    _demoXons.length = 0;
    const xonCount = f0 ? f0.xons.length : 6;
    for (let i = 0; i < xonCount; i++) {
        const startNode = f0 ? f0.xons[i].n : 0;
        _spawnPlaybackXon(startNode);
    }
    _playbackMode = true;
    _playbackFrame = 0;
    _demoPaused = false;
    _demoReversing = false;
    // Show playback controls + pause button
    const pc = document.getElementById('playback-controls');
    if (pc) pc.style.display = '';
    const pauseBtn = document.getElementById('btn-nucleus-pause');
    if (pauseBtn) { pauseBtn.style.display = ''; pauseBtn.textContent = '\u23F8'; }
    const scrubber = document.getElementById('timeline-scrubber');
    if (scrubber) {
        scrubber.max = _importedMovie.totalFrames - 1;
        scrubber.value = 0;
    }
    const val = document.getElementById('timeline-val');
    if (val) val.textContent = '0';
    // Show deuteron panel + populate legend (normally done by enterNucleusMode)
    // Respect user's sidebar toggle
    const dp = document.getElementById('deuteron-panel');
    const leftVis = typeof _isLeftSidebarVisible === 'function' ? _isLeftSidebarVisible() : true;
    if (dp && leftVis) dp.style.display = 'block';
    if (typeof _populateDeuteronQuarkLegend === 'function') _populateDeuteronQuarkLegend();
    // Reset visit tracking for sidebar
    _demoVisits = {};
    for (let f = 1; f <= 8; f++) _demoVisits[f] = {};
    _demoTypeBalanceHistory = [];
    _demoTick = 0;
    _pbVisitsUpTo = -1;
    // Apply first frame
    _applyMovieFrame(0);
    // Start playback interval (same speed as demo: 60fps tick)
    _demoInterval = setInterval(_moviePlaybackTick, 17);
    console.log(`[movie] Playing back ${_importedMovie.totalFrames} frames`);
}

function _moviePlaybackTick() {
    if (!_playbackMode || !_importedMovie) return;
    if (_demoPaused) return;
    if (_demoReversing) {
        if (_playbackFrame > 0) {
            _playbackFrame--;
            _applyMovieFrame(_playbackFrame);
        }
    } else {
        if (_playbackFrame < _importedMovie.totalFrames - 1) {
            _playbackFrame++;
            _applyMovieFrame(_playbackFrame);
        }
    }
    // Update scrubber
    const scrubber = document.getElementById('timeline-scrubber');
    if (scrubber) scrubber.value = _playbackFrame;
    const val = document.getElementById('timeline-val');
    if (val) val.textContent = _playbackFrame;
}

// Cached position reconstruction — avoids re-walking from frame 0 each time.
// _pbPosCache stores { idx, pos[][] } for the last reconstructed frame.
let _pbPosCache = null;

function _reconstructPos(frameIdx) {
    const frames = _importedMovie.frames;
    // If cache is at or before target, continue from cache; else restart from 0
    let startIdx = 0;
    let p;
    if (_pbPosCache && _pbPosCache.idx <= frameIdx) {
        startIdx = _pbPosCache.idx + 1;
        p = _pbPosCache.pos.map(v => [v[0], v[1], v[2]]);
    } else {
        p = frames[0].pos.map(v => [v[0], v[1], v[2]]);
        startIdx = 1;
    }
    for (let i = startIdx; i <= frameIdx; i++) {
        if (frames[i].dp) {
            for (const [nodeIdx, x, y, z] of frames[i].dp) {
                p[nodeIdx][0] = x; p[nodeIdx][1] = y; p[nodeIdx][2] = z;
            }
        }
    }
    _pbPosCache = { idx: frameIdx, pos: p.map(v => [v[0], v[1], v[2]]) };
    return p;
}

// Color for a given mode+quark
function _modeColor(mode, quark) {
    if (quark && QUARK_COLORS[quark]) return QUARK_COLORS[quark];
    if (mode === 'weak') return WEAK_FORCE_COLOR;
    if (mode === 'gluon') return GLUON_COLOR;
    return 0xffffff; // oct or unassigned
}

// Playback trail length: read from trails slider (matches live trail behavior)
function _playbackTrailLen() {
    const el = document.getElementById('tracer-lifespan-slider');
    return el ? +el.value : 55;
}

// Incremental visit tracker for playback
let _pbVisitsUpTo = -1; // last frame index we've counted visits for

// Reconstruct _demoVisits from movie frames [0..idx] for sidebar display
function _reconstructPlaybackVisits(idx) {
    const frames = _importedMovie.frames;
    // If scrubbing backwards, reset and rebuild
    if (idx < _pbVisitsUpTo) {
        _demoVisits = {};
        for (let f = 1; f <= 8; f++) _demoVisits[f] = {};
        _pbVisitsUpTo = -1;
    }
    // Ensure _demoVisits initialized
    if (!_demoVisits || _pbVisitsUpTo < 0) {
        _demoVisits = {};
        for (let f = 1; f <= 8; f++) _demoVisits[f] = {};
        _pbVisitsUpTo = -1;
    }
    // Incrementally add frames from _pbVisitsUpTo+1 to idx
    for (let t = _pbVisitsUpTo + 1; t <= idx && t < frames.length; t++) {
        const fr = frames[t];
        if (!fr || !fr.xons) continue;
        for (const x of fr.xons) {
            if (x.f && x.f > 0 && x.q && (x.m === 'tet' || x.m === 'idle_tet')) {
                _demoVisits[x.f][x.q] = (_demoVisits[x.f][x.q] || 0) + 1;
            }
        }
    }
    _pbVisitsUpTo = idx;
    _actualizationVisits = _demoVisits;
}

function _applyMovieFrame(idx) {
    if (!_importedMovie) return;
    const frames = _importedMovie.frames;
    const frame = frames[idx];
    if (!frame) return;

    // Set tick counter for sidebar
    _demoTick = idx;

    // 1. Reconstruct positions
    const rPos = _reconstructPos(idx);
    for (let i = 0; i < rPos.length && i < pos.length; i++) {
        pos[i][0] = rPos[i][0];
        pos[i][1] = rPos[i][1];
        pos[i][2] = rPos[i][2];
    }

    // 2. Restore SC sets
    activeSet.clear(); for (const id of frame.a) activeSet.add(id);
    xonImpliedSet.clear(); for (const id of frame.xi) xonImpliedSet.add(id);
    impliedSet.clear(); for (const id of frame.im) impliedSet.add(id);
    // Fold xonImplied into implied for rendering (mirrors what _detectImplied does)
    for (const id of frame.xi) impliedSet.add(id);

    // 3. Position xons with trails
    for (let i = 0; i < _demoXons.length && i < frame.xons.length; i++) {
        const x = _demoXons[i];
        const fx = frame.xons[i];
        // Derive prevNode from previous frame
        const prevFrame = idx > 0 ? frames[idx - 1] : null;
        const prevNode = (prevFrame && prevFrame.xons[i]) ? prevFrame.xons[i].n : fx.n;

        x._restoring = true;
        x.prevNode = prevNode;
        x.node = fx.n;
        x._restoring = false;
        x._mode = fx.m;
        x._quarkType = fx.q;
        x.col = _modeColor(fx.m, fx.q);
        x.alive = true;

        // Update spark color
        if (x.sparkMat) x.sparkMat.color.setHex(x.col);

        // Build trail from history (last N frames)
        // Use current reconstructed positions for all trail entries (close enough
        // since node positions barely shift between adjacent frames)
        const trailStart = Math.max(0, idx - _playbackTrailLen() + 1);
        x.trail = [];
        for (let t = trailStart; t <= idx; t++) {
            const tf = frames[t];
            if (!tf || !tf.xons[i]) continue;
            const tn = tf.xons[i].n;
            const m = tf.xons[i].m, q = tf.xons[i].q;
            const role = (m === 'tet' || m === 'idle_tet') ? (q || 'oct') :
                         m === 'gluon' ? 'gluon' : m === 'weak' ? 'weak' : 'oct';
            x.trail.push({ node: tn, role, pos: [rPos[tn][0], rPos[tn][1], rPos[tn][2]] });
        }

        // Snap sprite to position
        if (x.group && pos[x.node]) {
            x.group.position.set(pos[x.node][0], pos[x.node][1], pos[x.node][2]);
        }
        x.tweenT = 1;
    }

    // 4. Rebuild visuals
    rebuildBaseLines();
    rebuildShortcutLines();
    updateVoidSpheres();
    updateSpheres();
    // 5. Reconstruct visit data and update sidebar panels
    try {
        _reconstructPlaybackVisits(idx);
        // Reset balance history so sparkline builds up during playback
        if (idx === 0) _demoTypeBalanceHistory = [];
        if (typeof updateDemoPanel === 'function') updateDemoPanel();
    } catch(e) {}
    try { if (typeof updateXonPanel === 'function') updateXonPanel(); } catch(e) {}
}

function _stopMoviePlayback() {
    _playbackMode = false;
    _importedMovie = null;
    _playbackFrame = 0;
    _pbPosCache = null;
    if (_demoInterval) { clearInterval(_demoInterval); _demoInterval = null; }
}

// ── Trail curve sliders ──
{
    const _fadeSlider = document.getElementById('trail-fade-slider');
    const _fadeVal = document.getElementById('trail-fade-val');
    if (_fadeSlider) {
        _fadeSlider.addEventListener('input', e => {
            _trailFadeFloor = +e.target.value / 100;
            if (_fadeVal) _fadeVal.textContent = e.target.value + '%';
        });
    }
    const _curveSlider = document.getElementById('trail-curve-slider');
    const _curveVal = document.getElementById('trail-curve-val');
    if (_curveSlider) {
        _curveSlider.addEventListener('input', e => {
            _fjCurvature = +e.target.value * 0.89 / 100;
            if (_curveVal) _curveVal.textContent = e.target.value + '%';
        });
    }
    const _tensionSlider = document.getElementById('trail-tension-slider');
    const _tensionVal = document.getElementById('trail-tension-val');
    if (_tensionSlider) {
        _tensionSlider.addEventListener('input', e => {
            _fjTension = +e.target.value / 100;
            if (_tensionVal) _tensionVal.textContent = _fjTension.toFixed(2);
        });
    }
    const _alphaSlider = document.getElementById('trail-alpha-slider');
    const _alphaVal = document.getElementById('trail-alpha-val');
    if (_alphaSlider) {
        _alphaSlider.addEventListener('input', e => {
            _fjAlpha = +e.target.value / 100;
            if (_alphaVal) _alphaVal.textContent = _fjAlpha.toFixed(2);
        });
    }
}

// ── Rules switchboard toggles ──
{
    const _t20El = document.getElementById('rule-t20-strict-toggle');
    if (_t20El) {
        _ruleT20StrictMode = _t20El.checked; // HTML is source of truth
        _t20El.addEventListener('change', e => { _ruleT20StrictMode = e.target.checked; _populateCouncilDropdown(); });
    }
}
{
    const _octFullEl = document.getElementById('rule-oct-full-slider');
    if (_octFullEl) {
        _octFullEl.value = T79_MAX_FULL_TICKS; // reset to JS default
        _octFullEl.addEventListener('input', e => {
            T79_MAX_FULL_TICKS = parseInt(e.target.value, 10);
            const lbl = document.getElementById('rule-oct-full-value');
            if (lbl) lbl.textContent = T79_MAX_FULL_TICKS;
            _populateCouncilDropdown();
        });
    }
}
// OCT_CAPACITY_MAX and _populateCouncilDropdown live in flux-tests.js which loads AFTER this file.
// All switchboard listeners that reference late-loaded globals must be deferred to DOMContentLoaded
// (by which time all <script> tags have executed).
window.addEventListener('DOMContentLoaded', () => {
    const _octCapEl = document.getElementById('rule-oct-capacity-slider');
    if (_octCapEl) {
        _octCapEl.value = OCT_CAPACITY_MAX;
        _octCapEl.addEventListener('input', e => {
            OCT_CAPACITY_MAX = parseInt(e.target.value, 10);
            const lbl = document.getElementById('rule-oct-capacity-value');
            if (lbl) lbl.textContent = OCT_CAPACITY_MAX;
            _populateCouncilDropdown();
        });
    }
    // Rule toggle checkboxes: HTML checked attribute is the single source of truth.
    // JS variables sync FROM the DOM at init, not the other way around.
    const _gluonEl = document.getElementById('rule-gluon-mediated-toggle');
    if (_gluonEl) {
        _ruleGluonMediatedSC = _gluonEl.checked;
        _gluonEl.addEventListener('change', e => { _ruleGluonMediatedSC = e.target.checked; _populateCouncilDropdown(); });
    }
    const _bareEl = document.getElementById('rule-bare-tet-toggle');
    if (_bareEl) {
        _ruleBareTetrahedra = _bareEl.checked;
        _bareEl.addEventListener('change', e => { _ruleBareTetrahedra = e.target.checked; _populateCouncilDropdown(); });
    }
    const _projGuardEl = document.getElementById('rule-projected-guards-toggle');
    if (_projGuardEl) {
        _ruleProjectedGuards = _projGuardEl.checked;
        _projGuardEl.addEventListener('change', e => { _ruleProjectedGuards = e.target.checked; _populateCouncilDropdown(); });
    }
    const _t90TolEl = document.getElementById('rule-t90-tolerance-slider');
    if (_t90TolEl) {
        _t90TolEl.value = T90_TOLERANCE;
        _t90TolEl.addEventListener('input', e => {
            T90_TOLERANCE = parseInt(e.target.value, 10);
            const lbl = document.getElementById('rule-t90-tolerance-value');
            if (lbl) lbl.textContent = T90_TOLERANCE;
            _populateCouncilDropdown();
        });
    }
    const _t91TolEl = document.getElementById('rule-t91-tolerance-slider');
    if (_t91TolEl) {
        _t91TolEl.value = T91_TOLERANCE;
        _t91TolEl.addEventListener('input', e => {
            T91_TOLERANCE = parseInt(e.target.value, 10);
            const lbl = document.getElementById('rule-t91-tolerance-value');
            if (lbl) lbl.textContent = T91_TOLERANCE;
            _populateCouncilDropdown();
        });
    }
    const _t92TolEl = document.getElementById('rule-t92-tolerance-slider');
    if (_t92TolEl) {
        _t92TolEl.value = T92_TOLERANCE;
        _t92TolEl.addEventListener('input', e => {
            T92_TOLERANCE = parseInt(e.target.value, 10);
            const lbl = document.getElementById('rule-t92-tolerance-value');
            if (lbl) lbl.textContent = T92_TOLERANCE;
            _populateCouncilDropdown();
        });
    }
    // Rule 9 & 10: adaptive ejection — mutually exclusive with rule 8 sliders and each other
    const _adaptEl = document.getElementById('rule-adaptive-ejection-toggle');
    const _cubeEl = document.getElementById('rule-cuberoot-ejection-toggle');
    const _rule8Sliders = ['rule-t90-tolerance-slider', 'rule-t91-tolerance-slider', 'rule-t92-tolerance-slider'];
    function _syncRule8vs9vs10() {
        const on = _ruleAdaptiveEjection || _ruleCubeRootEjection;
        for (const id of _rule8Sliders) {
            const sl = document.getElementById(id);
            if (sl) { sl.disabled = on; sl.style.opacity = on ? '0.3' : '1'; }
        }
        // Also disable ticker buttons for rule 8 sliders
        document.querySelectorAll('.tol-tick').forEach(btn => {
            if (_rule8Sliders.includes(btn.dataset.slider)) {
                btn.disabled = on; btn.style.opacity = on ? '0.3' : '1';
            }
        });
        // Dim the other adaptive checkbox when one is active
        if (_adaptEl) { _adaptEl.disabled = _ruleCubeRootEjection; _adaptEl.parentElement.style.opacity = _ruleCubeRootEjection ? '0.3' : '1'; }
        if (_cubeEl) { _cubeEl.disabled = _ruleAdaptiveEjection; _cubeEl.parentElement.style.opacity = _ruleAdaptiveEjection ? '0.3' : '1'; }
    }
    if (_adaptEl) {
        _ruleAdaptiveEjection = _adaptEl.checked;
        _adaptEl.addEventListener('change', e => {
            _ruleAdaptiveEjection = e.target.checked;
            if (e.target.checked) { _ruleCubeRootEjection = false; if (_cubeEl) _cubeEl.checked = false; }
            _syncRule8vs9vs10();
            _populateCouncilDropdown();
        });
    }
    if (_cubeEl) {
        _ruleCubeRootEjection = _cubeEl.checked;
        _cubeEl.addEventListener('change', e => {
            _ruleCubeRootEjection = e.target.checked;
            if (e.target.checked) { _ruleAdaptiveEjection = false; if (_adaptEl) _adaptEl.checked = false; }
            _syncRule8vs9vs10();
            _populateCouncilDropdown();
        });
    }
    _syncRule8vs9vs10(); // apply initial state
    // Wire left/right ticker buttons for tolerance sliders
    document.querySelectorAll('.tol-tick').forEach(btn => {
        btn.addEventListener('click', () => {
            const sl = document.getElementById(btn.dataset.slider);
            if (!sl) return;
            const dir = parseInt(btn.dataset.dir, 10);
            const nv = Math.max(+sl.min, Math.min(+sl.max, +sl.value + dir));
            if (nv !== +sl.value) { sl.value = nv; sl.dispatchEvent(new Event('input')); }
        });
    });
    // Sync JS from DOM after browser form restoration.
    // Read DOM → JS so JS always matches what the user sees.
    const _syncJSFromDOM = () => {
        const t20El = document.getElementById('rule-t20-strict-toggle');
        const gluEl = document.getElementById('rule-gluon-mediated-toggle');
        const bareEl = document.getElementById('rule-bare-tet-toggle');
        const projEl = document.getElementById('rule-projected-guards-toggle');
        const octFullEl = document.getElementById('rule-oct-full-slider');
        const octCapEl = document.getElementById('rule-oct-capacity-slider');
        if (t20El) _ruleT20StrictMode = t20El.checked;
        if (gluEl) _ruleGluonMediatedSC = gluEl.checked;
        if (bareEl) _ruleBareTetrahedra = bareEl.checked;
        if (projEl) _ruleProjectedGuards = projEl.checked;
        if (octFullEl) T79_MAX_FULL_TICKS = parseInt(octFullEl.value, 10);
        if (octCapEl) OCT_CAPACITY_MAX = parseInt(octCapEl.value, 10);
        const t90TolEl = document.getElementById('rule-t90-tolerance-slider');
        const t91TolEl = document.getElementById('rule-t91-tolerance-slider');
        const t92TolEl = document.getElementById('rule-t92-tolerance-slider');
        if (t90TolEl) T90_TOLERANCE = parseInt(t90TolEl.value, 10);
        if (t91TolEl) T91_TOLERANCE = parseInt(t91TolEl.value, 10);
        if (t92TolEl) T92_TOLERANCE = parseInt(t92TolEl.value, 10);
        const adaptEl = document.getElementById('rule-adaptive-ejection-toggle');
        if (adaptEl) _ruleAdaptiveEjection = adaptEl.checked;
        const cubeEl = document.getElementById('rule-cuberoot-ejection-toggle');
        if (cubeEl) _ruleCubeRootEjection = cubeEl.checked;
        // Per-role opacity sliders
        for (const key of Object.keys(_roleOpacity)) {
            const sl = document.getElementById(`role-opacity-${key}`);
            if (sl) _roleOpacity[key] = +sl.value / 100;
        }
        // Per-xon opacity sliders
        for (let i = 0; i < 6; i++) {
            const sl = document.getElementById(`xon-opacity-${i}`);
            if (sl) _xonOpacity[i] = +sl.value / 100;
        }
        _populateCouncilDropdown();
    };
    // Fire after load, pageshow, AND with escalating delays to catch late browser restoration
    window.addEventListener('load', () => { setTimeout(_syncJSFromDOM, 0); setTimeout(_syncJSFromDOM, 100); });
    window.addEventListener('pageshow', () => { setTimeout(_syncJSFromDOM, 0); setTimeout(_syncJSFromDOM, 100); });

}); // end DOMContentLoaded

// ── Simulation UI state: button swap + rule locking ──
function _setSimUIActive(active) {
    const startRow = document.getElementById('sim-start-buttons');
    const activeRow = document.getElementById('sim-active-buttons');
    if (startRow) startRow.style.display = active ? 'none' : 'flex';
    if (activeRow) activeRow.style.display = active ? 'flex' : 'none';
    // Lock/unlock rule toggles
    const toggleIds = ['rule-t20-strict-toggle', 'rule-gluon-mediated-toggle', 'rule-bare-tet-toggle', 'rule-oct-full-slider', 'rule-oct-capacity-slider', 'rule-projected-guards-toggle', 'rule-t90-tolerance-slider', 'rule-t91-tolerance-slider', 'rule-t92-tolerance-slider', 'rule-adaptive-ejection-toggle'];
    for (const id of toggleIds) {
        const el = document.getElementById(id);
        if (el) { el.disabled = active; el.style.opacity = active ? '0.4' : '1'; }
    }
    const sbPanel = document.getElementById('switchboard-panel');
    if (sbPanel) sbPanel.style.opacity = active ? '0.6' : '1';
}

// Clear Simulation button with confirmation + cancel
function _clearSimReset() {
    const btn = document.getElementById('btn-clear-simulation');
    const cancel = document.getElementById('btn-clear-cancel');
    if (btn) { btn._confirmed = false; btn.textContent = 'Clear Simulation'; btn.style.background = 'rgba(255,60,60,0.15)'; }
    if (cancel) cancel.style.display = 'none';
}
document.getElementById('btn-clear-simulation')?.addEventListener('click', function() {
    if (!this._confirmed) {
        this._confirmed = true;
        this.textContent = 'Are you sure?';
        this.style.background = 'rgba(255,60,60,0.3)';
        const cancel = document.getElementById('btn-clear-cancel');
        if (cancel) cancel.style.display = '';
        return;
    }
    window.location.reload();
});
document.getElementById('btn-clear-cancel')?.addEventListener('click', _clearSimReset);
