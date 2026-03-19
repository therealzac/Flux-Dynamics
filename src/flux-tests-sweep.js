// flux-tests-sweep.js — Sweep loop, council management, auto-retry, clear cache
// Split from flux-tests.js (lines 3661-4132). Loaded after flux-tests-idb.js.

// ── startSweepSeed: single-seed entry point ──
// Initializes the demo for one seed and returns immediately.
// If replayMember is provided, populates the redo stack from cold storage
// (council replay = sweep with pre-seeded redo stack).
// The caller polls for completion — this function does NOT loop.
async function startSweepSeed(seed, replayMember, lvl, _startupLog) {
    if (!_startupLog) _startupLog = () => {};

    // 1. Set sweep/replay flags
    _forceSeed = seed;
    _sweepReplayActive = !!replayMember;
    _sweepReplayMember = replayMember || null;
    _replayAncestorPeak = replayMember ? replayMember.peak : -1;

    if (replayMember) {
        console.log(`%c[REPLAY] Starting council replay — seed 0x${seed.toString(16).padStart(8,'0')}, peak t${replayMember.peak}`, 'color:#66ccff;font-weight:bold');
    }

    // 2. Reset per-seed state
    simHalted = false;
    _btBadMoveLedger.clear();
    _btTriedFingerprints.clear();
    _sweepBlacklistHitsSeed = 0;
    _sweepGoldenHitsSeed = 0;
    _sweepSeedMoves = new Map();
    _searchTraversalLog = [];
    _searchEventCounter = 0;
    _searchPathStack = [];
    _searchParentNodeId = null;
    _searchLastCandidates = null;
    _searchStartTime = performance.now();
    if (typeof _lockRules === 'function') _lockRules(true);

    // 3. Set up nucleus
    if (typeof stopDemo === 'function' && _demoActive) stopDemo();
    _startupLog('simulateNucleus() starting...');
    NucleusSimulator.simulateNucleus();
    _startupLog('simulateNucleus() done, yielding 100ms...');
    await new Promise(r => setTimeout(r, 100));
    _startupLog('yield done');
    _bfsTestRandomChoreographer = false; // GC mode

    // 4. Hydrate from cold storage BEFORE starting the demo loop.
    //    The await yields to the event loop — if startDemoLoop() ran
    //    first, the uncapped tick loop would execute live ticks during
    //    the IDB read, corrupting the fresh state.
    let replaySnapshots = null;
    if (replayMember) {
        if (replayMember._cold) {
            _startupLog('Hydrating council member from cold storage...');
            await _hydrateCouncilMember(lvl || 2, replayMember);
            _startupLog('Council member hydrated');
        }
        if (replayMember.snapshots && replayMember.snapshots.length > 0) {
            replaySnapshots = replayMember.snapshots;
        }
    }

    // 5. Initialize simulation
    _setBfsTestPanelVisible(true);
    _startupLog('startDemoLoop() starting...');
    startDemoLoop();
    _startupLog('startDemoLoop() done');

    // 6. If replaying: load snapshots into _btSnapshots, restore t=0
    if (replaySnapshots) {
        _btSnapshots.length = 0;
        _btSnapshots.push(...replaySnapshots);
        // Cold boundary: all loaded snapshots are raw IDB format (already serialized).
        // Keep a copy so we can skip re-serialization at save time.
        _btColdBoundary = replaySnapshots.length;
        _btColdSnapshots = replaySnapshots.slice(); // shallow copy — raw objects are immutable
        _replayCursor = 0;
        _btRestoreSnapshot(_btSnapshots[0]);
        console.log(`%c[REPLAY] Loaded ${replaySnapshots.length} snapshots into _btSnapshots — cursor at 0`, 'color:#66ccff;font-weight:bold');
        // Kill the live tick loop so resumeDemo() starts cursor-based replay
        if (_demoInterval) { clearInterval(_demoInterval); _demoInterval = null; }
        if (_demoUncappedId) { clearTimeout(_demoUncappedId); _demoUncappedId = null; }
        // Council-grade runs have already revealed the oct — force it so
        // void spheres and shells render from the first replay frame.
        _demoOctRevealed = true;
        for (let f = 1; f <= 8; f++) _demoVisitedFaces.add(f);
        pauseDemo();
        _testRunning = false; // enable rendering for entire replay seed
        // Ensure opacity defaults are applied for replay visuals
        for (const [id, val] of DEMO_VISUAL_DEFAULTS) {
            const el = document.getElementById(id);
            if (el) { el.value = val; el.dispatchEvent(new Event('input')); }
        }
        resumeDemo();
    }
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
    _sweepUsedSeeds = new Set();
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
        _sweepUsedSeeds = new Set(cached.usedSeeds || []);
        if (cached.goldenCouncil && cached.goldenCouncil.length > 0) {
            _sweepGoldenCouncil = cached.goldenCouncil;
            // Trim to current max size (may have grown under older code)
            _trimCouncil(lvl);
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

    // ── Seed iteration loop ──
    // Runs startSweepSeed() for each new seed (or council replay).
    // startSweepSeed() initializes the demo and returns immediately;
    // the sweep loop polls for completion before moving to the next seed.
    let _replayOnFirstSeed = (typeof replayMemberIdx === 'number' && replayMemberIdx >= 0
        && _sweepGoldenCouncil.length > replayMemberIdx) ? _sweepGoldenCouncil[replayMemberIdx] : null;

    while (_sweepActive) {
        // Pick a random seed we haven't used yet
        let seed;
        do { seed = (Math.random() * 0xFFFFFFFF) >>> 0; } while (_sweepUsedSeeds.has(seed) || seed === 0);
        _sweepUsedSeeds.add(seed);

        // Council replay: optionally use a NEW seed for live play (checkbox).
        // The saved snapshots are deterministic (recorded state, not re-simulated),
        // so the seed only affects live play after snapshots end.
        const replayMember = _replayOnFirstSeed || null;
        const _nsChk = document.getElementById('chk-new-seed-retry');
        if (replayMember && !(_nsChk && _nsChk.checked)) {
            seed = replayMember.seed; // same seed: rely on blacklist for divergence
        }

        await startSweepSeed(seed, replayMember, lvl, _startupLog);

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
        // If sweep was killed (user stop or _replayTestFail), break BEFORE
        // stopDemo so the user can scrub through the failed replay
        // (stopDemo destroys xons & clears _demoActive).
        if (!_sweepActive) break;

        if (typeof stopDemo === 'function') stopDemo();

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

        // After each seed: decide next mode based on auto-retry-best checkbox
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
        let _seedBacktracked = false;

        // Golden council: insert this seed if it qualifies
        // Skip if seed backtrack just saved a shorter version — don't overwrite
        const _seedPeak = _btSnapshots.length > 0 ? _btSnapshots[_btSnapshots.length - 1].tick : 0;
        if (!_seedBacktracked && _sweepSeedMoves && _sweepSeedMoves.size > 0 && _seedPeak > 0) {
            const maxSize = _goldenCouncilSize();
            // Peak = last snapshot tick (actual traversal), not _maxTickReached (stale after backtracking)
            const peak = _btSnapshots.length > 0 ? _btSnapshots[_btSnapshots.length - 1].tick : (result.maxTick || 0);
            const lowestPeak = _sweepGoldenCouncil.length >= maxSize
                ? _sweepGoldenCouncil[_sweepGoldenCouncil.length - 1].peak : -1;
            if (_sweepGoldenCouncil.length >= maxSize && peak <= lowestPeak) {
                console.log(`%c[GOLDEN COUNCIL] Seed ${seed} (peak t${peak}) rejected — below lowest council member (t${lowestPeak})`, 'color:#cc6666');
            } else {
                // _btSnapshots IS the traversal — serialize directly, no archive needed
                const snapsCopy = _btSnapshots.slice();
                // Dedup: if this seed already exists in council, update it instead of pushing a duplicate
                const existingMember = _sweepGoldenCouncil.find(m => m.seed === seed);
                if (existingMember) {
                    if (peak <= existingMember.peak) {
                        console.log(`%c[GOLDEN COUNCIL] Seed ${seed} (peak t${peak}) not longer than existing (t${existingMember.peak}) — skip`, 'color:#cc8866');
                    } else {
                        existingMember.peak = peak;
                        if (!existingMember.moves) existingMember.moves = _sweepSeedMoves;
                        else { for (const [tick, tickMap] of _sweepSeedMoves) { if (!existingMember.moves.has(tick)) existingMember.moves.set(tick, tickMap); } }
                        await _blIDBSaveCouncilMember(lvl, seed, snapsCopy, existingMember.moves);
                        console.log(`%c[GOLDEN COUNCIL] Seed ${seed} (peak t${peak}) updated in council [${_sweepGoldenCouncil.map(m => 't' + m.peak).join(', ')}]`, 'color:#ffcc00;font-weight:bold');
                    }
                } else {
                    _sweepGoldenCouncil.push({ peak, seed, moves: _sweepSeedMoves, _cold: true });
                    await _blIDBSaveCouncilMember(lvl, seed, _btSnapshots.slice(), _sweepSeedMoves);
                    _trimCouncil(lvl);
                    console.log(`%c[GOLDEN COUNCIL] Seed ${seed} (peak t${peak}) joined council [${_sweepGoldenCouncil.map(m => 't' + m.peak).join(', ')}] (${_btSnapshots.length} snapshots → cold)`, 'color:#ffcc00;font-weight:bold');
                }
            }
        }

        // Persist blacklist + council index to IndexedDB after each seed
        _blIDBSave(lvl);

        // Auto-retry worst: AFTER council save so we pick from the updated council
        // (not a member that was just evicted)
        const _arbChk = document.getElementById('chk-auto-retry-best');
        if (_arbChk && _arbChk.checked) {
            const _arMaxSize = _goldenCouncilSize();
            if (_sweepGoldenCouncil.length >= _arMaxSize && _sweepGoldenCouncil.length > 0) {
                const worst = _sweepGoldenCouncil[_sweepGoldenCouncil.length - 1];
                if (worst.peak > 0) {
                    _replayOnFirstSeed = worst;
                    console.log(`%c[AUTO-RETRY WORST] Next seed will replay worst council member (t${worst.peak}, seed 0x${worst.seed.toString(16).padStart(8,'0')})`, 'color:#ff9944;font-weight:bold');
                }
            }
            // else: council not full — fall through to new seed
        }

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
    _trimCouncil(lvl);
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
    // Peak = last snapshot tick (the actual traversal), not _maxTickReached
    // (which can be stale after backtracking)
    const peak = _btSnapshots.length > 0 ? _btSnapshots[_btSnapshots.length - 1].tick : (_demoTick || 0);
    const maxSize = _goldenCouncilSize();
    const lowestPeak = _sweepGoldenCouncil.length >= maxSize
        ? _sweepGoldenCouncil[_sweepGoldenCouncil.length - 1].peak : -1;
    // Clone moves so the live map can keep growing
    const movesCopy = new Map();
    for (const [tick, tickMap] of _sweepSeedMoves) {
        movesCopy.set(tick, new Map(tickMap));
    }
    // Existing seed already in council can always update
    const existingMember = _sweepGoldenCouncil.find(m => m.seed === seed);
    if (!existingMember && _sweepGoldenCouncil.length >= maxSize && peak <= lowestPeak) {
        console.log(`%c[SAVE] Current run (peak t${peak}) rejected — below lowest council member (t${lowestPeak})`, 'color:#cc6666');
    } else if (_sweepGoldenCouncil.length < maxSize || peak > lowestPeak || existingMember) {
        const slider = document.getElementById('lattice-slider');
        const lvl = slider ? +slider.value : 2;
        // _btSnapshots IS the traversal — serialize directly, no archive needed
        const snapsCopy = _btSnapshots.slice();
        if (existingMember) {
            // Only overwrite if new run is longer — don't clobber 899 with 852
            // after backtracker truncation. Peak is now correctly derived from
            // _btSnapshots (not stale _maxTickReached), so this comparison is valid.
            if (peak <= existingMember.peak) {
                console.log(`%c[SAVE] Current run (peak t${peak}) not longer than existing (t${existingMember.peak}) — skip`, 'color:#cc8866');
                return;
            }
            existingMember.peak = peak;
            existingMember.moves = movesCopy;
            _blIDBSaveCouncilMember(lvl, seed, snapsCopy, existingMember.moves);
        } else {
            const prevSeeds = new Set(_sweepGoldenCouncil.map(m => m.seed));
            _sweepGoldenCouncil.push({ peak, seed, moves: movesCopy, _cold: true });
            _blIDBSaveCouncilMember(lvl, seed, snapsCopy, movesCopy);
        }
        _trimCouncil(lvl);
        console.log(`%c[SAVE] Saved current run (seed 0x${seed.toString(16).padStart(8,'0')}, peak t${peak}) to council (${snapsCopy.length} snapshots → cold, ${existingMember ? 'updated' : 'new'})`, 'color:#66cc88;font-weight:bold');
    } else {
        console.log(`%c[SAVE] Current run (peak t${peak}) doesn't beat lowest council member (t${lowestPeak})`, 'color:#cc8866');
    }
}

// ── Council member replay — starts a sweep with the member's seed first ──
// Replay phase uses forced moves + guard suppression up to peak,
// then continues as a normal greedy sweep (blacklist, council, etc.)
async function startCouncilReplay(memberIdx) {
    // Council replay IS a sweep — it goes through the full sweep loop
    // so auto-retry-best, blacklisting, and seed iteration all work.
    const slider = document.getElementById('lattice-slider');
    const lvl = slider ? +slider.value : 2;
    await startSweepTest(lvl, memberIdx);
}
