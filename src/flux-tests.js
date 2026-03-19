// flux-tests.js — SPLIT INTO 5 FILES (2026-03-19)
//
// This file has been split into:
//   1. flux-tests-guards.js  — LIVE_GUARD_REGISTRY, guard dispatcher, guard helpers
//   2. flux-tests-unit.js    — Run-once unit tests (runDemo3Tests), BFS comparison
//   3. flux-tests-idb.js     — IDB persistence, blacklist buckets, serialization
//   4. flux-tests-sweep.js   — Sweep loop, council management, auto-retry
//   5. flux-tests-replay.js  — Replay pipeline, UI wiring, council dropdown
//
// Load order: guards -> unit -> idb -> sweep -> replay
// All files share global scope (no module system).
//
// This file is no longer loaded by flux-v2.html.
