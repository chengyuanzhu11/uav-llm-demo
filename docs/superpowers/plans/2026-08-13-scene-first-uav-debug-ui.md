# Scene-First UAV Debug UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current crowded right-console layout with a desktop-first, scene-dominant UAV debugging workbench that exposes a structured four-stage decision trace and truthful runtime states without changing backend behavior.

**Architecture:** Keep the existing Flask APIs, Three.js scene, serialized Auto Run loop, and dual-runtime lifecycle. Rebuild only the browser presentation layer: a semantic toolbar and settings dialog in HTML, a two-width CSS grid, and focused JavaScript renderers for telemetry, trace, history, runtime state, and settings. Runtime cancellation semantics remain unchanged and the UI distinguishes an AirSim request cancelled before commit from an HTML step paused after commit.

**Tech Stack:** Flask/Jinja template, vanilla ES modules, Three.js, local Lucide 1.31.0 UMD bundle, CSS Grid, Python `unittest`, browser-based desktop verification.

---

## File Map And Parallel Boundaries

- Modify `tests/test_frontend_contract.py`: static frontend contracts and regression guards.
- Modify `uav_llm_demo/templates/index.html`: semantic workbench DOM and settings dialog.
- Create `uav_llm_demo/static/vendor/lucide.min.js`: offline Lucide browser bundle.
- Replace `uav_llm_demo/static/styles.css`: desktop workbench layout and state styles.
- Modify `uav_llm_demo/static/app.js`: operation state, structured renderers, inspector/history/settings interactions.

After Task 1 establishes a failing contract, Tasks 2, 3, and 4 may run concurrently because their write sets do not overlap. The parent agent owns integration, tests, browser verification, and final commits.

### Task 1: Lock The New Frontend Contract

**Files:**
- Modify: `tests/test_frontend_contract.py`
- Test: `tests/test_frontend_contract.py`

- [ ] **Step 1: Add failing semantic layout tests**

Add these methods to `FrontendContractTest`:

```python
def test_scene_first_workbench_structure_exists(self):
    html = (ROOT / "uav_llm_demo/templates/index.html").read_text(
        encoding="utf-8"
    )

    required = (
        'class="app-toolbar"',
        'id="decisionInspector"',
        'id="inspectorToggleBtn"',
        'id="settingsBtn"',
        'id="settingsDialog"',
        'id="trajectoryBtn"',
        'id="operationStatus"',
        'data-stage="observation"',
        'data-stage="agent"',
        'data-stage="guard"',
        'data-stage="execution"',
        'id="historyTableBody"',
    )
    for fragment in required:
        with self.subTest(fragment=fragment):
            self.assertIn(fragment, html)


def test_icons_are_local_and_accessible(self):
    html = (ROOT / "uav_llm_demo/templates/index.html").read_text(
        encoding="utf-8"
    )
    icon_bundle = ROOT / "uav_llm_demo/static/vendor/lucide.min.js"

    self.assertTrue(icon_bundle.is_file())
    self.assertIn("vendor/lucide.min.js", html)
    self.assertNotIn("unpkg.com", html)
    self.assertNotIn("cdn.jsdelivr.net", html)
    self.assertIn('aria-label="Settings"', html)
    self.assertIn('aria-label="Expand decision inspector"', html)
```

- [ ] **Step 2: Add failing JavaScript state/rendering tests**

Add:

```python
def test_javascript_exposes_focused_workbench_renderers(self):
    javascript = (ROOT / "uav_llm_demo/static/app.js").read_text(
        encoding="utf-8"
    )

    for function_name in (
        "renderToolbarState",
        "renderTelemetry",
        "renderDecisionTrace",
        "renderHistory",
        "renderRuntimeSettings",
        "setOperationState",
        "toggleInspectorWidth",
        "showHistoryView",
    ):
        self.assertIn(f"function {function_name}", javascript)
    self.assertIn('sessionStorage.getItem("uav-inspector-expanded")', javascript)
    self.assertIn('requestJson("/api/runtime/stop"', javascript)
    self.assertNotIn("setInterval(", javascript)


def test_styles_define_two_width_inspector_and_operation_states(self):
    styles = (ROOT / "uav_llm_demo/static/styles.css").read_text(
        encoding="utf-8"
    )

    for selector in (
        ".simulator.inspector-expanded",
        ".decision-inspector",
        '.simulator[data-operation="executing"]',
        '.simulator[data-operation="paused"]',
        '.simulator[data-operation="error"]',
        ".settings-dialog",
        ".history-table",
    ):
        self.assertIn(selector, styles)
    self.assertIn("minmax(340px, 28%)", styles)
    self.assertIn("minmax(400px, 34%)", styles)
```

- [ ] **Step 3: Run the contract test and verify RED**

Run:

```powershell
& 'C:\Users\kyrie\AppData\Local\Programs\Python\Python314\python.exe' `
  -m unittest tests.test_frontend_contract -v
```

Expected: the existing four tests pass and the four new tests fail because the new DOM, bundle, renderers, and CSS selectors do not exist.

### Task 2: Build The Semantic HTML Workbench And Local Icons

**Files:**
- Modify: `uav_llm_demo/templates/index.html`
- Create: `uav_llm_demo/static/vendor/lucide.min.js`
- Test: `tests/test_frontend_contract.py`

- [ ] **Step 1: Vendor Lucide 1.31.0 locally**

Download the exact browser bundle without adding a runtime package manager:

```powershell
Invoke-WebRequest `
  -Uri 'https://unpkg.com/lucide@1.31.0/dist/umd/lucide.min.js' `
  -OutFile 'uav_llm_demo\static\vendor\lucide.min.js'
```

Verify the file is non-empty and contains `createIcons`:

```powershell
rg -n "createIcons" uav_llm_demo\static\vendor\lucide.min.js
```

- [ ] **Step 2: Replace the page shell with the scene-first semantic structure**

Keep the import map and existing element IDs used by Three.js/API behavior. The body structure must use this exact hierarchy and IDs:

```html
<main id="simulator" class="simulator" data-operation="ready">
  <header class="app-toolbar">
    <div class="brand-block">...</div>
    <label class="toolbar-field" for="scenarioSelect">...</label>
    <div class="runtime-selector" role="radiogroup" aria-label="Simulation runtime">...</div>
    <span id="runtimeStatus" class="runtime-status" aria-live="polite"></span>
    <div class="toolbar-spacer"></div>
    <button id="settingsBtn" class="icon-button" aria-label="Settings" title="Settings">
      <i data-lucide="settings"></i>
    </button>
    <button id="resetBtn" class="command-button secondary"><i data-lucide="rotate-ccw"></i><span>Reset</span></button>
    <button id="autoBtn" class="command-button secondary"><i data-lucide="play"></i><span>Auto</span></button>
    <button id="stepBtn" class="command-button primary"><i data-lucide="step-forward"></i><span>Step</span></button>
  </header>

  <section class="scene-workspace">
    <div id="sceneContainer" class="scene-container" aria-label="3D UAV simulation"></div>
    <section class="scene-telemetry" aria-label="Flight telemetry">...</section>
    <div class="scene-tools">
      <button id="overviewBtn" class="icon-button" aria-label="Overview camera" title="Overview camera"><i data-lucide="scan"></i></button>
      <button id="followBtn" class="icon-button" aria-label="Follow UAV" title="Follow UAV"><i data-lucide="crosshair"></i></button>
      <button id="trajectoryBtn" class="icon-button active" aria-label="Toggle trajectory" title="Toggle trajectory"><i data-lucide="route"></i></button>
    </div>
    <div id="sceneExecutionStatus" class="scene-execution-status"></div>
  </section>

  <aside id="decisionInspector" class="decision-inspector" aria-label="Decision inspector">
    <header class="inspector-header">
      <span id="currentStepValue" class="step-index">#0</span>
      <div><h1>Decision Trace</h1><p><span id="backendValue">--</span> · <span id="phaseValue">--</span></p></div>
      <button id="inspectorToggleBtn" class="icon-button" aria-label="Expand decision inspector" title="Expand decision inspector"><i data-lucide="panel-right-open"></i></button>
    </header>
    <div id="operationBanner" class="operation-banner"><span id="operationStatus">Ready</span></div>
    <div id="traceView" class="inspector-view">...</div>
    <div id="historyView" class="inspector-view is-hidden">...</div>
    <footer class="recent-history">...</footer>
  </aside>

  <dialog id="settingsDialog" class="settings-dialog" aria-labelledby="settingsTitle">...</dialog>
</main>
<script src="{{ url_for('static', filename='vendor/lucide.min.js') }}"></script>
<script type="module" src="{{ url_for('static', filename='app.js') }}"></script>
```

Inside `traceView`, provide four `<section class="trace-stage" data-stage="...">` elements and all IDs consumed by Task 4:

```text
goalDirectionValue, goalDistanceValue, staticRiskValue, dynamicRiskValue,
observationText, agentActionValue, agentWaypointValue, agentRiskValue,
agentSkillValue, agentReasonValue, decisionJson, proposedActionValue,
executedActionValue, guardStatus, guardReasonValue, executionPositionValue,
executionRuntimeValue, executionDurationValue, executionBatteryValue,
feedbackText, kinematicsValue
```

Inside `historyView`, include `historyBackBtn`, `historyTableBody`, and `historyEmpty`. Keep `historyList`, `historyViewBtn`, `stepCount`, `metricsValue`, and `missionBadge` in the Inspector.

Move the existing AirSim fields unchanged into `settingsDialog` under `airsimPanel`. Add `closeSettingsBtn` and keep `vehicleName`, `airsimHost`, `airsimPort`, `reconnectAirSimBtn`, `windNorth`, `windEast`, `windDown`, and `applyWindBtn`.

- [ ] **Step 3: Run the HTML/icon contract subset**

Run:

```powershell
& 'C:\Users\kyrie\AppData\Local\Programs\Python\Python314\python.exe' `
  -m unittest `
  tests.test_frontend_contract.FrontendContractTest.test_scene_first_workbench_structure_exists `
  tests.test_frontend_contract.FrontendContractTest.test_icons_are_local_and_accessible -v
```

Expected: PASS.

### Task 3: Implement The Desktop Scene-First Layout

**Files:**
- Modify: `uav_llm_demo/static/styles.css`
- Test: `tests/test_frontend_contract.py`

- [ ] **Step 1: Replace legacy overlay rules with stable desktop grid tokens**

Define neutral tokens and the two workbench columns:

```css
:root {
  color-scheme: dark;
  --surface-0: #0d1416;
  --surface-1: #121a1d;
  --surface-2: #182225;
  --line: #334247;
  --text: #e5edeb;
  --muted: #879a97;
  --cyan: #45d5d2;
  --green: #62d57b;
  --amber: #f2b84b;
  --red: #ff6b57;
  --toolbar-height: 52px;
}

.simulator {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(340px, 28%);
  grid-template-rows: var(--toolbar-height) minmax(0, 1fr);
  height: 100vh;
  overflow: hidden;
}

.simulator.inspector-expanded {
  grid-template-columns: minmax(0, 1fr) minmax(400px, 34%);
}
```

- [ ] **Step 2: Style the toolbar, scene, telemetry, inspector, and settings dialog**

Use full-width bands rather than nested cards. Required behavior:

```css
.app-toolbar { grid-column: 1 / -1; }
.scene-workspace { min-height: 0; min-width: 0; position: relative; }
.scene-container { height: 100%; width: 100%; }
.decision-inspector { min-height: 0; overflow: hidden; }
.inspector-view { min-height: 0; overflow-y: auto; }
.scene-telemetry { position: absolute; left: 14px; top: 14px; }
.scene-tools { bottom: 14px; left: 14px; position: absolute; }
.scene-execution-status { bottom: 14px; position: absolute; right: 14px; }
.history-table { border-collapse: collapse; table-layout: fixed; width: 100%; }
.settings-dialog { border: 1px solid var(--line); max-height: 80vh; width: min(520px, calc(100vw - 48px)); }
```

Use status color only with a text/icon label. Add explicit selectors for:

```css
.simulator[data-operation="executing"] .operation-banner { ... }
.simulator[data-operation="paused"] .operation-banner { ... }
.simulator[data-operation="error"] .operation-banner { ... }
.trace-stage[data-status="pass"] .stage-status { ... }
.trace-stage[data-status="override"] .stage-status { ... }
```

Retain a basic `@media (max-width: 980px)` fallback but do not optimize it beyond preventing overflow.

- [ ] **Step 3: Run the CSS contract test**

Run:

```powershell
& 'C:\Users\kyrie\AppData\Local\Programs\Python\Python314\python.exe' `
  -m unittest `
  tests.test_frontend_contract.FrontendContractTest.test_styles_define_two_width_inspector_and_operation_states -v
```

Expected: PASS.

### Task 4: Implement Truthful UI State And Structured Trace Rendering

**Files:**
- Modify: `uav_llm_demo/static/app.js`
- Test: `tests/test_frontend_contract.py`

- [ ] **Step 1: Cache the new DOM and initialize UI-only state**

Keep all existing scene variables. Add references for every Task 2 ID plus:

```javascript
const simulator = document.getElementById("simulator");
const settingsBtn = document.getElementById("settingsBtn");
const settingsDialog = document.getElementById("settingsDialog");
const closeSettingsBtn = document.getElementById("closeSettingsBtn");
const inspectorToggleBtn = document.getElementById("inspectorToggleBtn");
const trajectoryBtn = document.getElementById("trajectoryBtn");
const traceView = document.getElementById("traceView");
const historyView = document.getElementById("historyView");
const historyTableBody = document.getElementById("historyTableBody");

let operationState = "ready";
let operationMessage = "Ready";
let historyMode = false;
let selectedHistoryStep = null;
let trajectoryVisible = true;
```

Read `sessionStorage.getItem("uav-inspector-expanded") === "true"`, apply `inspector-expanded`, and call `window.lucide.createIcons()` after initial DOM setup.

- [ ] **Step 2: Add focused renderers**

Implement these exact functions:

```javascript
function renderToolbarState() { /* labels, icons, disabled controls */ }
function renderTelemetry() { /* position, altitude, battery, phase, threat */ }
function renderDecisionTrace(entry = state.history[0] || null) { /* four stages */ }
function renderHistory() { /* recent three plus full table */ }
function renderRuntimeSettings() { /* runtime fields, AirSim conditional panel */ }
function setOperationState(nextState, message) { /* body dataset + render */ }
function toggleInspectorWidth() { /* class, aria-label, icon, sessionStorage */ }
function showHistoryView(show) { /* trace/history visibility without state mutation */ }
```

Observation values come from `state.observation` for step zero and from the latest state snapshot when a history entry contains only `observation` text. Use `--` for missing structured fields. Agent values come from `entry.decision`. Guard rendering must compare `proposed_action || action` with `action`, show `PASS` or `OVERRIDE`, and never hide `safety_intervention`.

- [ ] **Step 3: Refactor command state without changing runtime semantics**

At the start of `stepSimulation`, call:

```javascript
setOperationState("executing", `${state.runtime_backend} · executing`);
```

On success, commit the returned state, start the existing visual trajectory, and set `ready` unless Auto remains active. On failure, set `error` with the response message and stop future Auto iterations.

`stopAuto(true)` must continue posting `/api/runtime/stop`. After the response:

- if an in-flight AirSim request failed/cancelled before state commit, show `Paused · current action cancelled`;
- if the HTML response was already committed, show `Paused after step #N` and do not decrement or rewrite `step_count`.

Do not add `setInterval`; retain the existing serialized `while (autoRunning)` loop.

- [ ] **Step 4: Add inspector, history, settings, and trajectory interactions**

Register:

```javascript
inspectorToggleBtn.addEventListener("click", toggleInspectorWidth);
settingsBtn.addEventListener("click", () => settingsDialog.showModal());
closeSettingsBtn.addEventListener("click", () => settingsDialog.close());
historyViewBtn.addEventListener("click", () => showHistoryView(true));
historyBackBtn.addEventListener("click", () => showHistoryView(false));
trajectoryBtn.addEventListener("click", () => {
  trajectoryVisible = !trajectoryVisible;
  if (pathLine) pathLine.visible = trajectoryVisible;
  trajectoryBtn.classList.toggle("active", trajectoryVisible);
  trajectoryBtn.setAttribute("aria-pressed", String(trajectoryVisible));
});
```

History table rows must be buttons or keyboard-focusable rows. Selecting a row renders that entry read-only without changing `state`, then the Back control restores the current trace.

- [ ] **Step 5: Run JavaScript checks and the renderer contract**

Run:

```powershell
& 'C:\Program Files\nodejs\node.exe' --check uav_llm_demo\static\app.js
& 'C:\Users\kyrie\AppData\Local\Programs\Python\Python314\python.exe' `
  -m unittest `
  tests.test_frontend_contract.FrontendContractTest.test_javascript_exposes_focused_workbench_renderers -v
```

Expected: both commands exit 0.

### Task 5: Integrate, Review, And Commit The Workbench

**Files:**
- Modify only if integration fixes are required: the five files listed in File Map.

- [ ] **Step 1: Run the complete frontend contract suite**

```powershell
& 'C:\Users\kyrie\AppData\Local\Programs\Python\Python314\python.exe' `
  -m unittest tests.test_frontend_contract -v
```

Expected: all existing and new frontend tests pass.

- [ ] **Step 2: Run the complete Python suite and JavaScript syntax check**

```powershell
& 'C:\Users\kyrie\AppData\Local\Programs\Python\Python314\python.exe' `
  -m unittest discover -s tests -v
& 'C:\Program Files\nodejs\node.exe' --check uav_llm_demo\static\app.js
```

Expected: 0 failures and JavaScript exit 0.

- [ ] **Step 3: Review the diff for scope and secrets**

```powershell
git diff --stat 3b5e388
git diff --check
git diff | rg "sk-[A-Za-z0-9_-]{16,}"
```

Expected: only planned frontend/test/vendor/plan files; no API key matches. Ignore pre-existing whitespace inside untouched Three.js bundles, but do not introduce new whitespace errors in changed source.

- [ ] **Step 4: Commit the implementation**

```powershell
git add tests/test_frontend_contract.py `
  uav_llm_demo/templates/index.html `
  uav_llm_demo/static/styles.css `
  uav_llm_demo/static/app.js `
  uav_llm_demo/static/vendor/lucide.min.js `
  docs/superpowers/plans/2026-08-13-scene-first-uav-debug-ui.md
git commit -m "feat: add scene-first UAV debug workbench"
```

### Task 6: Verify Real Desktop Behavior And Fix Visual Regressions

**Files:**
- Modify only if browser evidence exposes a defect: `index.html`, `styles.css`, `app.js`, `test_frontend_contract.py`.

- [ ] **Step 1: Start or restart the current Flask app**

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1
```

Expected: `http://127.0.0.1:5000` returns the current workbench in HTML Runtime.

- [ ] **Step 2: Verify three desktop viewports**

Use browser automation at `1280x720`, `1440x900`, and `1920x1080`. Capture screenshots and verify:

- canvas pixels are nonblank;
- toolbar, HUD, scene controls, and Inspector do not overlap;
- no text escapes buttons or table cells;
- Inspector toggles between 28% and 34% without blanking the canvas;
- no horizontal page scroll;
- browser console has no errors.

- [ ] **Step 3: Verify the user workflow**

In HTML Runtime:

1. Click Step and verify the four stages update.
2. Start Auto, then Pause; verify no second request starts after Pause.
3. Open history, select an older step, and return without changing `step_count`.
4. Open Settings; verify AirSim fields are hidden in HTML mode.
5. Attempt AirSim switching while Blocks is unavailable; verify HTML selection and mission state remain.

If Blocks is already running, also verify successful AirSim switching and one real step. Do not start AirSim solely for UI acceptance unless needed to reproduce a defect.

- [ ] **Step 4: Add a regression test before each browser-discovered fix**

For every observed defect, first add the smallest failing assertion to `tests/test_frontend_contract.py`, run it RED, apply the minimal HTML/CSS/JS fix, then run it GREEN.

- [ ] **Step 5: Run final verification and commit browser fixes**

```powershell
& 'C:\Users\kyrie\AppData\Local\Programs\Python\Python314\python.exe' `
  -m unittest discover -s tests -v
& 'C:\Program Files\nodejs\node.exe' --check uav_llm_demo\static\app.js
git status --short
```

Expected: all tests pass, syntax exit 0, and only intentional browser-fix files are modified. If fixes were required:

```powershell
git add tests/test_frontend_contract.py uav_llm_demo/templates/index.html `
  uav_llm_demo/static/styles.css uav_llm_demo/static/app.js
git commit -m "fix: polish UAV workbench interactions"
```

## Plan Self-Review

- Every requirement in the approved design maps to Tasks 2-6.
- Backend APIs, Agent logic, mission arbitration, HTML Runtime, and AirSim Runtime are explicitly out of the write set.
- HTML/CSS/JS parallel workers have disjoint files.
- Runtime cancellation language distinguishes pre-commit AirSim cancellation from post-commit HTML playback.
- The plan contains no deferred implementation markers, CDN runtime dependency, mobile redesign, or hidden-chain-of-thought requirement.
