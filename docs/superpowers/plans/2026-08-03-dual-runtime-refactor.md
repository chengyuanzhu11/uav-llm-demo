# UAV LLM Dual Runtime Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the UAV LLM demo so the same mission and agent layer can execute against an always-available HTML runtime by default or an independently selected AirSim validation runtime.

**Architecture:** `MissionService` owns mission state, the selected agent, the active `SimulationRuntime`, and the synchronization lock. `HtmlRuntime` and `AirSimRuntime` implement one execution contract; Flask only translates HTTP requests, while Three.js renders the unified state returned by the service.

**Tech Stack:** Python 3.14, Flask, `unittest`, vanilla JavaScript, Three.js, Cosys-AirSim Python client, PowerShell.

## Global Constraints

- Default runtime is `html`; application startup must not connect to port `41451`, launch Unreal, or import an AirSim client eagerly.
- Runtime choices are exactly `html` and `airsim`.
- Switching runtime resets the mission at the base; HTML mission progress is never copied into AirSim.
- A failed HTML-to-AirSim switch is atomic: retain the HTML runtime and existing mission state.
- The LLM remains a high-level decision maker. Runtime code executes validated trajectories and does not choose mission actions.
- Do not expose hidden chain-of-thought. Continue returning the existing concise `reason`, `risk`, and `expected_result` decision fields.
- Remove `/api/simulator/config`, `/api/simulator/stop`, and `/api/simulator/wind`; do not retain compatibility aliases.
- Remove response fields `simulator_backend`, `simulator`, and `logical_uav`; use `runtime_backend` and `runtime`.
- HTML wind requests return HTTP `409`; AirSim execution failures return HTTP `503` without committing the candidate mission state.
- Preserve the current DeepSeek and mock agent configuration behavior. Never write an API key to source, logs, or browser state.
- Keep `.idea/`, logs, virtual environments, generated artifacts, and process ID files outside version control.

## File Structure

- Create `uav_llm_demo/runtime.py`: runtime protocol, execution result, and runtime capability/error types.
- Create `uav_llm_demo/html_runtime.py`: deterministic browser-oriented trajectory executor with no external simulator dependency.
- Create `uav_llm_demo/mission_service.py`: mission ownership, agent arbitration, runtime switching, execution commit, and unified state serialization.
- Modify `uav_llm_demo/airsim_runtime.py`: conform the existing RPC worker to `SimulationRuntime` without changing its flight controller internals.
- Modify `uav_llm_demo/__init__.py`: thin Flask route adapters and AirSim runtime factory only.
- Modify `uav_llm_demo/templates/index.html`: segmented runtime selector and AirSim-only settings panel.
- Modify `uav_llm_demo/static/app.js`: consume unified runtime fields and call only the new runtime endpoints.
- Modify `uav_llm_demo/static/style.css`: selector, pending, hidden-panel, desktop, and mobile states.
- Modify `scripts/start_demo.ps1`: start Flask only.
- Retain `scripts/start_airsim.ps1`, `scripts/stop_demo.ps1`, and `scripts/verify_airsim.py` as explicit AirSim validation tools.
- Modify `README.md`: document default HTML startup, optional AirSim validation, new API, and limitations.
- Create `tests/test_runtime.py`: runtime contract and HTML execution tests.
- Create `tests/test_mission_service.py`: service lifecycle, switching, and commit tests.
- Modify `tests/test_environment.py`: keep environment, agent, and AirSim unit coverage; replace AirSim-only Flask tests.

---

### Task 1: Runtime Contract and HTML Runtime

**Files:**
- Create: `uav_llm_demo/runtime.py`
- Create: `uav_llm_demo/html_runtime.py`
- Create: `tests/test_runtime.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `RuntimeExecution`, `SimulationRuntime`, `RuntimeCapabilityError`, `RuntimeUnavailableError`.
- Produces: `HtmlRuntime.backend_name == "html"`, `reset(base)`, `execute(trajectory)`, `stop()`, `status()`, and `close()`.
- `RuntimeExecution.trajectory` is a tuple of point dictionaries; `trajectory_duration` is seconds; `kinematics` uses the existing environment payload shape.

- [ ] **Step 1: Add the runtime contract tests**

```python
from uav_llm_demo.html_runtime import HtmlRuntime


def test_html_runtime_is_connected_without_external_process():
    runtime = HtmlRuntime()
    assert runtime.backend_name == "html"
    assert runtime.status()["connected"] is True
    assert runtime.status()["supports_wind"] is False


def test_html_runtime_executes_the_planned_trajectory():
    runtime = HtmlRuntime()
    planned = [{"x": 1.0, "y": 18.0, "z": 2.0}, {"x": 2.0, "y": 18.0, "z": 2.0}]
    result = runtime.execute(planned)
    assert list(result.trajectory) == planned
    assert result.trajectory_duration > 0
    assert result.kinematics["source"] == "html"


def test_html_runtime_stop_and_close_are_idempotent():
    runtime = HtmlRuntime()
    runtime.stop()
    runtime.close()
    runtime.close()
    assert runtime.status()["connected"] is True
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
& 'C:\Users\kyrie\AppData\Local\Programs\Python\Python314\python.exe' -m unittest tests.test_runtime -v
```

Expected: import failure for `uav_llm_demo.html_runtime` or `uav_llm_demo.runtime`.

- [ ] **Step 3: Implement the runtime types**

```python
@dataclass(frozen=True)
class RuntimeExecution:
    trajectory: tuple[dict[str, float], ...]
    trajectory_duration: float
    kinematics: dict[str, object]


@runtime_checkable
class SimulationRuntime(Protocol):
    backend_name: str
    def reset(self, base: tuple[int, int]) -> dict[str, object]: ...
    def execute(self, trajectory: Sequence[Mapping[str, float]]) -> RuntimeExecution: ...
    def stop(self) -> None: ...
    def status(self) -> dict[str, object]: ...
    def close(self) -> None: ...
```

Define `RuntimeCapabilityError` for unsupported runtime features and `RuntimeUnavailableError` for failed external execution.

- [ ] **Step 4: Implement `HtmlRuntime`**

Copy incoming trajectory points into immutable result storage, compute duration from the existing `TRAJECTORY_DURATION_SECONDS`, and call the existing kinematics helper used by `UAVEnvironment`. Return `source: "html"`; `reset`, `stop`, and `close` must not depend on an external process.

- [ ] **Step 5: Run runtime tests and the existing environment suite**

Run:

```powershell
& 'C:\Users\kyrie\AppData\Local\Programs\Python\Python314\python.exe' -m unittest tests.test_runtime tests.test_environment -v
```

Expected: all tests pass; no network or AirSim process is required.

---

### Task 2: AirSim Adapter and Mission Service

**Files:**
- Create: `uav_llm_demo/mission_service.py`
- Create: `tests/test_mission_service.py`
- Modify: `uav_llm_demo/airsim_runtime.py`
- Modify: `uav_llm_demo/__init__.py`

**Interfaces:**
- Consumes: the runtime contract from Task 1 and existing `UAVEnvironment`, agent factory, and arbitration functions.
- Produces: `MissionService(agent=None, runtime=None, runtime_factory=None)`.
- Produces: `state(feedback=None)`, `step()`, `reset()`, `switch_runtime(backend, config)`, `stop_runtime()`, `set_wind(wind)`, and `configure_agent(config)`.
- Produces: `AirSimRuntime.execute(trajectory) -> RuntimeExecution`, `stop()`, and `close()` while retaining `set_wind()`.

- [ ] **Step 1: Write service lifecycle and execution tests**

```python
def test_service_defaults_to_html_runtime():
    service = MissionService(agent=MockLLMAgent())
    state = service.state()
    assert state["runtime_backend"] == "html"
    assert state["runtime"]["connected"] is True
    assert "simulator_backend" not in state
    assert "logical_uav" not in state


def test_step_commits_runtime_execution_result():
    runtime = RecordingRuntime("html")
    service = MissionService(agent=MockLLMAgent(), runtime=runtime)
    before = service.state()["step_count"]
    after = service.step()
    assert after["step_count"] == before + 1
    assert runtime.executions == 1


def test_failed_runtime_execution_does_not_commit_candidate_state():
    service = MissionService(agent=MockLLMAgent(), runtime=FailingRuntime())
    before = service.state()
    with pytest.raises(RuntimeUnavailableError):
        service.step()
    after = service.state()
    assert after["uav"] == before["uav"]
    assert after["step_count"] == before["step_count"]
```

Use `unittest.TestCase.assertRaises` if the suite remains pure `unittest`; do not add pytest as a dependency.

- [ ] **Step 2: Run service tests and verify RED**

Run:

```powershell
& 'C:\Users\kyrie\AppData\Local\Programs\Python\Python314\python.exe' -m unittest tests.test_mission_service -v
```

Expected: import failure for `MissionService`.

- [ ] **Step 3: Adapt `AirSimRuntime` to the common contract**

- Set `backend_name = "airsim"`.
- Wrap the existing `fly_trajectory()` result in `RuntimeExecution` from `execute()`.
- Make `reset(base)` perform reset, API control, arming, takeoff, base positioning, and telemetry refresh.
- Make `stop()` cancel the current worker command and hover.
- Make `close()` release API control and close the worker exactly once.
- Add `supports_wind: True` and `source: "airsim"` to `status()`.
- Keep NED conversion, actual-position rebasing, segment caps, arrival stabilization, and six-degree-of-freedom telemetry unchanged.

- [ ] **Step 4: Implement `MissionService` ownership and commit logic**

Move `_validated_decision` and related goal-distance arbitration helpers out of the Flask module. Under one `RLock`, clone the environment, apply the validated action to the candidate, call `runtime.execute(candidate.last_trajectory)`, apply the returned actual trajectory/kinematics to the candidate, and only then replace the live environment.

The unified state payload must contain:

```python
payload["runtime_backend"] = self.runtime.backend_name
payload["runtime"] = self.runtime.status()
```

When AirSim reports `position_grid`, render that actual position as `payload["uav"]`; do not publish a second logical position field.

- [ ] **Step 5: Add atomic switch tests**

```python
def test_switch_to_airsim_resets_mission_after_candidate_connects():
    created = RecordingRuntime("airsim")
    service = moved_html_service(runtime_factory=lambda backend, config: created)
    state = service.switch_runtime("airsim", {"host": "127.0.0.1", "port": 41451})
    assert state["runtime_backend"] == "airsim"
    assert state["uav"] == state["base"]
    assert state["step_count"] == 0


def test_failed_airsim_switch_preserves_html_runtime_and_mission():
    service = moved_html_service(runtime_factory=failing_factory)
    before = service.state()
    with self.assertRaises(RuntimeUnavailableError):
        service.switch_runtime("airsim", {})
    after = service.state()
    assert after["runtime_backend"] == "html"
    assert after["uav"] == before["uav"]
    assert after["step_count"] == before["step_count"]


def test_switch_back_to_html_closes_airsim_and_resets():
    airsim = RecordingRuntime("airsim")
    service = MissionService(agent=MockLLMAgent(), runtime=airsim)
    state = service.switch_runtime("html", {})
    assert airsim.closed is True
    assert state["runtime_backend"] == "html"
    assert state["uav"] == state["base"]
```

- [ ] **Step 6: Run service and AirSim unit tests**

Run:

```powershell
& 'C:\Users\kyrie\AppData\Local\Programs\Python\Python314\python.exe' -m unittest tests.test_mission_service tests.test_environment -v
```

Expected: all tests pass without a live Unreal process.

---

### Task 3: Replace the Flask API Surface

**Files:**
- Modify: `uav_llm_demo/__init__.py`
- Modify: `tests/test_environment.py`

**Interfaces:**
- Consumes: one `MissionService` stored at `app.extensions["mission_service"]`.
- Produces: `GET /api/state`, `POST /api/step`, `POST /api/reset`, `POST /api/runtime`, `POST /api/runtime/stop`, `POST /api/runtime/wind`, and `POST /api/config`.
- `POST /api/runtime` accepts `{"backend":"html"}` or `{"backend":"airsim","config":{"host":"127.0.0.1","port":41451,"vehicle_name":"","wind":{"x":0,"y":0,"z":0}}}`.

- [ ] **Step 1: Replace Flask tests with the new contract**

```python
def test_app_starts_in_html_without_constructing_airsim():
    app = create_app(runtime_factory=forbidden_factory)
    response = app.test_client().get("/api/state")
    assert response.status_code == 200
    assert response.get_json()["runtime_backend"] == "html"


def test_old_simulator_routes_are_removed():
    client = create_app().test_client()
    for path in ("/api/simulator/config", "/api/simulator/stop", "/api/simulator/wind"):
        assert client.post(path, json={}).status_code == 404


def test_html_wind_is_an_unsupported_capability():
    response = create_app().test_client().post("/api/runtime/wind", json={"x": 1, "y": 0, "z": 0})
    assert response.status_code == 409
    assert response.get_json()["error"].startswith("Runtime error:")


def test_runtime_execution_failure_returns_503_without_commit():
    app = create_app(service=MissionService(agent=MockLLMAgent(), runtime=FailingRuntime()))
    client = app.test_client()
    before = client.get("/api/state").get_json()
    response = client.post("/api/step")
    after = client.get("/api/state").get_json()
    assert response.status_code == 503
    assert after["step_count"] == before["step_count"]
```

- [ ] **Step 2: Run the focused Flask tests and verify RED**

Run the named new test class with `python -m unittest ... -v`.

Expected: failures because the old simulator routes and response fields still exist.

- [ ] **Step 3: Make Flask a thin adapter**

Use this application factory shape:

```python
def create_app(*, service: MissionService | None = None, runtime_factory=None) -> Flask:
    app = Flask(__name__)
    mission_service = service or MissionService(runtime_factory=runtime_factory or _build_runtime)
    app.extensions["mission_service"] = mission_service
```

Parse request JSON in the route, delegate once to `MissionService`, and map exceptions consistently:

```python
except RuntimeCapabilityError as exc:
    return jsonify(error=f"Runtime error: {exc}"), 409
except RuntimeUnavailableError as exc:
    return jsonify(error=f"Runtime error: {exc}"), 503
except (TypeError, ValueError) as exc:
    return jsonify(error=str(exc)), 400
```

Delete Flask-level environment holders, runtime holders, arbitration logic, AirSim reset code, and all `/api/simulator/*` decorators.

- [ ] **Step 4: Verify API tests and scan removed names**

Run:

```powershell
& 'C:\Users\kyrie\AppData\Local\Programs\Python\Python314\python.exe' -m unittest discover -s tests -v
rg -n "simulator_backend|logical_uav|/api/simulator" uav_llm_demo tests
```

Expected: tests pass and `rg` returns no matches.

---

### Task 4: Same-Page Runtime Selector and Conditional Controls

**Files:**
- Modify: `uav_llm_demo/templates/index.html`
- Modify: `uav_llm_demo/static/app.js`
- Modify: `uav_llm_demo/static/style.css`

**Interfaces:**
- Consumes: `state.runtime_backend`, `state.runtime`, and the Task 3 endpoints.
- Produces: `switchRuntime(backend)`, `setRuntimePending(pending)`, and runtime-aware rendering.

- [ ] **Step 1: Add static contract assertions before editing production files**

Add tests that read the template and script and assert:

```python
assert 'data-runtime="html"' in html
assert 'data-runtime="airsim"' in html
assert 'id="airsimPanel"' in html
assert '"/api/runtime"' in javascript
assert '"/api/runtime/stop"' in javascript
assert '"/api/runtime/wind"' in javascript
assert "/api/simulator/" not in javascript
assert "simulator_backend" not in javascript
```

- [ ] **Step 2: Run the static tests and verify RED**

Expected: missing runtime selector and remaining old endpoint references.

- [ ] **Step 3: Add the segmented selector and conditional AirSim panel**

Add a two-option `role="radiogroup"` control labelled Runtime. Use `data-runtime="html"` and `data-runtime="airsim"`; set `aria-checked`, `aria-busy`, and disabled states from JavaScript. Wrap host, port, vehicle, wind, and six-degree-of-freedom controls in `#airsimPanel`, hidden whenever the active backend is HTML.

- [ ] **Step 4: Replace frontend runtime calls**

Implement switching with an explicit mission-reset confirmation, stop auto-run first, and POST the selected backend to `/api/runtime`. For AirSim include the form configuration; if the call fails, render the server error and restore the selector to `state.runtime_backend`.

Replace pause/stop with `/api/runtime/stop`, wind with `/api/runtime/wind`, and all rendering reads with `state.runtime`. Keep Step, Auto Run, Reset, trajectory animation, agent log, and camera behavior intact.

- [ ] **Step 5: Add responsive selector styling**

Use fixed-height segmented buttons, a clear selected state, and `hidden`/`.is-hidden` rules that remove the AirSim panel from layout. At mobile widths, stack runtime settings without allowing labels, buttons, canvas, or logs to overlap. Do not add gradients, decorative cards, or viewport-scaled font sizes.

- [ ] **Step 6: Run tests and browser smoke checks**

Run the unit suite, start Flask, and verify at `http://127.0.0.1:5000`:

- Desktop `1440x900`: HTML selected, AirSim panel absent from layout, canvas and log visible.
- Mobile `390x844`: no horizontal overflow or overlap.
- Step and Auto Run advance in HTML without Unreal.
- Pause stops automatic stepping.
- AirSim selection warns about reset; a failed connection restores HTML and preserves the prior mission.

Capture desktop and mobile screenshots in `artifacts/visual-check/` and inspect them at original resolution.

---

### Task 5: Startup, Documentation, and Redundant Code Removal

**Files:**
- Modify: `scripts/start_demo.ps1`
- Modify: `scripts/stop_demo.ps1`
- Modify: `README.md`
- Modify: `.gitignore`
- Delete only code proven unreachable by the new runtime/service architecture.

**Interfaces:**
- Produces: `scripts/start_demo.ps1` that starts Flask and prints the browser URL without launching Blocks.
- Retains: `scripts/start_airsim.ps1` and `scripts/verify_airsim.py` for explicit validation.

- [ ] **Step 1: Add startup static assertions**

```python
start_demo = Path("scripts/start_demo.ps1").read_text(encoding="utf-8")
assert "start_airsim.ps1" not in start_demo
assert "/api/simulator/config" not in start_demo
assert "41451" not in start_demo
```

- [ ] **Step 2: Run startup assertions and verify RED**

Expected: the current script still launches AirSim and posts the old configuration endpoint.

- [ ] **Step 3: Simplify process scripts**

Make `start_demo.ps1` start only the Flask process at `127.0.0.1:5000`, preserve hidden-window logging and PID behavior, and print the URL plus the command for optional AirSim startup. Make `stop_demo.ps1` reliably stop Flask; retain explicit Blocks cleanup only when it identifies the packaged AirSim process by executable path or process name.

- [ ] **Step 4: Rewrite README around dual runtimes**

Document:

- `HTML` is default and needs no Unreal or API key when mock agent is selected.
- DeepSeek agent selection remains independent from runtime selection.
- AirSim is optional final validation and is launched separately with `scripts/start_airsim.ps1`.
- Runtime switching resets the mission.
- New REST endpoints and response names.
- AirSim provides higher-fidelity validation but is not a perfect model of reality.

- [ ] **Step 5: Delete confirmed redundancy and scan**

Delete old route handlers, old holder dictionaries, duplicate AirSim reset orchestration, obsolete frontend functions such as `configureSimulator`, and stale README claims that AirSim is the only executor. Keep environment planning, dynamic obstacles, DeepSeek/mock agents, AirSim NED mapping, six-degree-of-freedom telemetry, and verification scripts.

Run:

```powershell
rg -n "simulator_backend|logical_uav|/api/simulator|current production executor only supports AirSim" . --glob '!docs/superpowers/**' --glob '!.git/**'
```

Expected: no matches in active code or documentation.

---

### Task 6: Full Verification and Live AirSim Validation

**Files:**
- Modify only files required by failures reproduced in this task.

**Interfaces:**
- Verifies the complete application contract; produces no new public API.

- [ ] **Step 1: Run the complete automated suite**

```powershell
& 'C:\Users\kyrie\AppData\Local\Programs\Python\Python314\python.exe' -m unittest discover -s tests -v
```

Expected: all runtime, environment, agent, AirSim adapter, service, Flask, static frontend, and startup tests pass.

- [ ] **Step 2: Verify default startup without AirSim**

Stop existing demo processes, run `scripts/start_demo.ps1`, verify port `5000` is listening, verify port `41451` was not opened by the script, and call:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/state
```

Expected: `runtime_backend` is `html`, `runtime.connected` is true, and one `/api/step` request succeeds.

- [ ] **Step 3: Verify browser behavior on desktop and mobile**

Inspect screenshots and console output. Confirm no overlap, no horizontal overflow, a nonblank moving Three.js scene, correct runtime selection, working Step/Auto/Pause/Reset controls, and readable `Observation -> rationale -> Action JSON -> feedback` logs.

- [ ] **Step 4: Run optional live AirSim validation**

Start Blocks with `scripts/start_airsim.ps1`, select AirSim in the browser, and execute three steps. Confirm actual AirSim position/attitude/velocity updates, no continuous oscillation, and no mission commit on a forced runtime error. Switch back to HTML and confirm AirSim API control is released and the mission resets at base.

- [ ] **Step 5: Final security and cleanup scan**

```powershell
rg -n --hidden --glob '!.git/**' "sk-[A-Za-z0-9_-]{12,}" .
git status --short
```

Expected: no real API key, no generated logs/screenshots staged, `.idea/` ignored, and only intended source, tests, scripts, docs, and configuration changes remain.

