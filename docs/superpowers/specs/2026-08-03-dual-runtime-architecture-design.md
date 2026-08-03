# UAV LLM Dual Runtime Architecture Design

## Objective

Restructure the UAV LLM demo so the LLM Agent and mission logic select one of
two independent execution runtimes:

```text
                 LLM Agent
                     |
              Mission Service
                     |
           SimulationRuntime
              /            \
     HtmlRuntime         AirSimRuntime
     default demo        final validation
```

The HTML runtime is the default and must operate without AirSim. AirSim is an
optional final validation environment selected from the same browser page.
AirSim must not consume or synchronize a trajectory that HTML already
executed.

## Confirmed Product Behavior

- The browser exposes a two-option runtime selector:
  `HTML Simulation` and `AirSim Validation`.
- Application startup selects HTML and does not require port `41451` or a
  running Unreal process.
- Switching runtimes always stops Auto Run first.
- Switching from HTML to AirSim connects a candidate AirSim runtime, resets
  AirSim, takes off, and resets the mission to the base. It does not continue
  from the HTML mission position.
- Switching from AirSim to HTML releases AirSim API control, creates a new
  HTML runtime, and resets the mission to the base.
- A failed AirSim switch leaves the HTML runtime and current HTML mission
  state unchanged.
- The Three.js page renders whichever runtime is active. It is not an
  execution stage in front of AirSim.

## Components

### MissionService

`MissionService` is the sole owner of:

- `UAVEnvironment`
- the selected LLM Agent
- the selected `SimulationRuntime`
- the lock that serializes state changes

It exposes application operations for state, agent configuration, runtime
switching, reset, step, stop, and wind. Flask routes are thin adapters around
these methods.

During a step, MissionService obtains an observation, asks the Agent for a
decision, applies executor arbitration, applies the action to a copied mission
environment, and asks the active runtime to execute the candidate trajectory.
It commits the copied mission state only after runtime execution succeeds.

### SimulationRuntime

Both runtimes implement the same behavioral contract:

```python
class SimulationRuntime(Protocol):
    backend_name: str

    def reset(self, base: tuple[float, float, float]) -> dict[str, object]: ...
    def execute(self, trajectory: list[dict[str, float]]) -> RuntimeResult: ...
    def stop(self) -> None: ...
    def status(self) -> dict[str, object]: ...
    def close(self) -> None: ...
```

`RuntimeResult` contains the executed trajectory, duration, and kinematics in
the same shape for both runtimes.

### HtmlRuntime

HtmlRuntime has no external dependencies and is always connected. It returns
the environment's minimum-jerk trajectory as the executed trajectory and
returns planning-level kinematics with `source: "html"`. Reset returns the
base position. Stop is a no-op. Wind is not supported.

HtmlRuntime is a real independent executor for the educational demo. It does
not start, call, or synchronize with AirSim.

### AirSimRuntime

AirSimRuntime retains the existing dedicated RPC worker, NED conversion,
actual-position path rebasing, short-segment velocity cap, wind control, and
six-degree-of-freedom telemetry. It adapts its existing methods to the common
runtime contract.

AirSim remains optional. Its package, configuration, launch script, and live
verification script remain in the repository.

## Data Flow

Every step follows one path:

```text
Observation
    -> LLM Agent decision
    -> executor arbitration
    -> copied mission action
    -> active SimulationRuntime.execute
    -> commit copied mission state
    -> unified API state
    -> Three.js rendering
```

The active runtime is selected before a step. A single step never executes in
both runtimes.

## Runtime Switching

### HTML to AirSim

1. Stop Auto Run in the browser.
2. Create a candidate AirSimRuntime from the submitted connection settings.
3. Connect, reset, take off, apply configured wind, and read telemetry.
4. If every operation succeeds, close the HTML runtime, replace the active
   runtime, and reset `UAVEnvironment` to the base.
5. If any operation fails, close the candidate AirSim runtime and leave the
   active HTML runtime and mission state unchanged.

### AirSim to HTML

1. Stop Auto Run in the browser.
2. Create an HtmlRuntime.
3. Close the active AirSim runtime and release API control.
4. Replace the active runtime and reset `UAVEnvironment` to the base.

## REST API

The supported API is:

```text
GET  /api/state
POST /api/step
POST /api/reset
POST /api/runtime
POST /api/runtime/stop
POST /api/runtime/wind
POST /api/config
```

`POST /api/runtime` accepts:

```json
{
  "backend": "airsim",
  "config": {
    "host": "127.0.0.1",
    "port": 41451,
    "vehicle_name": "Drone1",
    "velocity_mps": 4,
    "meters_per_cell": 3,
    "meters_per_altitude_level": 5,
    "wind_ned": {
      "north": 0,
      "east": 0,
      "down": 0
    }
  }
}
```

Selecting HTML requires only `{"backend": "html"}`.

State responses use:

```json
{
  "runtime_backend": "html",
  "runtime": {
    "backend": "html",
    "connected": true,
    "source": "html"
  }
}
```

The existing mission, observation, history, trajectory, and kinematics fields
remain. `simulator_backend`, `simulator`, and `logical_uav` are removed.

The old `/api/simulator/config`, `/api/simulator/stop`, and
`/api/simulator/wind` routes are removed without compatibility aliases.

## Browser Interaction

The page adds a segmented runtime selector with HTML selected by default.
The selector is a mode control, not a command button.

In HTML mode:

- The AirSim host, port, vehicle, wind, and 6-DOF configuration panel is
  hidden.
- Step, Auto Run, Pause, and Reset work without AirSim.
- Runtime status identifies `html / ready`.

In AirSim mode:

- The AirSim connection and wind controls are visible.
- Runtime status identifies `airsim / connected` or the connection error.
- Kinematics display real AirSim telemetry.

Switching shows an explicit reset warning, stops Auto Run, and disables step
controls until switching completes. A failed AirSim switch returns the
selector to HTML and displays the error.

## Startup Scripts

`scripts/start_demo.ps1` starts only Flask in the project AirSim-compatible
Python environment. It does not launch Blocks and does not connect AirSim.

`scripts/start_airsim.ps1` continues to launch the packaged Unreal Blocks
environment. A user who wants validation runs this script before selecting
AirSim in the browser.

`scripts/stop_demo.ps1` stops the project Flask process and any packaged
Blocks process started for validation.

## Error Handling

- HTML runtime operations do not require external connection checks.
- An unsupported HTML wind request returns HTTP `409` with an
  `unsupported capability` error.
- A failed AirSim switch is atomic and does not mutate the active runtime or
  mission state.
- A failed runtime execution returns HTTP `503`, does not advance the mission,
  and leaves the selected runtime available for reconnect or replacement.
- Runtime errors use generic `Runtime error` wording. AirSim-specific detail
  remains in the AirSim runtime's exception message.
- Runtime switching, reset, step, stop, wind, and agent configuration are
  serialized by MissionService.

## Removal And Refactoring Scope

Remove:

- AirSim-specific control flow from Flask routes.
- The old `/api/simulator/*` routes.
- The `simulator_backend`, `simulator`, and `logical_uav` response fields.
- The fixed always-visible AirSim panel and associated frontend assumptions.
- Automatic AirSim launch and connection from `start_demo.ps1`.
- Helpers that become unused after MissionService and runtime extraction.

Retain:

- `UAVEnvironment`, Three.js visualization, dynamic obstacles, and trajectory
  rendering for HtmlRuntime.
- AirSimRuntime, AirSim settings, `start_airsim.ps1`, and
  `verify_airsim.py` for validation.
- Mock and DeepSeek Agent selection.

No unrelated refactoring or compatibility layer is included.

## Verification

### Runtime Contract

- HtmlRuntime resets, executes, stops, and reports status without AirSim.
- HtmlRuntime output uses `source: "html"`.
- AirSimRuntime retains RPC serialization, actual-position path rebasing,
  stable arrival, wind, and 6-DOF telemetry tests.

### MissionService

- The default runtime is HTML.
- A complete mock mission can run in HTML mode.
- A successful AirSim switch resets the mission to the base.
- A failed AirSim switch preserves the HTML runtime and current mission state.
- An AirSim execution failure does not advance the mission.
- Switching back to HTML closes AirSim and resets the mission.

### REST

- The new runtime endpoints return the specified state and status codes.
- Old `/api/simulator/*` endpoints return `404`.
- Default Flask startup succeeds while port `41451` is closed.

### Browser

- HTML is selected on first load.
- Step, Auto Run, Pause, and Reset work in HTML mode.
- AirSim controls are hidden in HTML mode and visible in AirSim mode.
- A failed AirSim switch rolls the selector back to HTML.
- Desktop and mobile screenshots have no overlap, blank WebGL canvas, or
  console errors.

### Live AirSim

- Start Blocks independently and switch the page to AirSim.
- Confirm the mission resets to the base.
- Execute at least three steps with AirSim 6-DOF telemetry.
- Switch back to HTML, verify AirSim API control is released, and verify the
  HTML mission starts at the base.
