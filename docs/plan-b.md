# Plan B Implementation Notes

## Runtime shape

```text
LLM (Mock or DeepSeek)
        |
Structured AgentDecision
  skill + waypoint + action
        |
MissionService / safety arbitration
        |
  HTML Runtime (default) <----> AirSim Runtime (optional validation)
        |
EvaluationLog: events, interventions, path, battery, success
```

The LLM is a high-level planner. It does not control motor outputs, attitude,
stabilization, or actuator timing. `MissionService` validates the proposal,
applies mission invariants, chooses a local safe step when necessary, and
passes only a smooth trajectory to the selected runtime.

## Data-driven scenarios

The built-in scenario ids are:

- `urban_crossing`: the default 3D city-like obstacle field with three birds.
- `open_field`: no static buildings and one dynamic flying object.
- `narrow_corridor`: fewer obstacle cells that make local planning visible.

Scenario definitions include base, target, altitude bounds, buildings, and
dynamic-object initial state. `UAVEnvironment` accepts a scenario id or a
`ScenarioDefinition` directly, while legacy default constants remain available
for compatibility with existing callers.

## Agent contract

DeepSeek and Mock return the same structure:

```json
{
  "action": "move_east",
  "skill": "navigate_to",
  "waypoint": {"x": 8, "y": 2, "z": 2},
  "reason": "The next safe waypoint reduces the goal distance.",
  "risk": "low",
  "expected_result": "Reach the next waypoint."
}
```

`proposed_action` and `safety_intervened` are added by the executor trace when
the safety layer changes the action. This makes cases such as a DeepSeek
`move_north` proposal during `returning_home` observable instead of silently
blaming or hiding the final executor decision.

## Evaluation

`GET /api/state` exposes `metrics` and `evaluation`. Metrics include success,
steps, safety interventions, intervention rate, agent errors, separation or
collision-related feedback, path length, and battery used. The event log keeps
the observation, structured decision, selected runtime, and environment
feedback for replay and analysis.

Additional endpoints:

```text
GET  /api/scenarios
POST /api/scenario       {"scenario_id":"open_field"}
```

Run all scenarios with the offline Agent:

```powershell
& '<workspace-python>\python.exe' scripts/run_benchmark.py --max-steps 200
```

Run one scenario or write compact JSON results:

```powershell
& '<workspace-python>\python.exe' scripts/run_benchmark.py `
  --scenario urban_crossing --output artifacts\urban.json
```
