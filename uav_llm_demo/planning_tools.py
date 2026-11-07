"""Read-only, snapshot-bound tools for model-directed task planning.

Candidates are never executed here. The same parser, route validator and task
contract used at acceptance evaluate every preview; tools never repair intent.
"""
from copy import deepcopy
import json
import math
import time

from .safety import SCALE, SAFETY_MARGIN, SafetyChecker
from .path_tracking import PathFollower
from .semantic_tasks import semantic_plan_from_json_text, iter_plan_actions
from .semantic_navigation import validate_task_routes, apply_task_constraints
from .spatial_motion import spatial_catalog, compile_motion, motion_flight_envelope
from .task_contract import TaskContractMonitor
from .task_protocol import TaskProtocolError, validate_structure
from .observed_targets import UnresolvedTargetReference


MAX_PLANNING_ROUNDS = 8
MAX_PLANNING_SECONDS = 120


class PlanningTools:
    def __init__(self, scenario, context, catalog, *, requirements=None, contract=None,
                 telemetry=None, cancelled=lambda: False, event=lambda **data: None, additional_request=lambda: None):
        self.scenario = scenario
        self.context, self.catalog = deepcopy(context), deepcopy(catalog)
        mission = catalog.get("mission_context", {})
        self.evidence = [deepcopy(r) for r in mission.get("evidence", [])
                         if r.get("mission_id") == mission.get("mission_id")
                         and r.get("vehicle_id") == catalog["vehicle"]["id"]]
        self.requirements, self.contract = requirements, contract
        self.telemetry = deepcopy(telemetry)
        self.cancelled, self.event = cancelled, event
        self.additional_request = additional_request
        self.deadline = time.monotonic() + MAX_PLANNING_SECONDS
        self.rounds = 0
        self._candidates = {}
        self.last_diagnostic = None

    @staticmethod
    def catalog_entry():
        return dict(
            protocol='Return {"tool": name, "arguments": object}. Tools return facts, not instructions.',
            max_rounds=MAX_PLANNING_ROUNDS, wall_budget_seconds=MAX_PLANNING_SECONDS,
            tools={
                "query_geometry": {"arguments": {"target_id": "published spatial object ID; optional when evidence_id is supplied",
                    "evidence_id": "confirmed visual evidence ID; enough on its own to query the observed anchor",
                    "radius_m": "optional 1..100 metres, default 30"},
                    "result": "anchor, region footprint, obstacles, trajectory_bounds_m INCLUDING tracking margin, and relative_path_offset_bounds_m for curve/polyline x/y/z; GRID and metres explicitly labelled"},
                "preview_plan": {"arguments": {"plan": "complete task_program_schema JSON"},
                    "result": "read-only compilation, required obligations, all branch/loop transfers and static safety; accepted candidate_id or precise diagnostic; no automatic geometric edits"},
                "submit_plan": {"arguments": {"candidate_id": "ID returned by a successful preview in this session"},
                    "result": "commit exactly that validated candidate for final acceptance; never an execution-success claim"}})

    def check_budget(self, *, starting_request=False):
        # An in-flight reply must not bypass expiry or a superseding user update.
        if self.cancelled():
            raise TaskProtocolError("$", "planning_cancelled", "Planning snapshot was superseded.")
        round_limit = starting_request and self.rounds >= MAX_PLANNING_ROUNDS
        if round_limit or time.monotonic() >= self.deadline:
            error = TaskProtocolError("$", "planning_budget_exhausted", "Tool planning budget exhausted; no candidate was accepted.")
            error.diagnostic.update(rounds=self.rounds, exhausted_limit="rounds" if round_limit else "wall_time",
                                    last_diagnostic=deepcopy(self.last_diagnostic))
            raise error

    def before_request(self):
        self.check_budget(starting_request=True)
        if self.rounds:
            self.additional_request()
        self.rounds += 1

    def validate(self, plan):
        if plan.status != "ready":
            return plan
        # Check obligations before costly geometry, and do not change any choice.
        try:
            if self.contract is not None:
                self.contract.validate_revision(plan, self.telemetry, after_review=True)
            elif self.requirements is not None:
                TaskContractMonitor(self.requirements, "preview", plan=plan, scenario=self.scenario,
                                    telemetry=self.telemetry, cache_geometry=False)
            return validate_task_routes(plan, self.scenario, self.context, evidence=self.evidence)
        except ValueError as exc:
            exc.candidate_plan = plan
            raise

    def call(self, request):
        """Return (tool observation, accepted plan or None); failures are feedback."""
        name = request.get("tool") if isinstance(request, dict) else None
        try:
            validate_structure(request, {"type": "object", "properties": {
                "tool": {"enum": ["query_geometry", "preview_plan", "submit_plan"]},
                "arguments": {"type": "object"}}, "required": ["tool", "arguments"], "additionalProperties": False})
            args = request["arguments"]
            if name == "query_geometry":
                result, plan = self._geometry(args), None
            elif name == "preview_plan":
                self._arguments(args, {"plan": {}}, ["plan"])
                plan = semantic_plan_from_json_text(json.dumps(args["plan"], allow_nan=False), self.catalog)
                self.validate(plan)
                if plan.status != "ready":
                    raise TaskProtocolError("$.plan.status", "candidate_not_ready", "Preview needs an executable candidate; clarify only essential user ambiguity.")
                candidate_id = f"candidate-{len(self._candidates) + 1}"
                motions, deferred = [], []
                for node in iter_plan_actions(plan):
                    if node["action"] == "motion":
                        try:
                            geometry = compile_motion(node, self.scenario, tuple(self.context["position"][axis] for axis in "xyz"),
                                                      plan.constraints, evidence=self.evidence)
                        except UnresolvedTargetReference:
                            deferred.append(dict(node_id=node["id"], target_ref=node["target_ref"]))
                            continue
                        motions.append(dict(node_id=node["id"], length_m=geometry.length_m,
                            samples=len(geometry.points), start_grid=list(geometry.points[0]), end_grid=list(geometry.points[-1]),
                            altitude_change_m=(geometry.points[-1][2] - geometry.points[0][2]) * SCALE[2],
                            bounds_grid=[[min(p[i] for p in geometry.points) for i in range(3)],
                                         [max(p[i] for p in geometry.points) for i in range(3)]]))
                self._candidates[candidate_id] = plan
                deferred.extend(dict(node_id=node["id"], target_ref=node["target_ref"])
                                for node in iter_plan_actions(plan) if node["action"] == "control"
                                and "step_id" in node.get("target_ref", {}))
                result, plan = dict(ok=True, candidate_id=candidate_id, motions=motions, deferred_targets=deferred,
                                    scope="Known static geometry only. Deferred target paths are NOT yet safe/executable; observation must resolve them and runtime must validate the full path and transfer."), None
            else:
                self._arguments(args, {"candidate_id": {"type": "string"}}, ["candidate_id"])
                plan = self._candidates.get(args["candidate_id"])
                if plan is None:
                    raise TaskProtocolError("$.arguments.candidate_id", "unknown_candidate", "Submit an ID from a successful preview in this session.")
                result = dict(ok=True, candidate_id=args["candidate_id"])
        except (ValueError, TypeError, RecursionError) as exc:
            result, plan = dict(ok=False, diagnostic=getattr(exc, "diagnostic", {
                "code": "invalid_tool_request", "path": "$.arguments", "message": str(exc)[:600]})), None
            self.last_diagnostic = deepcopy(result["diagnostic"])
        # Persist bounded metadata, not raw generated plans or prompts.
        self.event(tool=name if isinstance(name, str) and name in self.catalog_entry()["tools"] else "unknown",
                   ok=result.get("ok", True), candidate_id=result.get("candidate_id"),
                   diagnostic=result.get("diagnostic"))
        return result, plan

    @staticmethod
    def _arguments(args, properties, required):
        validate_structure(args, dict(type="object", properties=properties,
                                       required=required, additionalProperties=False), "$.arguments")

    def _geometry(self, args):
        self._arguments(args, {"target_id": {"type": "string"},
                               "evidence_id": {"type": "string"},
                               "radius_m": {"type": "number", "minimum": 1, "maximum": 100}}, [])
        if "target_id" not in args and "evidence_id" not in args:
            raise TaskProtocolError("$.arguments", "missing_target_reference", "Supply target_id or evidence_id.")
        objects = {item["id"]: item for item in spatial_catalog(self.scenario)["objects"]}
        target_id = args.get("target_id")
        binding = None
        if "evidence_id" in args:
            from .observed_targets import bind_spatial_object, validate_target_reference
            node = dict(action="motion", target_ref=dict(evidence_id=args["evidence_id"]))
            records = {r["evidence_id"]: r for r in self.evidence}
            validate_target_reference(node, {}, records, "$.arguments.evidence_id")
            if target_id is None:
                target_id = records[args["evidence_id"]]["target_id"]
        if target_id not in objects:
            raise TaskProtocolError("$.arguments.target_id", "invalid_reference", "Unknown geometry target.", allowed=list(objects))
        obj = objects[target_id]
        if "evidence_id" in args:
            obj, binding = bind_spatial_object(node, obj, self.evidence)
        radius, center = args.get("radius_m", 30), obj["center_grid"]
        layout = getattr(self.scenario, "campus_layout", None) or {}
        parts = [*layout.get("buildings", []), *layout.get("roofs", [])]
        if not parts:
            parts = [dict(id=f"building-{i}", footprint=[b["x"], b["x"], b["y"], b["y"]], height=b["height"])
                     for i, b in enumerate(self.context.get("buildings", []))]
        nearby = []
        pad = .39 + (SAFETY_MARGIN + PathFollower.CORRIDOR_M) / SCALE[0]
        for part in parts:
            left, right, north, south = part["footprint"]
            distance = math.hypot(max(left-pad-center[0], 0, center[0]-right-pad) * SCALE[0],
                                  max(north-pad-center[1], 0, center[1]-south-pad) * SCALE[1])
            if distance <= radius:
                nearby.append(dict(id=part["id"], distance_m=distance,
                    envelope_min_grid=[left-pad, north-pad, 0],
                    envelope_max_grid=[right+pad, south+pad, part["height"]+(SAFETY_MARGIN+PathFollower.CORRIDOR_M)/SCALE[2]],
                    roof_altitude_m=part["height"]*SCALE[2]))
        nearby.sort(key=lambda item: (item["distance_m"], item["id"]))
        constraints = (self.requirements.constraints if self.requirements else
                       self.contract.context(self.telemetry)["constraints"] if self.contract else {})
        constrained_context = deepcopy(self.context)
        constrained_context["ground_sites"] = []
        apply_task_constraints(constrained_context, self.scenario, constraints)
        checker = SafetyChecker(constrained_context)
        return dict(ok=True, object=obj, target_binding=binding, nearby_obstacles=nearby[:64], omitted_obstacles=max(0, len(nearby)-64),
                    radius_m=radius, scale_m_per_grid=list(SCALE),
                    **motion_flight_envelope(checker.lower, checker.upper, anchor=center), required_constraints=constraints,
                    note="Use relative_path_offset_bounds_m for curve/polyline offsets, not grid altitude. Bounds already reserve the tracking corridor; leave numerical headroom inside them. Footprint is the navigation region, not the detected object. Obstacles still need full preview; future observations require runtime validation.")
