"""Translate validated task constraints into the shared geometric safety context."""
from copy import deepcopy
from dataclasses import replace

from .agents import GeometricPlanner, checked_plan
from .environment import point_to_dict
from .semantic_tasks import SemanticConstraintError, resolve_task_goal, iter_plan_actions, task_point_in_avoided_region
from .safety import SafetyChecker, SAFETY_MARGIN, SCALE, _intersects
from .spatial_motion import (compile_motion, spatial_catalog, _Expression, normalize_motion, MotionConstraintError,
                             motion_flight_envelope, motion_bound_violations)
from .path_tracking import PathFollower
from .spatial_control import control_catalog, normalize_control, resolve_control_goal
from .task_contract import TaskContractMonitor, TaskContractError
from .observed_targets import UnresolvedTargetReference, resolve_target_reference


def apply_task_constraints(context, scenario, constraints):
    """Apply validated constraints in place to a context owned by the caller."""
    ceiling = constraints.get("max_altitude_m")
    if ceiling is not None:
        context["altitude_range"]["max"] = min(context["altitude_range"]["max"], ceiling / 5.0)
    regions = {item["id"]: item for item in scenario.semantic_regions}
    for region_id in constraints.get("avoid_regions", []):
        left, right, north, south = regions[region_id]["bounds"]
        context["no_fly_zones"].append(dict(
            min=dict(x=left, y=north, z=0),
            max=dict(x=right, y=south, z=scenario.max_altitude)))

    # A previously used landing site is not a permanent exception to new constraints.
    # Retain only sites outside the newly prohibited volume. Start validation below
    # rejects a revision if the aircraft itself is still inside that volume.
    for region_id in constraints.get("avoid_regions", []):
        left, right, north, south = regions[region_id]["bounds"]
        lower = (left - SAFETY_MARGIN / SCALE[0], north - SAFETY_MARGIN / SCALE[1], -1)
        upper = (right + SAFETY_MARGIN / SCALE[0], south + SAFETY_MARGIN / SCALE[1], scenario.max_altitude + 1)
        context["ground_sites"] = [site for site in context.get("ground_sites", [])
                                   if not _intersects(tuple(site[a] for a in "xyz"),
                                                      tuple(site[a] for a in "xyz"), lower, upper)]


def validate_motion_path(motion, context):
    """Validate every compiled segment, including the tracking corridor."""
    # Tracking paths stay above cruise floor; old landing columns are irrelevant
    # to their enlarged tracking corridor and must not invalidate unrelated paths.
    checker = SafetyChecker({**context, "ground_sites": []}, clearance=SAFETY_MARGIN + PathFollower.CORRIDOR_M)
    envelope = motion_flight_envelope(checker.lower, checker.upper)
    def reject(a, b, index):
        obstacle = next(({"min_grid": list(lo), "max_grid": list(hi), "kind": label}
                         for lo, hi, label in checker.boxes if _intersects(a, b, lo, hi)), None)
        violations = motion_bound_violations(a, envelope)
        detail = {}
        if violations:
            detail["message"] = "Trajectory reference violates bounds after reserving the tracking corridor: " + "; ".join(
                f"{v['axis']}={v['actual_m']:.9f} m, allowed [{v['minimum_m']:.9f}, {v['maximum_m']:.9f}] m, "
                f"minimum signed correction {v['required_shift_m']:+.9f} m" for v in violations)
        raise SemanticConstraintError("motion_path_unsafe", diagnostic=dict(
            sample_index=index, segment_grid=[list(a), list(b)], obstacle_envelope=obstacle,
            **envelope, bound_violations=violations, **detail, clearance_m=checker.clearance))
    for index, point in enumerate(motion.points):
        # A curve can overshoot at a vertical or horizontal turn even when every
        # reference point is in bounds. Reserve the same verified tracking corridor.
        if motion_bound_violations(point, envelope):
            reject(point, point, index)
        if checker.validate_waypoint(point):
            reject(point, point, index)
    for index, (a, b) in enumerate(zip(motion.points, motion.points[1:])):
        if not checker.segment_safe(a, b):
            reject(a, b, index)
    return motion


def task_ground_sites(plan, scenario, *, evidence=()):
    sites = []
    for step in iter_plan_actions(plan):
        if step["action"] == "control":
            try:
                goal = resolve_control_goal(step, scenario, plan.constraints, evidence=evidence)
            except UnresolvedTargetReference:
                continue
            except ValueError:
                raise SemanticConstraintError("control_goal_unsafe") from None
            if goal["ground_site"] is not None:
                site = dict(zip("xyz", goal["ground_site"]))
                if site not in sites:
                    sites.append(site)
    return sites


def validate_task_routes(plan, scenario, context, *, evidence=()):
    """Check every branch and loop transition off the control thread."""
    if plan.status != "ready":
        return plan
    # This public validation boundary may receive a shared or caller-owned snapshot.
    context = deepcopy(context)
    apply_task_constraints(context, scenario, plan.constraints)
    context["ground_sites"] = [*context.get("ground_sites", []), *task_ground_sites(plan, scenario, evidence=evidence)]
    start = tuple(context["position"][axis] for axis in "xyz")
    if SafetyChecker(context).validate_waypoint(start):
        raise SemanticConstraintError("current_position_outside_constraints")
    points = {item["id"]: resolve_task_goal(item, plan.constraints)
              for item in scenario.points_of_interest}
    checked = set()

    def walk(nodes, origins):
        for step in nodes:
            action = step["action"]
            if action == "branch":
                origins = walk(step["then"], origins) | walk(step["else"], origins)
                continue
            if action == "repeat":
                ends = walk(step["steps"], origins)
                if step["count"] is None or step["count"] > 1:
                    # Possible exits are the same after each pass; the second
                    # traversal adds every back edge without unrolling counts.
                    ends = walk(step["steps"], ends)
                origins = origins | ends if step["when"] is not None else ends
                continue
            if action == "review":
                continue
            if step.get("target_ref"):
                try:
                    resolve_target_reference(step, evidence)
                except UnresolvedTargetReference:
                    # No fabricated coordinates and no fallback to the region.
                    # The execution worker must compile and validate after detection.
                    origins = origins | {None} if step["when"] is not None else {None}
                    continue
            motion = None
            if action == "motion":
                try:
                    motion = validate_motion_path(
                        compile_motion(step, scenario, start, plan.constraints, evidence=evidence), context)
                except SemanticConstraintError as exc:
                    exc.diagnostic["node_id"] = step["id"]
                    raise
                except ValueError as exc:
                    raise SemanticConstraintError("motion_path_unsafe", diagnostic={
                        **getattr(exc, "diagnostic", {"message": str(exc)[:600]}),
                        "node_id": step["id"]}) from None
                goal, end = motion.points[0], motion.points[-1]
            elif action == "control":
                goal = end = resolve_control_goal(step, scenario, plan.constraints, evidence=evidence)["position"]
            else:
                goal = end = points[step["target_id"]]
            for origin in origins:
                if origin is None:
                    continue
                if (origin, goal) in checked:
                    continue
                context.update(position=point_to_dict(origin), goal=point_to_dict(goal))
                try:
                    checked_plan(GeometricPlanner(), context, "")
                except ValueError:
                    raise SemanticConstraintError("route_unreachable", diagnostic={
                        "node_id": step["id"], "origin_grid": list(origin), "goal_grid": list(goal)}) from None
                checked.add((origin, goal))
            # A visual search may move after reaching its starting region.
            if action == "visual":
                end = None
            origins = origins | {end} if step["when"] is not None else {end}
        return origins

    walk(plan.steps, {start})
    return plan


def hard_requirement_conflict(requirements, scenario, context):
    """Prove conflicts only from explicit user geometry or hard constraints.

    Absence of a proof means the planner may still have another execution choice.
    In particular, a failed search is never treated as an infeasibility proof.
    """
    if requirements is None:
        return None
    checked = deepcopy(context)
    apply_task_constraints(checked, scenario, requirements.constraints)
    if checked["altitude_range"]["max"] < checked["altitude_range"]["min"]:
        return "altitude_out_of_range"
    start = tuple(checked["position"][axis] for axis in "xyz")
    if SafetyChecker(checked).validate_waypoint(start):
        return "current_position_outside_constraints"
    points = {item["id"]: item for item in scenario.points_of_interest}
    regions = {item["id"]: item for item in scenario.semantic_regions}
    for node in iter_plan_actions(requirements):
        if node.get("target_ref"):
            continue
        if node["action"] == "control":
            control = node.get("control", {})
            fixed = (node.get("target_id") is not None
                     and all(key in control for key in ("state", "anchor", "offset_m"))
                     and (control["state"] == "landed" or control.get("altitude_m") is not None))
            if fixed:
                try:
                    normalize_control(control, node["target_id"], control_catalog(scenario), requirements.constraints)
                except ValueError:
                    continue  # Unsupported parameters are not geometric proof.
                try:
                    resolve_control_goal(node, scenario, requirements.constraints)
                except ValueError:
                    return "control_goal_unsafe"
            continue
        if node["action"] != "motion":
            target = points.get(node.get("target_id"))
            if target is not None and task_point_in_avoided_region(
                    resolve_task_goal(target, requirements.constraints), regions,
                    requirements.constraints.get("avoid_regions", [])):
                return "target_in_avoided_region"
            continue
        path = node.get("motion", {}).get("path", {})
        if any(path.get("flexible", {}).values()):
            continue
        explicit = (path.get("kind") == "polyline" and "points_m" in path
                    or path.get("kind") == "curve" and all(key in path for key in ("x", "y", "z", "t_start", "t_end")))
        if explicit:
            concrete = deepcopy(node)
            concrete["motion"]["path"].pop("altitude_delta_m", None)
            concrete["motion"]["path"].pop("altitude_trend", None)
            concrete["motion"]["path"].pop("flexible", None)
            try:
                concrete["motion"] = normalize_motion(concrete["motion"], concrete["target_id"],
                                                       spatial_catalog(scenario), requirements.constraints)
                geometry = compile_motion(concrete, scenario, start, requirements.constraints)
            except MotionConstraintError:
                return "motion_path_unsafe"
            except ValueError:
                continue  # A missing execution parameter is not geometric proof.
            try:
                validate_motion_path(geometry, checked)
            except SemanticConstraintError:
                return "motion_path_unsafe"
    return None


def repair_task_routes(plan, scenario, context, requirements=None, contract=None, telemetry=None, *, evidence=()):
    """Repair free parameters in at most 64 local candidates, then validate all routes.

    Candidate construction is geometric. The independent contract decides which
    edits are allowed; a fixed node is never changed to repair a different node.
    """
    def check_contract(candidate, *, geometry=False):
        if contract is not None:
            contract.validate_revision(candidate, telemetry, after_review=True)
        elif requirements is not None:
            TaskContractMonitor(requirements, "preflight", plan=candidate, scenario=scenario,
                                telemetry=telemetry, cache_geometry=geometry)

    def validate(candidate):
        validate_task_routes(candidate, scenario, context, evidence=evidence)
        check_contract(candidate, geometry=True)
        return candidate

    try:
        return validate(plan)
    except SemanticConstraintError as original:
        original_code = original.code
    if plan.status != "ready" or (requirements is None and contract is None):
        raise SemanticConstraintError(original_code)
    required_constraints = (requirements.constraints if requirements is not None
                            else contract.context(telemetry)["constraints"])
    working = replace(plan, constraints={"max_altitude_m": required_constraints.get("max_altitude_m"),
                                        "avoid_regions": list(required_constraints.get("avoid_regions", []))})
    try:
        return validate(working)
    except (ValueError, TaskContractError):
        pass
    ceiling = min(scenario.max_altitude * SCALE[2], required_constraints.get("max_altitude_m")
                  or scenario.max_altitude * SCALE[2])
    floor = scenario.min_altitude * SCALE[2]
    margin = PathFollower.CORRIDOR_M + .1
    heights = sorted({floor + margin, ceiling - margin,
                      *[height for height in (10., 15., 20., 30.) if floor < height < ceiling]})
    objects = {item["id"]: item for item in spatial_catalog(scenario)["objects"]}

    def local_safe(node, candidate):
        # Other still-unsafe nodes must not prevent incremental repair. Every
        # actual branch/loop edge is checked again before returning the plan.
        try:
            validate_task_routes(replace(candidate, steps=(node,)), scenario, context, evidence=evidence)
            return True
        except ValueError:
            return False

    def variants(node):
        if node["action"] == "control" and node["control"]["state"] == "airborne":
            for height in (None, *heights):
                candidate = deepcopy(node)
                candidate["control"]["altitude_m"] = (None if height is None else
                    height - candidate["control"]["offset_m"][2])
                yield candidate
        elif node["action"] == "motion":
            path = node["motion"]["path"]
            if path["kind"] == "boundary":
                for height in heights:
                    candidate = deepcopy(node)
                    changed = candidate["motion"]["path"]
                    changed["altitude_m"] = height
                    changed["end_altitude_m"] = height + path["end_altitude_m"] - path["altitude_m"]
                    yield candidate
                return
            origin_z = objects[node["target_id"]]["center_grid"][2] * SCALE[2]
            if node.get("target_ref"):
                try:
                    origin_z = resolve_target_reference(node, evidence)["point_grid"][2] * SCALE[2]
                except UnresolvedTargetReference:
                    return
            try:
                lower_z = (min(point[2] for point in path["points_m"]) if path["kind"] == "polyline"
                           else _Expression(path["z"]).range_and_derivative(path["t_start"], path["t_end"])[0])
            except ValueError:
                return  # Translation/scaling cannot repair an undefined expression.
            offsets = [0., *[height - origin_z - lower_z for height in heights]]
            # Try translation first; then uniform and independent horizontal
            # scale. Contract predicates reject any undeclared shape freedom.
            for sx, sy in ((1., 1.), (.75, .75), (.5, .5), (.75, 1.), (1., .75)):
                for dz in offsets:
                    if (sx, sy, dz) == (1., 1., 0.):
                        continue
                    candidate = deepcopy(node)
                    changed = candidate["motion"]["path"]
                    if path["kind"] == "polyline":
                        changed["points_m"] = [[x * sx, y * sy, z + dz] for x, y, z in path["points_m"]]
                    else:
                        if sx != 1:
                            changed["x"] = f"{sx}*({path['x']})"
                        if sy != 1:
                            changed["y"] = f"{sy}*({path['y']})"
                        if dz:
                            changed["z"] = f"({path['z']})+({dz})"
                    yield candidate

    attempts = 0
    for node_id in [node["id"] for node in iter_plan_actions(working)]:
        node = next(node for node in iter_plan_actions(working) if node["id"] == node_id)
        if local_safe(node, working):
            continue
        for replacement in variants(node):
            if attempts >= 64:
                raise SemanticConstraintError(original_code)
            attempts += 1
            candidate = deepcopy(working)
            destination = next(item for item in iter_plan_actions(candidate) if item["id"] == node_id)
            destination.clear()
            destination.update(replacement)
            try:
                check_contract(candidate)
                if not local_safe(destination, candidate):
                    continue
                working = candidate
                break
            except (ValueError, TaskContractError):
                continue
    try:
        return validate(working)
    except (ValueError, TaskContractError):
        raise SemanticConstraintError(original_code) from None
