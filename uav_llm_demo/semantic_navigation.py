"""Translate validated task constraints into the shared geometric safety context."""
from copy import deepcopy

from .agents import GeometricPlanner, checked_plan
from .environment import point_to_dict
from .semantic_tasks import SemanticConstraintError, resolve_task_goal, iter_plan_actions
from .safety import SafetyChecker, SAFETY_MARGIN, SCALE, _intersects
from .spatial_motion import compile_motion
from .path_tracking import PathFollower
from .spatial_control import resolve_control_goal


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
    for point in motion.points:
        # A curve can overshoot at a vertical or horizontal turn even when every
        # reference point is in bounds. Reserve the same verified tracking corridor.
        if any(value < checker.lower[i] + PathFollower.CORRIDOR_M / SCALE[i] - 1e-8
               or value > checker.upper[i] - PathFollower.CORRIDOR_M / SCALE[i] + 1e-8
               for i, value in enumerate(point)):
            raise SemanticConstraintError("motion_path_unsafe")
        if checker.validate_waypoint(point):
            raise SemanticConstraintError("motion_path_unsafe")
    for a, b in zip(motion.points, motion.points[1:]):
        if not checker.segment_safe(a, b):
            raise SemanticConstraintError("motion_path_unsafe")
    return motion


def task_ground_sites(plan, scenario):
    sites = []
    for step in iter_plan_actions(plan):
        if step["action"] == "control":
            try:
                goal = resolve_control_goal(step, scenario, plan.constraints)
            except ValueError:
                raise SemanticConstraintError("control_goal_unsafe") from None
            if goal["ground_site"] is not None:
                site = dict(zip("xyz", goal["ground_site"]))
                if site not in sites:
                    sites.append(site)
    return sites


def validate_task_routes(plan, scenario, context):
    """Check every branch and loop transition off the control thread."""
    if plan.status != "ready":
        return plan
    # This public validation boundary may receive a shared or caller-owned snapshot.
    context = deepcopy(context)
    apply_task_constraints(context, scenario, plan.constraints)
    context["ground_sites"] = [*context.get("ground_sites", []), *task_ground_sites(plan, scenario)]
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
            motion = None
            if action == "motion":
                try:
                    motion = validate_motion_path(
                        compile_motion(step, scenario, start, plan.constraints), context)
                except ValueError:
                    raise SemanticConstraintError("motion_path_unsafe") from None
                goal, end = motion.points[0], motion.points[-1]
            elif action == "control":
                goal = end = resolve_control_goal(step, scenario, plan.constraints)["position"]
            else:
                goal = end = points[step["target_id"]]
            for origin in origins:
                if (origin, goal) in checked:
                    continue
                context.update(position=point_to_dict(origin), goal=point_to_dict(goal))
                try:
                    checked_plan(GeometricPlanner(), context, "")
                except ValueError:
                    raise SemanticConstraintError("route_unreachable") from None
                checked.add((origin, goal))
            origins = origins | {end} if step["when"] is not None else {end}
        return origins

    walk(plan.steps, {start})
    return plan
