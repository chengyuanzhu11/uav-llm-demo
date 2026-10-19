"""Translate validated task constraints into the shared geometric safety context."""
from copy import deepcopy

from .agents import GeometricPlanner, checked_plan
from .environment import point_to_dict
from .semantic_tasks import SemanticConstraintError, resolve_task_goal
from .safety import SafetyChecker


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


def validate_task_routes(plan, scenario, context):
    """Check every branch and loop transition off the control thread."""
    if plan.status != "ready":
        return plan
    # This public validation boundary may receive a shared or caller-owned snapshot.
    context = deepcopy(context)
    apply_task_constraints(context, scenario, plan.constraints)
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
            goal = points[step["target_id"]]
            for origin in origins:
                if (origin, goal) in checked:
                    continue
                context.update(position=point_to_dict(origin), goal=point_to_dict(goal))
                try:
                    checked_plan(GeometricPlanner(), context, "")
                except ValueError:
                    raise SemanticConstraintError("route_unreachable") from None
                checked.add((origin, goal))
            origins = origins | {goal} if step["when"] is not None else {goal}
        return origins

    walk(plan.steps, {start})
    return plan
