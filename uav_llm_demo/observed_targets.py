"""Bind later spatial actions to confirmed observations, never to a search region."""
from copy import deepcopy

from .onboard_observation import validate_visual_evidence
from .safety import _point, metric_distance
from .task_protocol import TaskProtocolError


class UnresolvedTargetReference(ValueError):
    """The referenced observation must finish before geometry can be resolved."""


def validate_target_reference(node, preceding, evidence, path):
    reference = node.get("target_ref")
    if reference is None:
        if (node.get("motion", {}).get("heading") or {}).get("mode") == "face_reference":
            raise TaskProtocolError(path, "missing_target_reference", "face_reference requires target_ref.")
        return
    if "step_id" in reference:
        source = preceding.get(reference["step_id"])
        if source is None or source["action"] != "visual":
            raise TaskProtocolError(path, "invalid_target_reference",
                "target_ref must name a guaranteed preceding visual step in this revision.")
    else:
        record = evidence.get(reference["evidence_id"])
        if record is None:
            raise TaskProtocolError(path, "invalid_target_reference", "Unknown visual evidence in this mission.")
        try:
            _validate_record(record)
        except ValueError as exc:
            raise TaskProtocolError(path, "invalid_target_reference", str(exc)) from None
    if node["action"] == "motion" and node.get("motion", {}).get("path", {}).get("kind") == "boundary":
        raise TaskProtocolError(path, "unavailable_target_boundary",
            "An observed point has no measured boundary. Use a relative curve or polyline, not the search region boundary.")
    if node["action"] == "control" and (node.get("control") or {}).get("anchor", "center") != "center":
        raise TaskProtocolError(path, "invalid_target_anchor", "Observed targets use their measured center, not a place arrival point.")


def _validate_record(record):
    if record.get("action") != "visual" or record.get("source") != "onboard_rgbd_model":
        raise ValueError("Target reference requires completed RGB-D visual evidence.")
    result = record.get("result", {})
    validate_visual_evidence({"visual": {"description": result.get("description"), "max_distance_m": 100}}, record)


def resolve_target_reference(node, evidence=()):
    reference = node.get("target_ref")
    if reference is None:
        return None
    records = list(evidence)
    if "evidence_id" in reference:
        record = next((r for r in records if r.get("evidence_id") == reference["evidence_id"]), None)
    else:
        # A static future plan cannot borrow an old execution with the same ID.
        # Runtime nodes carry the revision; repeated steps select its latest completion.
        record = next((r for r in reversed(records) if node.get("revision") is not None
                       and r.get("revision") == node["revision"]
                       and r.get("step_id") == reference["step_id"]), None)
    if record is None:
        raise UnresolvedTargetReference("Target observation is unresolved; keep the reference and wait for visual completion.")
    _validate_record(record)
    views = record["result"]["views"]
    points = [_point(v["point_grid"]) for v in views]
    point = [sum(p[i] for p in points) / len(points) for i in range(3)]
    return dict(evidence_id=record["evidence_id"], source_step_id=record["step_id"],
                source_execution_id=record["execution_id"], source_revision=record["revision"],
                frame_ids=[v["frame_id"] for v in views], point_grid=point,
                observation_spread_m=max(metric_distance(point, p) for p in points))


def bind_spatial_object(node, obj, evidence=()):
    binding = resolve_target_reference(node, evidence)
    if binding is None:
        return obj, None
    obj = deepcopy(obj)
    obj["center_grid"] = list(binding["point_grid"])
    obj["arrival_grid"] = list(binding["point_grid"])
    # Footprint can still constrain explicitly requested flight within the region;
    # it never becomes a boundary of the visually observed object.
    obj.pop("boundary_grid", None)
    return obj, binding
