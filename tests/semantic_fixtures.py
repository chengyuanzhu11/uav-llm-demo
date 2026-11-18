"""Test-authored semantic intent for executor and provider-transport fixtures."""
from copy import deepcopy
from collections import deque
import json
from unittest.mock import Mock

from uav_llm_demo.scenarios import load_scenario
from uav_llm_demo.semantic_tasks import task_catalog
from uav_llm_demo.task_requirements import requirements_from_json_text, audit_requirements_projection


def fixture_requirements(plan, catalog=None, instruction="执行预置测试要求。"):
    """Declare executor-test intent from its fixture, never production NL parsing.

    Language-understanding tests and the benchmark must author requirements
    independently. This helper only avoids duplicating the accepted intent in
    existing tests whose subject is execution, updates, or recovery.
    """
    raw = deepcopy(plan if isinstance(plan, dict) else plan.to_dict())
    identifiers = []

    def sparse(value):
        if isinstance(value, dict):
            if "id" in value and "action" in value:
                identifiers.append(value["id"])
            result = {key: sparse(item) for key, item in value.items()
                    if (item is not None or (value.get("action") == "repeat" and key in ("count", "until")))
                    and not (value.get("kind") == "curve" and key == "samples")}
            if result.get("kind") in ("curve", "polyline") and "flexible" not in result:
                # These executor fixtures explicitly prescribe their geometry.
                result["flexible"] = dict(scale_groups=[], offset_axes=[])
            return result
        if isinstance(value, list):
            return [sparse(item) for item in value]
        return value

    raw = sparse(raw)
    if raw["status"] == "needs_clarification":
        raw["constraints"], raw["coverage"] = {}, []
    else:
        raw["coverage"] = [dict(quote=instruction, requirement_ids=identifiers,
                                constraint_keys=list(raw.get("constraints", {})))]
    if catalog is None:
        catalog = task_catalog(load_scenario("semantic_campus"))
    return requirements_from_json_text(json.dumps(raw), catalog, instruction)


def fixture_audit(requirements, instruction, *, projection=False):
    """Assertions for explicitly authored execution fixtures, not language labels."""
    nodes = requirements if projection else audit_requirements_projection(requirements)
    return dict(status="approved", summary="执行器测试要求由测试作者预置。", issues=[], assertions=[
        dict(quote=instruction, requirement_id=identifier, expected_json=json.dumps(fields, ensure_ascii=False))
        for identifier, fields in nodes.items()])


def response(data):
    return Mock(status_code=200, json=Mock(return_value={
        "choices": [{"message": {"content": json.dumps(data, ensure_ascii=False)}}],
        "usage": dict(prompt_tokens=100, prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=100,
                      completion_tokens=50, total_tokens=150)}))


def semantic_responses(intent, replies=None):
    """HTTP fixtures for three phases, with explicitly declared executor intent.

    ``intent`` is a preset program or an instruction-to-program callable. Optional
    replies are the existing plan response stream, including errors and delays.
    Every phase remains an actual mocked HTTP request and usage-accounted call.
    """
    pending = deque(replies) if replies is not None else None

    def respond(*args, **kwargs):
        source = kwargs["json"]["messages"][1]["content"]
        catalog_text, _, rest = source.removeprefix("Catalog:\n").partition("\nInstruction:\n")
        catalog = json.loads(catalog_text)
        instruction = rest.partition("\nFeedback:\n")[0]
        declared = intent(instruction) if callable(intent) else intent
        if "proposed_requirements" in catalog:
            return response(fixture_audit(fixture_requirements(declared, catalog, instruction).to_dict(), instruction))
        if "task_requirements_schema" in catalog:
            return response(fixture_requirements(declared, catalog, instruction).to_dict())
        if pending is None:
            return response(declared)
        item = pending.popleft()
        if isinstance(item, Exception):
            raise item
        return item(*args, **kwargs) if callable(item) and not isinstance(item, Mock) else item

    return respond
