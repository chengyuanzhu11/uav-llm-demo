"""Source-grounded task requirements, independent of executable plan defaults.

Source coverage is a mechanical omission check, not a proof of interpretation.
An independent semantic audit must approve these requirements before planning.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import unicodedata

from .task_protocol import TaskProtocolError, validate_structure


class TaskRequirementsError(TaskProtocolError):
    """A requirements or audit response needs bounded, field-addressed repair."""


@dataclass(frozen=True)
class TaskRequirements:
    status: str
    summary: str
    steps: tuple[dict, ...]
    constraints: dict
    coverage: tuple[dict, ...]
    instruction: str

    def to_dict(self):
        return {"status": self.status, "summary": self.summary,
                "steps": deepcopy(list(self.steps)), "constraints": deepcopy(self.constraints),
                "coverage": deepcopy(list(self.coverage))}


@dataclass(frozen=True)
class RequirementsAudit:
    status: str
    summary: str
    issues: tuple[dict, ...]
    assertions: tuple[dict, ...] = ()
    confirmed_assertions: tuple[dict, ...] = ()

    def to_dict(self):
        return {"status": self.status, "summary": self.summary, "issues": deepcopy(list(self.issues)),
                "assertions": deepcopy(list(self.assertions))}


def _object(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": False}


def _explicit_fields(schema):
    """Defaults and nullable numeric defaults belong to execution, not intent."""
    if isinstance(schema, dict):
        schema.pop("default", None)
        if schema.get("type") == ["number", "null"]:
            schema["type"] = "number"
        for value in schema.values():
            _explicit_fields(value)
    elif isinstance(schema, list):
        for value in schema:
            _explicit_fields(value)


_REFERENCE_GEOMETRY_FIELDS = {"curve": ("x", "y", "t_start", "t_end"), "polyline": ("points_m",)}


def requirement_schema(*, for_extraction=False):
    # Deferred import keeps semantic_tasks free to consume requirements without
    # duplicating its capability vocabulary or creating an import-time cycle.
    from .semantic_tasks import _task_schema

    schema = _task_schema()
    _explicit_fields(schema)
    schema["title"] = "Single UAV task requirements"
    schema["description"] = (
        "Only user-specified obligations and constraints. Omit unspecified parameters; "
        "never copy execution defaults or planner choices. Source coverage requires a separate semantic audit. "
        "needs_clarification requires steps=[], constraints={}, coverage=[]; ready requires at least one obligation.")
    for variant in schema["$defs"]["node"]["oneOf"]:
        fields = variant["properties"]
        action = fields["action"]["const"]
        if action == "branch":
            continue
        if action == "repeat":
            variant["required"] = ["id", "action", "count", "until", "steps"]
        elif action == "review":
            variant["required"] = ["id", "action", "instruction"]
        else:
            variant["required"] = ["id", "action"] + ([] if action in ("motion", "control", "visual") else ["target_id"])
        if action == "control":
            fields["control"] = fields["control"]["oneOf"][1]
        if action == "visual":
            variant["required"].append("visual")
            fields["visual"]["required"] = ["description"]
        if action == "motion":
            if for_extraction:
                variant["anyOf"] = [{"type": "object", "required": [key]} for key in ("target_id", "target_ref")]
            variant["description"] = ("A motion obligation requires target_id or target_ref. target_id is a node-level "
                                      "field, beside motion; geometry belongs under motion.path and orientation under motion.heading.")
            motion = fields["motion"]
            motion["required"] = []
            heading = motion["properties"]["heading"]
            heading["oneOf"] = [item for item in heading["oneOf"] if item.get("type") != "null"]
            for path in motion["properties"]["path"]["oneOf"]:
                path["required"] = ["kind"]
                path["properties"].pop("samples", None)
                path["properties"]["altitude_delta_m"] = {"type": "number", "minimum": -100000, "maximum": 100000}
                path["properties"]["altitude_trend"] = {"enum": ["nondecreasing", "nonincreasing", "level"]}
                kind = path["properties"]["kind"]["const"]
                if kind in _REFERENCE_GEOMETRY_FIELDS:
                    if for_extraction:
                        path["required"].extend(["flexible", *_REFERENCE_GEOMETRY_FIELDS[kind]])
                    axes = {"type": "array", "items": {"type": "string", "enum": list("xyz")},
                            "uniqueItems": True, "maxItems": 3}
                    path["properties"]["flexible"] = {
                        **_object({"scale_groups": {"type": "array", "items": {**axes, "minItems": 1}, "maxItems": 3},
                                   "offset_axes": axes}),
                        "description": ("Only authorize dimensions or positions the user left unspecified. "
                                        "Each disjoint axis group shares one positive scale; offset axes allow translation. "
                                        "Curves require x/y and t interval; omit z when only height change/trend is specified. "
                                        "Polylines require reference points. Both lists may be empty only when the source fixes geometry; "
                                        "omitting freedom must never freeze model-selected dimensions or altitude."),
                    }
    constraints = schema["properties"]["constraints"]
    constraints["required"] = []
    constraints["properties"]["max_altitude_m"]["minimum"] = 0
    schema["properties"]["coverage"] = {
        "type": "array", "maxItems": 128,
        "items": _object({
            "quote": {"type": "string", "minLength": 1, "maxLength": 10000},
            "requirement_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True, "maxItems": 128},
            "constraint_keys": {"type": "array", "items": {"enum": list(constraints["properties"])},
                                "uniqueItems": True, "maxItems": len(constraints["properties"])},
        }),
    }
    schema["required"].append("coverage")
    return schema


def requirements_audit_schema():
    return _object({
        "status": {"enum": ["approved", "needs_clarification", "revision_required"]},
        "summary": {"type": "string", "minLength": 1, "maxLength": 2000},
        "assertions": {"type": "array", "maxItems": 256, "items": _object({
            "quote": {"type": "string", "minLength": 1, "maxLength": 10000},
            "requirement_id": {"type": "string", "minLength": 1, "maxLength": 40},
            "expected_json": {"type": "string", "minLength": 2, "maxLength": 16000},
        })},
        "issues": {"type": "array", "maxItems": 128, "items": _object({
            "code": {"type": "string", "minLength": 1, "maxLength": 80},
            "message": {"type": "string", "minLength": 1, "maxLength": 600},
            "quote": {"type": "string", "minLength": 1, "maxLength": 10000},
        })},
    })


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise TaskRequirementsError("$", "duplicate_field", "Duplicate JSON field.")
        result[key] = value
    return result


def _reject_nonfinite(value):
    raise TaskRequirementsError("$", "invalid_json_number", "Non-finite JSON number.")


def _read(text, schema):
    if not isinstance(text, str) or len(text) > 100000:
        raise TaskRequirementsError("$", "response_size", "Requirements response must be text of at most 100000 characters.")
    try:
        data = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_nonfinite)
        validate_structure(data, schema)
    except json.JSONDecodeError as error:
        raise TaskRequirementsError("$", "invalid_json", f"Invalid JSON syntax at line {error.lineno}, column {error.colno}.") from None
    except TaskProtocolError as error:
        _raise_protocol_error(error)
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise TaskRequirementsError("$", "invalid_structure", "Invalid requirements JSON structure.") from None
    if not data["summary"].strip():
        raise TaskRequirementsError("$.summary", "empty_text", "Summary cannot be blank.")
    return data


def _raise_protocol_error(error):
    if isinstance(error, TaskRequirementsError):
        raise error
    details = {key: value for key, value in error.diagnostic.items() if key not in ("path", "code", "message")}
    raise TaskRequirementsError(error.path, error.code, error.diagnostic["message"], **details) from None


def _instruction_text(instruction):
    if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 100000:
        raise TaskRequirementsError("$.instruction", "invalid_source", "Original instruction must be nonblank text of at most 100000 characters.")


def _coverage(data, instruction, identifiers):
    # Difference marks cover repeated/overlapping literal quotes without quadratic
    # range insertion. Quotes have no offsets, so every occurrence is covered;
    # multiplicity and the meaning of each mapping remain the audit's job.
    covered = [0] * (len(instruction) + 1)
    for index, item in enumerate(data["coverage"]):
        path = f"$.coverage[{index}]"
        quote = item["quote"]
        if not quote.strip() or quote not in instruction:
            raise TaskRequirementsError(f"{path}.quote", "invalid_source", "Quote must be a nonblank verbatim fragment of the current instruction.")
        if not item["requirement_ids"] and not item["constraint_keys"]:
            raise TaskRequirementsError(path, "unlinked_source", "A source fragment must refer to a requirement or explicit constraint.")
        for position, identifier in enumerate(item["requirement_ids"]):
            if identifier not in identifiers:
                raise TaskRequirementsError(f"{path}.requirement_ids[{position}]", "invalid_reference", "Unknown requirement ID.", allowed=sorted(identifiers))
        for position, key in enumerate(item["constraint_keys"]):
            if key not in data["constraints"]:
                raise TaskRequirementsError(f"{path}.constraint_keys[{position}]", "invalid_reference", "Source refers to an absent constraint.", allowed=list(data["constraints"]))
        start = instruction.find(quote)
        while start >= 0:
            covered[start] += 1
            covered[start + len(quote)] -= 1
            start = instruction.find(quote, start + 1)
    if data["status"] == "ready":
        count, missing = 0, []
        for index, character in enumerate(instruction):
            count += covered[index]
            if not count and not character.isspace() and not unicodedata.category(character).startswith("P"):
                missing.append(character)
        if missing:
            raise TaskRequirementsError("$.coverage", "source_coverage_missing", "Uncovered instruction text: " + "".join(missing)[:160])


def requirements_from_json_text(text, catalog, instruction, *, for_extraction=False):
    """Validate requirements without assigning defaults, paths or feasibility."""
    from .semantic_tasks import PROGRAM_LIMITS, _condition
    from .spatial_motion import _Expression

    _instruction_text(instruction)
    # Older persisted specifications remain readable; newly generated geometry
    # must explicitly declare which parameters belong to the planner.
    data = _read(text, requirement_schema(for_extraction=for_extraction))
    if data["status"] == "ready" and not data["steps"]:
        raise TaskRequirementsError("$.steps", "empty_requirements", "Ready requirements need at least one obligation.")
    if data["status"] == "needs_clarification" and (data["steps"] or data["constraints"] or data["coverage"]):
        raise TaskRequirementsError("$", "clarification_contract", "Clarification must leave steps, constraints and coverage empty.")
    targets = {point["id"]: point for point in catalog["points_of_interest"]}
    regions = {region["id"] for region in catalog["regions"]}
    spatial_targets = {point["id"] for point in catalog.get("spatial", {}).get("objects", [])}
    for index, region in enumerate(data["constraints"].get("avoid_regions", [])):
        if region not in regions:
            raise TaskRequirementsError(f"$.constraints.avoid_regions[{index}]", "invalid_reference", "Unknown avoid region.", allowed=sorted(regions))
    context = catalog.get("mission_context", {})
    evidence = {item["evidence_id"]: item for item in context.get("evidence", [])
                if isinstance(item, dict) and isinstance(item.get("evidence_id"), str)
                and item.get("mission_id") == context.get("mission_id")
                and item.get("vehicle_id") == catalog["vehicle"]["id"]}
    identifiers = set()

    def walk(nodes, previous, depth=0, path="$.steps"):
        if depth >= PROGRAM_LIMITS["max_depth"]:
            raise TaskRequirementsError(path, "nesting_limit", "Requirements nesting exceeds the supported budget.")
        guaranteed = dict(previous)
        for index, node in enumerate(nodes):
            node_path = f"{path}[{index}]"
            identifier, action = node["id"], node["action"]
            if identifier in identifiers:
                raise TaskRequirementsError(f"{node_path}.id", "duplicate_id", "Requirement IDs must be globally unique.")
            identifiers.add(identifier)
            if len(identifiers) > PROGRAM_LIMITS["max_nodes"]:
                raise TaskRequirementsError(node_path, "node_budget", "Requirements node budget exceeded.")
            if node.get("when") is not None:
                _condition(node["when"], guaranteed, evidence, path=f"{node_path}.when")
            if action == "branch":
                _condition(node["condition"], guaranteed, evidence, path=f"{node_path}.condition")
                left = walk(node["then"], guaranteed, depth + 1, f"{node_path}.then")
                right = walk(node["else"], guaranteed, depth + 1, f"{node_path}.else")
                guaranteed = {key: value for key, value in left.items() if key in right}
            elif action == "repeat":
                body = walk(node["steps"], guaranteed, depth + 1, f"{node_path}.steps")
                if node["until"] is not None:
                    _condition(node["until"], body, evidence, path=f"{node_path}.until")
                if node.get("when") is None:
                    guaranteed = body
            else:
                if action == "review":
                    if not node["instruction"].strip():
                        raise TaskRequirementsError(f"{node_path}.instruction", "empty_text", "Review instruction cannot be blank.")
                else:
                    target = node.get("target_id")
                    from .observed_targets import validate_target_reference
                    validate_target_reference(node, guaranteed, evidence, f"{node_path}.target_ref")
                    if action == "motion" and target is None and "target_ref" not in node:
                        raise TaskRequirementsError(f"{node_path}.target_id", "invalid_reference", "Motion needs a named place or an observed target reference.")
                    if target is not None and (target not in targets or action not in targets[target]["allowed_actions"]):
                        raise TaskRequirementsError(f"{node_path}.target_id", "invalid_reference", "Requirement target or capability is unavailable.",
                                                    allowed=[key for key, point in targets.items() if action in point["allowed_actions"]])
                    if action == "return_home" and target != "home":
                        raise TaskRequirementsError(f"{node_path}.target_id", "invalid_reference", "Return must target home.", allowed=["home"])
                    if action == "visual" and not node["visual"]["description"].strip():
                        raise TaskRequirementsError(f"{node_path}.visual.description", "empty_text", "Visual description cannot be blank.")
                    duration = node.get("duration_seconds")
                    if duration is not None and action in ("inspect", "thermal_scan", "capture") and duration <= 0:
                        raise TaskRequirementsError(f"{node_path}.duration_seconds", "number_range", "An observation duration must be positive.")
                    if action == "motion":
                        motion = node.get("motion", {})
                        geometry = motion.get("path", {})
                        if "flexible" in geometry:
                            flexible = geometry["flexible"]
                            geometry_path = f"{node_path}.motion.path"
                            required = _REFERENCE_GEOMETRY_FIELDS[geometry["kind"]]
                            missing = [key for key in required if key not in geometry]
                            if missing:
                                raise TaskRequirementsError(f"{geometry_path}.{missing[0]}", "incomplete_reference_geometry",
                                    "Flexible geometry requires a complete reference path.", missing=missing)
                            scaled_axes = set()
                            for group_index, group in enumerate(flexible["scale_groups"]):
                                if scaled_axes.intersection(group):
                                    raise TaskRequirementsError(f"{geometry_path}.flexible.scale_groups[{group_index}]", "duplicate_axis",
                                        "An axis cannot belong to more than one scale group.")
                                scaled_axes.update(group)
                        if geometry.get("kind") == "curve":
                            if ("t_start" in geometry and "t_end" in geometry
                                    and not 0 < geometry["t_end"] - geometry["t_start"] <= 1000):
                                raise TaskRequirementsError(f"{node_path}.motion.path.t_end", "parameter_range",
                                    "t_end must be greater than t_start, by at most 1000.")
                            for axis in "xyz":
                                if axis in geometry:
                                    try:
                                        expression = _Expression(geometry[axis])
                                        if "flexible" in geometry:
                                            expression.range_and_derivative(geometry["t_start"], geometry["t_end"])
                                    except ValueError as error:
                                        raise TaskRequirementsError(f"{node_path}.motion.path.{axis}", "invalid_expression", str(error)) from None
                        if "altitude_delta_m" in geometry:
                            # A shape reference and a relative-height obligation
                            # must describe the same motion before they are frozen.
                            delta = None
                            if geometry.get("kind") == "curve" and all(k in geometry for k in ("z", "t_start", "t_end")):
                                expr = _Expression(geometry["z"])
                                delta = expr.at(geometry["t_end"]) - expr.at(geometry["t_start"])
                            elif geometry.get("kind") == "polyline" and "points_m" in geometry:
                                from .safety import _point
                                delta = _point(geometry["points_m"][-1])[2] - _point(geometry["points_m"][0])[2]
                            elif all(k in geometry for k in ("altitude_m", "end_altitude_m")):
                                delta = geometry["end_altitude_m"] - geometry["altitude_m"]
                            wanted_delta = geometry["altitude_delta_m"]
                            trend = geometry.get("altitude_trend")
                            if ((trend == "nondecreasing" and wanted_delta < 0)
                                    or (trend == "nonincreasing" and wanted_delta > 0)
                                    or (trend == "level" and wanted_delta != 0)):
                                raise TaskRequirementsError(f"{node_path}.motion.path.altitude_trend", "inconsistent_height_requirement",
                                    "Height trend contradicts the required height change.")
                            z_scalable = any("z" in group for group in geometry.get("flexible", {}).get("scale_groups", []))
                            can_scale = z_scalable and delta is not None and delta * wanted_delta > 0
                            if delta is not None and not can_scale and not math.isclose(delta, wanted_delta, abs_tol=1e-7):
                                raise TaskRequirementsError(f"{node_path}.motion.path.altitude_delta_m", "inconsistent_height_requirement",
                                    "Reference geometry contradicts the required height change. Correct the reference; keep the source height obligation.")
                        heading = motion.get("heading", {})
                        if heading.get("mode") == "face_target" and heading["target_id"] not in spatial_targets:
                            raise TaskRequirementsError(f"{node_path}.motion.heading.target_id", "invalid_reference", "Heading target is unavailable.", allowed=sorted(spatial_targets))
                if node.get("when") is None:
                    guaranteed[identifier] = node
        return guaranteed

    try:
        walk(data["steps"], {})
    except TaskProtocolError as error:
        _raise_protocol_error(error)
    _coverage(data, instruction, identifiers)
    return TaskRequirements(data["status"], data["summary"].strip(), tuple(deepcopy(data["steps"])),
                            deepcopy(data["constraints"]), tuple(deepcopy(data["coverage"])), instruction)


def audit_requirements_projection(requirements):
    """Expose structure and order without the extractor's persuasive prose."""
    data = requirements.to_dict() if hasattr(requirements, "to_dict") else requirements
    result = {"$": {"steps": [node["id"] for node in data["steps"]],
                    "constraints": deepcopy(data.get("constraints", {}))}}

    def visit(nodes):
        for node in nodes:
            item = {key: deepcopy(value) for key, value in node.items() if key != "id"}
            for key in ("steps", "then", "else"):
                if key in node:
                    item[key] = [child["id"] for child in node[key]]
                    visit(node[key])
            result[node["id"]] = item
    visit(data["steps"])
    return result


def _assertion_leaves(value, prefix=()):
    if isinstance(value, dict) and value:
        for key, item in value.items():
            yield from _assertion_leaves(item, (*prefix, key))
    else:
        yield prefix, value


def _semantic_assertion_fields(projection):
    # Reference formulae are implementation choices for a requested shape. Its
    # direction, amount, height predicates and all task structure remain audited.
    basis = {"x", "y", "z", "t_start", "t_end", "points_m", "flexible"}
    fields = {}
    for identifier, item in projection.items():
        fields[identifier] = [path for path, value in _assertion_leaves(item)
            if not (len(path) >= 3 and path[:2] == ("motion", "path") and path[2] in basis)
            and not (identifier == "$" and path == ("constraints",) and value == {})]
    return fields


def _assertion_expected(assertion, index):
    location = f"$.assertions[{index}].expected_json"
    try:
        expected = json.loads(assertion["expected_json"], object_pairs_hook=_unique_object,
                              parse_constant=_reject_nonfinite)
    except (json.JSONDecodeError, ValueError, RecursionError):
        raise TaskRequirementsError(location, "invalid_assertion", "expected_json must encode one JSON object.") from None
    if not isinstance(expected, dict) or not expected or "id" in expected:
        raise TaskRequirementsError(location, "invalid_assertion", "Assert nonempty node fields; identify the node with requirement_id.")
    for path, value in _assertion_leaves(expected):
        if len(path) > 24 or any(not isinstance(key, str) or not key or len(key) > 80 for key in path):
            raise TaskRequirementsError(location, "invalid_assertion", "Assertion field nesting is invalid.")
        if isinstance(value, (float, int)) and not math.isfinite(value):
            raise TaskRequirementsError(location, "invalid_assertion", "Assertion numbers must be finite.")
    return expected


def _asserted_equal(expected, actual):
    # JSON booleans must not compare equal to the numbers 0 and 1.
    if isinstance(expected, bool) or isinstance(actual, bool):
        return type(expected) is type(actual) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(expected, list) and isinstance(actual, (tuple, list)):
        return len(expected) == len(actual) and all(_asserted_equal(a, b) for a, b in zip(expected, actual))
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(_asserted_equal(value, actual[key]) for key, value in expected.items())
    return type(expected) is type(actual) and expected == actual


def compare_requirement_assertions(requirements, assertions, instruction):
    """Compare source assertions locally and retain matching facts across repair."""
    projection = audit_requirements_projection(requirements)
    issues, confirmed, asserted = [], [], {}
    for index, assertion in enumerate(assertions):
        quote = assertion.get("quote", "")
        if not quote.strip() or quote not in instruction:
            raise TaskRequirementsError(f"$.assertions[{index}].quote", "invalid_source", "Assertion quote must occur verbatim in the instruction.")
        expected = _assertion_expected(assertion, index)
        identifier = assertion["requirement_id"]
        matched = {}
        for path, wanted in _assertion_leaves(expected):
            key = (identifier, path)
            if key in asserted and not _asserted_equal(asserted[key], wanted):
                raise TaskRequirementsError(f"$.assertions[{index}]", "conflicting_assertions", "Source assertions disagree about the same field.")
            asserted[key] = wanted
            actual, found = projection.get(identifier), identifier in projection
            for name in path:
                if not isinstance(actual, dict) or name not in actual:
                    found = False
                    break
                actual = actual[name]
            if found and _asserted_equal(wanted, actual):
                branch = matched
                for name in path[:-1]:
                    branch = branch.setdefault(name, {})
                branch[path[-1]] = deepcopy(wanted)
            else:
                field = identifier + "." + ".".join(path)
                actual_text = json.dumps(actual, ensure_ascii=False) if found else "<missing>"
                issues.append({"code": "source_assertion_mismatch", "quote": quote,
                    "message": f"{field}: source requires {json.dumps(wanted, ensure_ascii=False)}, candidate has {actual_text}."[:600]})
        if matched:
            confirmed.append({"quote": quote, "requirement_id": identifier,
                              "expected_json": json.dumps(matched, ensure_ascii=False, separators=(",", ":"))})
    return issues, tuple(confirmed), asserted


def audit_requirements_from_json_text(text, requirements, instruction):
    """Audit decisions are derived from field comparisons, never empty approval."""
    _instruction_text(instruction)
    if requirements.instruction != instruction:
        raise TaskRequirementsError("$.instruction", "source_mismatch", "Audit source differs from the extracted requirements source.")
    data = _read(text, requirements_audit_schema())
    issues = deepcopy(data["issues"])
    for index, issue in enumerate(issues):
        for key in ("code", "message", "quote"):
            if not issue[key].strip():
                raise TaskRequirementsError(f"$.issues[{index}].{key}", "empty_text", "Audit issue fields cannot be blank.")
        if issue["quote"] not in instruction:
            raise TaskRequirementsError(f"$.issues[{index}].quote", "invalid_source", "Audit quote must occur verbatim in the instruction.")
    mismatches, confirmed, asserted = compare_requirement_assertions(requirements, data["assertions"], instruction)
    issues.extend(mismatches)
    if data["status"] == "needs_clarification":
        return RequirementsAudit(data["status"], data["summary"].strip(), tuple(issues), tuple(data["assertions"]), confirmed)
    if not data["assertions"]:
        raise TaskRequirementsError("$.assertions", "missing_assertions", "Audit must independently state source obligations and compare their fields.")
    # Every asserted clause and every source character is accounted for. Explicit
    # issues can cover clauses whose required structure is still missing.
    quoted = [item["quote"] for item in [*data["assertions"], *issues]]
    covered = [False] * len(instruction)
    for quote in quoted:
        start = instruction.find(quote)
        while start >= 0:
            covered[start:start + len(quote)] = [True] * len(quote)
            start = instruction.find(quote, start + 1)
    if any(not seen and not char.isspace() and not unicodedata.category(char).startswith("P")
           for char, seen in zip(instruction, covered)):
        raise TaskRequirementsError("$.assertions", "source_coverage_missing", "Audit assertions must cover every source clause, including missing obligations.")
    for identifier, paths in _semantic_assertion_fields(audit_requirements_projection(requirements)).items():
        for path in paths:
            if (identifier, path) not in asserted:
                issues.append({"code": "unaudited_requirement", "quote": instruction,
                    "message": f"{identifier}.{'.'.join(path)} has no source-grounded field assertion; audit it or remove an invented requirement."})
    if data["status"] == "revision_required" and not issues:
        raise TaskRequirementsError("$.issues", "missing_issues", "A required revision must identify a source-grounded difference.")
    status = "revision_required" if issues else "approved"
    if requirements.status != "ready":
        raise TaskRequirementsError("$.status", "invalid_approval", "Audit requires ready requirements.")
    summary = (f"原文事项与结构字段有 {len(issues)} 处差异，需要修正。"
               if issues and data["status"] == "approved" else data["summary"].strip())
    return RequirementsAudit(status, summary, tuple(issues[:128]), tuple(data["assertions"]), confirmed)
