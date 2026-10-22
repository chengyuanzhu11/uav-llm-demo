"""Shared structural contract for task publication and parser diagnostics.

The JSON Schema describes the transport shape. Existing semantic, capability,
geometry and execution checks remain authoritative after structural validation.
No model-generated source is executed here.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re


PROTOCOL_VERSION = "1.0"


class TaskProtocolError(ValueError):
    """A bounded, field-addressed error suitable for model correction feedback."""

    def __init__(self, path, code, message, **details):
        self.path, self.code = path[:600], code
        self.diagnostic = {"path": self.path, "code": code, "message": message[:600]}
        for key in ("missing", "unexpected", "allowed", "expected"):
            if key in details:
                value = details[key]
                self.diagnostic[key] = ([str(item)[:80] for item in value[:24]]
                                        if isinstance(value, (list, tuple)) else str(value)[:200])
        super().__init__(f"{self.path}: {message[:600]}")


def field_path(path, key):
    return f"{path}.{key}" if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) else f"{path}[{key!r}]"


def _object(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else list(required),
            "additionalProperties": False}


def _number(nullable=False, **bounds):
    return {"type": ["number", "null"] if nullable else "number", **bounds}


def _array(items, **bounds):
    return {"type": "array", "items": items, **bounds}


def motion_schema(limits, defaults):
    expression = {"type": ["number", "string"], "minLength": 1,
                  "maxLength": limits["max_expression_characters"]}
    coordinate = _number()
    vector = {"oneOf": [_array(coordinate, minItems=3, maxItems=3),
                         _object({axis: coordinate for axis in "xyz"})]}
    path = {"oneOf": [
        _object({"kind": {"const": "boundary"},
                 "direction": {"enum": ["clockwise", "counterclockwise"], "default": defaults["direction"]},
                 "laps": {"type": "integer", "minimum": 1, "maximum": limits["max_laps"], "default": defaults["laps"]},
                 "altitude_m": _number(True, minimum=0), "end_altitude_m": _number(True, minimum=0)}, ["kind"]),
        _object({"kind": {"const": "curve"}, **{axis: expression for axis in "xyz"},
                 "t_start": _number(minimum=-1000, maximum=1000),
                 "t_end": _number(minimum=-1000, maximum=1000),
                 "samples": {"type": "integer", "minimum": 8, "maximum": limits["max_points"], "default": 256}},
                ["kind", "x", "y", "z", "t_start", "t_end"]),
        _object({"kind": {"const": "polyline"},
                 "points_m": _array(vector, minItems=2, maxItems=limits["max_points"])}),
    ]}
    heading = {"oneOf": [
        {"type": "null"}, _object({"mode": {"const": "tangent"}}),
        _object({"mode": {"const": "fixed"}, "yaw_deg": _number(minimum=-180, maximum=180)}),
        _object({"mode": {"const": "face_target"}, "target_id": {"type": "string"}}),
    ], "default": deepcopy(defaults["heading"])}
    return _object({"path": path, "speed_mps": {**_number(True, minimum=.1, maximum=limits["max_speed_mps"]),
                                                 "default": defaults["speed_mps"]}, "heading": heading}, ["path"])


def control_schema(defaults, max_duration):
    fields = {
        "anchor": {"enum": ["center", "arrival"]},
        "offset_m": _array(_number(minimum=-100000, maximum=100000), minItems=3, maxItems=3),
        "altitude_m": _number(True, minimum=0, maximum=100000),
        "state": {"enum": ["landed", "airborne"]},
        "duration_seconds": _number(True, minimum=0, maximum=max_duration),
        "yaw_deg": _number(True),
    }
    for key, value in fields.items():
        value["default"] = deepcopy(defaults[key])
    return {"oneOf": [{"type": "null"}, _object(fields, [])]}


def task_program_schema(limits, motion_limits, motion_defaults, control_defaults, action_durations):
    condition = {"$ref": "#/$defs/condition"}
    optional_condition = {"oneOf": [{"type": "null"}, condition]}
    nodes = _array({"$ref": "#/$defs/node"}, maxItems=limits["max_nodes"])
    common = {"id": {"type": "string", "pattern": "^[A-Za-z][A-Za-z0-9_-]{0,39}$"},
              "target_id": {"type": "string"}, "when": optional_condition}
    node_variants = []
    for action in action_durations:
        if action == "review":
            fields = {"id": common["id"], "action": {"const": action},
                      "instruction": {"type": "string", "minLength": 1, "maxLength": 1000},
                      "when": {"type": "null"}}
        else:
            fields = {**common, "action": {"const": action}}
            if action == "motion":
                fields[action] = motion_schema(motion_limits, motion_defaults)
            elif action == "control":
                fields[action] = control_schema(control_defaults, limits["max_duration_seconds"])
            else:
                fields["duration_seconds"] = _number(True, minimum=0,
                    maximum=0 if action == "return_home" else limits["max_duration_seconds"])
                fields["duration_seconds"]["default"] = action_durations[action]
        node_variants.append(_object(fields))
    node_variants.extend([
        _object({"id": common["id"], "action": {"const": "repeat"},
                 "count": {"type": ["integer", "null"], "minimum": 1, "maximum": limits["max_loop_iterations"]},
                 "until": optional_condition, "steps": {**nodes, "minItems": 1}, "when": optional_condition}),
        _object({"id": common["id"], "action": {"const": "branch"},
                 "condition": condition, "then": nodes, "else": nodes}),
    ])
    comparison = {"field": {"type": "string"}, "op": {"enum": ["<", "<=", "==", "!=", ">=", ">"]},
                  "value": {"type": ["number", "string", "boolean"], "maxLength": 200}}
    conditions = [_object({key: _array(condition, minItems=1, maxItems=limits["max_nodes"])}) for key in ("all", "any")]
    conditions.extend([_object({"not": condition}),
                       _object({"step_id": {"type": "string"}, **comparison}),
                       _object({"evidence_id": {"type": "string"}, **comparison}),
                       _object({"source": {"const": "telemetry"}, **comparison})])
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "title": "Single UAV task program",
        "description": "Structural contract; catalog capabilities, condition evidence, geometry and safety limits are checked separately.",
        **_object({"status": {"enum": ["ready", "needs_clarification"]},
                   "summary": {"type": "string", "minLength": 1, "maxLength": 2000},
                   "steps": nodes,
                   "constraints": _object({"max_altitude_m": _number(True),
                                            "avoid_regions": _array({"type": "string"}, uniqueItems=True)})}),
        "$defs": {"node": {"oneOf": node_variants}, "condition": {"oneOf": conditions}},
    }


def _matches_type(value, kind):
    return {"null": lambda: value is None, "object": lambda: isinstance(value, dict),
            "array": lambda: isinstance(value, (list, tuple)), "string": lambda: isinstance(value, str),
            "boolean": lambda: isinstance(value, bool),
            "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
            "number": lambda: isinstance(value, (float, int)) and not isinstance(value, bool)}[kind]()


def validate_structure(value, schema, path="$", *, root=None, depth=0):
    """Validate only the schema vocabulary produced above, with precise errors."""
    root = schema if root is None else root
    if depth > 80:
        raise TaskProtocolError(path, "nesting_limit", "Structure exceeds the supported nesting limit.")
    if "$ref" in schema:
        return validate_structure(value, root["$defs"][schema["$ref"].split("/")[-1]], path, root=root, depth=depth + 1)
    if "oneOf" in schema:
        variants = schema["oneOf"]
        # Nullable objects/conditions and vector representations discriminate by type.
        if value is None and any(item.get("type") == "null" for item in variants):
            return
        variants = [item for item in variants if item.get("type") != "null"]
        if len(variants) == 1:
            return validate_structure(value, variants[0], path, root=root, depth=depth + 1)
        if isinstance(value, dict):
            for discriminator in ("action", "kind", "mode"):
                choices = [item.get("properties", {}).get(discriminator, {}).get("const") for item in variants]
                if all(choice is not None for choice in choices):
                    selected = next((item for item, choice in zip(variants, choices)
                                     if value.get(discriminator) == choice), None)
                    if selected is None:
                        raise TaskProtocolError(field_path(path, discriminator), "invalid_choice",
                            f"Expected {discriminator} to be one of {choices}.", allowed=choices,
                            missing=[discriminator] if discriminator not in value else [])
                    return validate_structure(value, selected, path, root=root, depth=depth + 1)
            # Conditions have one identifying key; a mixed shape is rejected by
            # that variant's additionalProperties check, not silently rewritten.
            for key in ("all", "any", "not", "step_id", "evidence_id", "source"):
                if key in value:
                    selected = next((item for item in variants if key in item.get("required", [])), None)
                    if selected is not None:
                        return validate_structure(value, selected, path, root=root, depth=depth + 1)
        typed = [item for item in variants if isinstance(item.get("type"), str)
                 and _matches_type(value, item["type"])]
        if len(typed) == 1:
            return validate_structure(value, typed[0], path, root=root, depth=depth + 1)
        raise TaskProtocolError(path, "invalid_variant", "Value does not match a published structural form.",
                                allowed=["/".join(item.get("required", [])) or item.get("type", "object") for item in variants])
    kinds = schema.get("type")
    if kinds is not None:
        kinds = [kinds] if isinstance(kinds, str) else kinds
        if not any(_matches_type(value, kind) for kind in kinds):
            raise TaskProtocolError(path, "wrong_type", f"Expected {' or '.join(kinds)}.", expected=kinds)
    if "const" in schema and value != schema["const"]:
        raise TaskProtocolError(path, "invalid_choice", "Value differs from the required constant.", allowed=[schema["const"]])
    if "enum" in schema and value not in schema["enum"]:
        raise TaskProtocolError(path, "invalid_choice", "Value is not a supported choice.", allowed=schema["enum"])
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        missing = [key for key in schema.get("required", []) if key not in value]
        unexpected = sorted(set(value) - set(properties)) if schema.get("additionalProperties") is False else []
        if missing or unexpected:
            location = field_path(path, unexpected[0] if unexpected else missing[0])
            message = (f"Missing fields: {', '.join(missing[:12])}. " if missing else "")
            message += (f"Unexpected fields: {', '.join(str(key)[:80] for key in unexpected[:12])}. " if unexpected else "")
            message += "Use only the fields published for this object."
            raise TaskProtocolError(location, "object_fields", message, missing=missing,
                                    unexpected=unexpected, allowed=list(properties))
        for key, item in value.items():
            if key in properties:
                validate_structure(item, properties[key], field_path(path, key), root=root, depth=depth + 1)
    elif isinstance(value, (list, tuple)):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", math.inf):
            raise TaskProtocolError(path, "array_length", "Array length is outside the published limits.")
        for index, item in enumerate(value):
            validate_structure(item, schema.get("items", {}), f"{path}[{index}]", root=root, depth=depth + 1)
        if schema.get("uniqueItems") and len(value) != len(set(value)):
            raise TaskProtocolError(path, "duplicate_item", "Array items must be unique.")
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", math.inf):
            raise TaskProtocolError(path, "string_length", "String length is outside the published limits.")
        if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
            raise TaskProtocolError(path, "string_pattern", "Value must be an ASCII identifier of 1 to 40 characters, beginning with a letter.")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            raise TaskProtocolError(path, "number_range", "Number must be finite and within the published bounds.")
