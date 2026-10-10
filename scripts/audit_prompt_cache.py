"""Capture offline request fixtures or summarize numeric DeepSeek usage logs."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uav_llm_demo import agents
from uav_llm_demo.mission_service import MissionService


def fixtures():
    service = MissionService()
    first = service._context()
    service._environment.position = (1.0, 12.0, 2.0)
    service._environment.step_count = 100
    service._environment.battery = 99
    second = service._context()
    third = deepcopy(second)
    third["dynamic_objects"][0]["position"]["x"] += 0.01
    return [dict(name=name, context=context, feedback=feedback) for name, context, feedback in (
        ("initial", first, ""), ("moving_replan", second, ""),
        ("stationary_retry", third, "Segment 0 is statically unsafe. Replan."))]


def first_difference(a, b):
    prefix = 0
    for index, (left, right) in enumerate(zip(a, b)):
        if left["role"] != right["role"]:
            return dict(message=index, char_offset=0, common_content_chars=prefix)
        x, y = left["content"], right["content"]
        offset = next((i for i, pair in enumerate(zip(x, y)) if pair[0] != pair[1]), min(len(x), len(y)))
        prefix += offset
        if x != y:
            return dict(message=index, char_offset=offset, common_content_chars=prefix,
                        before=x[max(0, offset-30):offset+45], after=y[max(0, offset-30):offset+45])
    return dict(message=None, char_offset=None, common_content_chars=prefix)


def capture(destination, source=None):
    records = json.loads(source.read_text(encoding="utf-8"))["requests"] if source else fixtures()
    agent = agents.DeepSeekAgent(api_key="offline-test-only")
    for record in records:
        response = Mock(status_code=200)
        response.json.return_value = {"choices": [{"message": {"content": json.dumps(dict(
            waypoints=[record["context"]["goal"]], reason="Offline fixture", risk="low", expected_result="Inspect only"))}}]}
        with patch.object(agents, "_post_deepseek", return_value=response) as send, \
                patch.object(agents, "record_request", create=True):
            agent.plan(record["context"], record["feedback"])
        record["messages"] = send.call_args.kwargs["json"]["messages"]
    comparisons = [dict(pair=f'{a["name"]} -> {b["name"]}', **first_difference(a["messages"], b["messages"]))
                   for a, b in zip(records, records[1:])]
    payload = dict(source="Synthetic offline contexts; transport mocked; NOT provider cache measurements",
                   requests=records, comparisons=comparisons)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))


def summarize(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = {}
    for row in records:
        group = groups.setdefault((row.get("prompt_version"), row.get("model")),
                                 dict(requests=0, measured_requests=0, prompt_tokens=0,
                                      prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=0))
        group["requests"] += 1
        if row.get("usage_consistent"):
            group["measured_requests"] += 1
            for key in ("prompt_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
                group[key] += row[key]
    for (version, model), totals in groups.items():
        totals["weighted_hit_rate"] = totals["prompt_cache_hit_tokens"] / totals["prompt_tokens"] if totals["prompt_tokens"] else None
        print(json.dumps(dict(prompt_version=version, model=model, **totals)))
    for row in records[-12:]:
        print(json.dumps({key: row.get(key) for key in ("timestamp", "prompt_version", "model", "http_status",
            "prompt_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "cache_hit_rate", "usage_consistent")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--snapshot", type=Path)
    mode.add_argument("--usage-log", type=Path)
    parser.add_argument("--source", type=Path, help="Reuse an earlier snapshot's identical contexts")
    args = parser.parse_args()
    if args.snapshot:
        capture(args.snapshot, args.source)
    else:
        summarize(args.usage_log)
