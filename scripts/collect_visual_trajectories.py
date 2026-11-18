"""Collect small, sequential real-model Gazebo/PX4 episodes without training."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
import requests
from uav_llm_demo.agents import DeepSeekAgent
from uav_llm_demo.direct_vision import DirectVisionAgent
from uav_llm_demo.provider_config import load_planner_config
from uav_llm_demo.gazebo_session import GazeboSession
from uav_llm_demo.scenarios import load_scenario
from scripts.verify_gazebo import visual_check


def load_cases(path, limit):
    data = json.loads(Path(path).read_text())
    cases = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("Expected a nonempty cases list")
    required = {"id", "scene", "instruction", "oracle_target"}
    allowed = required | {"seed", "split", "group", "oracle_maximum_distance_m"}
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or not required <= set(case) or set(case) - allowed:
            raise ValueError("Invalid case fields")
        name = case["id"]
        if (not isinstance(name, str) or not name or len(name) > 80
                or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name)
                or name in seen):
            raise ValueError("Case IDs must be unique simple directory names")
        seen.add(name)
        for field in ("scene", "instruction"):
            if not isinstance(case[field], str) or not case[field].strip():
                raise ValueError("Case text fields must be nonempty")
        if case["scene"] not in {"visual_garden", "semantic_campus"}:
            raise ValueError("Case requires a native visual scene")
        targets = case['oracle_target']
        targets = [targets] if isinstance(targets, str) else targets
        if not isinstance(targets, list) or not targets or any(not isinstance(t,str) or not t.strip() for t in targets):
            raise ValueError("Case needs explicit independent target IDs")
        maximum = case.get('oracle_maximum_distance_m')
        if maximum is not None and (type(maximum) not in (int,float) or not math.isfinite(maximum) or maximum <= 0):
            raise ValueError("Case confirmation range must be finite and positive")
        seed = case.get("seed", 17)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Case seed must be an unsigned integer")
    return cases[:limit]


def check_model_endpoint(agent, role="Vision"):
    """Fail before launching PX4 when the configured model is not serving."""
    headers = {"Authorization": "Bearer " + agent.api_key} if agent.api_key else {}
    try:
        with requests.get(agent.base_url + "/models", headers=headers, timeout=10,
                          allow_redirects=False) as response:
            if response.status_code != 200:
                raise ValueError(role + " model discovery failed with HTTP " + str(response.status_code))
            body = response.json()
        models = body.get("data") if isinstance(body, dict) else None
        if not isinstance(models, list) or not any(isinstance(item, dict) and item.get("id") == agent.model
                                                  for item in models):
            raise ValueError(role + " model is not listed by its configured endpoint")
    except requests.RequestException:
        raise ValueError(role + " endpoint is unreachable; check the model service and SSH tunnel") from None


def collect(cases, output, timeout, *, planner_timeout=25):
    if any(not math.isfinite(value) or value <= 0 for value in (timeout, planner_timeout)):
        raise ValueError("Episode and planner timeouts must be finite positive seconds")
    planner_timeout = min(planner_timeout, timeout)
    output.mkdir(parents=True, exist_ok=False)
    (output/"cases.json").write_text(json.dumps({"cases":cases}, ensure_ascii=False, indent=2)+"\n")
    results = []
    summary = dict(mode="real_model_gazebo_pilot", requested=len(cases), episodes=results,
                   wall_budget_seconds=timeout, planner_timeout_seconds=planner_timeout,
                   note="Repeated layouts are pilot coverage, not held-out scene generalization.")
    for case in cases:
        directory = output/case["id"]
        directory.mkdir()
        session = GazeboSession(load_scenario(case["scene"]), seed=case.get("seed", 17))
        try:
            session.start(timeout=min(180, timeout))
            args = SimpleNamespace(scene=case["scene"], instruction=case["instruction"],
                oracle_target=case["oracle_target"], timeout=timeout,
                oracle_maximum_distance_m=case.get("oracle_maximum_distance_m"),
                trajectory_dir=directory/"trajectory", case_id=case["id"],
                seed=case.get("seed",17), split=case.get("split","pilot"),
                simulator_manifest=session.manifest)
            result = visual_check(args, directory/"result.json", session.runtime, planner_timeout=planner_timeout)
            result.update(simulator=session.manifest, simulator_logs=str(session.log_dir))
        except Exception as exc:
            result = dict(passed=False, error_type=type(exc).__name__,
                          infrastructure_failed=True, failure_code="episode_infrastructure_failed",
                          error="Episode infrastructure failed; inspect its simulator logs",
                          simulator_logs=str(session.log_dir))
        finally:
            session.close()
        (directory/"result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
        complete = bool(result.get("trajectory",{}).get("validation",{}).get("complete"))
        row = dict(id=case["id"], complete=complete, terminal_complete=result.get("terminal_complete", complete),
                   failure_stage=result.get("failure_stage"), passed=bool(result.get("passed")),
                   result=str(directory/"result.json"), wall_seconds=result.get("wall_seconds"),
                   error=result.get("error"), failure_code=result.get("failure_code"),
                   infrastructure_failed=result.get("infrastructure_failed",False))
        results.append(row)
        summary.update(attempted=len(results), complete=sum(r["complete"] for r in results),
                       terminal_complete=sum(r["terminal_complete"] for r in results),
                       successful=sum(r["passed"] for r in results))
        (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        # An incomplete collector or broken simulator must not silently burn the
        # remaining budget. Complete model failures remain in the pilot dataset.
        if not row["terminal_complete"] or result.get("infrastructure_failed"):
            break
    summary["passed"] = (summary.get("complete",0) == len(cases)
                         and summary.get("successful",0) > 0
                         and not any(r["infrastructure_failed"] for r in results))
    (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases",default=str(ROOT/"benchmarks/visual_bootstrap.json"))
    parser.add_argument("--output",required=True,help="New collection directory")
    parser.add_argument("--limit",type=int,default=20)
    parser.add_argument("--timeout",type=float,default=900)
    parser.add_argument("--planner-timeout",type=float,default=25,
                        help="Per-request planner timeout; capped by episode timeout; web defaults are unchanged")
    parser.add_argument("--check-cases",action="store_true",help="Validate manifest without inference or simulation")
    args=parser.parse_args()
    if not 1 <= args.limit <= 50 or not 0 < args.timeout <= 1800:
        parser.error("Pilot limit must be 1..50 and timeout 0..1800 seconds")
    if not math.isfinite(args.planner_timeout) or args.planner_timeout <= 0:
        parser.error("Planner timeout must be finite positive seconds")
    cases=load_cases(args.cases,args.limit)
    if args.check_cases:
        print(json.dumps(dict(valid=True,cases=len(cases),scenes=sorted({c["scene"] for c in cases}))))
        return 0
    load_dotenv(ROOT/".env")
    if os.getenv("VISION_STRATEGY") != "direct":
        parser.error("Set VISION_STRATEGY=direct for exact model-input collection")
    config = load_planner_config()
    planner = DeepSeekAgent(**config, timeout_seconds=min(args.planner_timeout,args.timeout))
    if not planner.configured:
        parser.error("Configure the planner model and endpoint; non-local endpoints also require an API key")
    agent = DirectVisionAgent(config)
    if not agent.configured:
        parser.error("Configure VISION_MODEL and VISION_BASE_URL; non-local endpoints require VISION_API_KEY")
    try:
        check_model_endpoint(planner, "Planner")
        if (agent.base_url, agent.model, agent.api_key) != (planner.base_url, planner.model, planner.api_key):
            check_model_endpoint(agent)
    except ValueError as exc:
        parser.error(str(exc))
    summary=collect(cases,Path(args.output).resolve(),args.timeout, planner_timeout=args.planner_timeout)
    print(json.dumps({k:v for k,v in summary.items() if k!="episodes"},ensure_ascii=False))
    return 0 if summary["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
