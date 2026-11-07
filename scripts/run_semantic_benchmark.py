"""Run the campus task benchmark; remote API calls are opt-in."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uav_llm_demo.semantic_benchmark import (
    DEFAULT_DATASET, SEMANTIC_POLICIES, compare_policies, load_dataset, run_case, summarize,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--mode", choices=("virtual", "realtime"), default="virtual")
    parser.add_argument("--backend", choices=("offline", "deepseek"), default="offline")
    parser.add_argument("--runtime", choices=("html", "gazebo"), default="html",
                        help="Gazebo owns a fresh PX4/SITL process per case; requires --mode realtime")
    parser.add_argument("--seed", type=int, action="append", help="Repeat for paired environment seeds; default 17")
    parser.add_argument("--policy", choices=(*SEMANTIC_POLICIES, "all"), default="contract")
    parser.add_argument("--world", choices=("fixture", "seeded"), default="fixture",
                        help="Replay annotated world values or generate hidden temperatures from the paired seed")
    parser.add_argument("--control-dt", type=float, default=.05, help="Fixed simulation step; realtime uses .05")
    parser.add_argument("--control-delay-ms", type=float, default=0, help="Realtime-only injected actuator-call delay")
    parser.add_argument("--prices-json", type=Path, help="Optional user-supplied input_miss/input_hit/output rates per million tokens and currency")
    parser.add_argument("--max-calls", type=int, default=6, help="Per case hard cap, including reviews and corrections")
    parser.add_argument("--simulation-budget", type=float, default=600)
    parser.add_argument("--wall-budget", type=float, default=300)
    parser.add_argument("--planner-timeout", type=float, default=30)
    parser.add_argument("--planner-latency", type=float, default=0, help="Virtual-only fixed delivery latency per request, in simulation seconds")
    parser.add_argument("--traffic", choices=("none", "fixed"), default="none")
    parser.add_argument("--output", type=Path, help="Optional JSON result path; must not already exist")
    args = parser.parse_args()
    if args.backend == "deepseek" and args.mode != "realtime":
        parser.error("Real API requests require --mode realtime --backend deepseek")
    if args.runtime == "gazebo" and (args.mode != "realtime" or args.traffic != "none"):
        parser.error("Gazebo requires --mode realtime --traffic none; close other project SITL sessions first")
    if args.control_delay_ms and args.mode != "realtime":
        parser.error("Control-delay stress requires --mode realtime")
    seeds = args.seed or [17]
    if len(seeds) != len(set(seeds)):
        parser.error("Seeds must not repeat")
    if args.output and args.output.exists():
        parser.error("Output already exists; choose a new report path")
    data = load_dataset(args.dataset)
    ids = {c["id"] for c in data["cases"]}
    if args.case_ids and (len(args.case_ids) != len(set(args.case_ids)) or set(args.case_ids) - ids):
        parser.error("Case IDs must exist and must not repeat")
    cases = [c for c in data["cases"] if (c["id"] in args.case_ids if args.case_ids
             else args.split == "all" or c["split"] == args.split)]
    if not cases:
        parser.error("No cases selected")
    results = []
    prices = json.loads(args.prices_json.read_text(encoding="utf-8")) if args.prices_json else None
    policies = SEMANTIC_POLICIES if args.policy == "all" else (args.policy,)
    if args.backend == "deepseek":
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1]/".env")
    report = dict(dataset_version=data["version"], dataset_revision=data.get("revision"),
                  selected_runs=len(seeds)*len(cases)*len(policies), results=results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.open("x", encoding="utf-8").close()

    def checkpoint():
        report.update(summary=summarize(results), complete=len(results) == report["selected_runs"])
        if args.output:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                             prefix=".benchmark-", delete=False) as destination:
                temporary = Path(destination.name)
                json.dump(report, destination, ensure_ascii=False, indent=2, allow_nan=False)
                destination.write("\n")
            try:
                os.replace(temporary, args.output)
            finally:
                temporary.unlink(missing_ok=True)

    checkpoint()
    try:
        for seed in seeds:
            for case in cases:
                for policy in policies:
                    result = run_case(case, mode=args.mode, backend=args.backend, seed=seed,
                                      max_calls=args.max_calls, simulation_budget=args.simulation_budget,
                                      wall_budget=args.wall_budget, planner_timeout=args.planner_timeout, traffic=args.traffic,
                                      policy=policy, control_dt=args.control_dt,
                                      control_delay_ms=args.control_delay_ms, prices=prices, world_mode=args.world,
                                      planner_latency=args.planner_latency, runtime_backend=args.runtime)
                    results.append(result)
                    checkpoint()
                    print(f'{case["id"]} [{policy}, seed={seed}]: {result["status"]}; '
                          f'oracle={result["oracle_success"]}; service={result["service_success"]}', flush=True)
    except KeyboardInterrupt:
        checkpoint()
        print('Interrupted; completed cases saved, report marked incomplete.', flush=True)
        return 130
    if len(policies) > 1:
        report["policy_comparison"] = compare_policies(results)
    checkpoint()
    summary = report["summary"]
    print(f'{summary["successes"]}/{summary["runs"]} tasks satisfied; '
          f'{summary["regression_passes"]}/{summary["runs"]} expected outcomes')
    if len(policies) > 1:
        for policy, measured in report["policy_comparison"]["summaries"].items():
            print(f'{policy}: {measured["successes"]}/{measured["runs"]}; '
                  f'false completion={measured["false_completions"]}; calls={measured["total_interpretation_calls"]}')
        # A deliberately ablated baseline may fail the task; that is data.
        # Full-system regressions and unexpected runtime failures still fail CI.
        return 0 if all(r["termination_reason"] != "runtime_error" and
                        (r["protocol"]["policy"] != "contract" or r["regression_pass"]) for r in results) else 1
    return 0 if all(r["regression_pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
