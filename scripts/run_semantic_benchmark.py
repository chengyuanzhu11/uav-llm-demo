"""Run the campus task benchmark; remote API calls are opt-in."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uav_llm_demo.semantic_benchmark import DEFAULT_DATASET, load_dataset, run_case, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--mode", choices=("virtual", "realtime"), default="virtual")
    parser.add_argument("--backend", choices=("offline", "deepseek"), default="offline")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-calls", type=int, default=6, help="Per case hard cap, including reviews and corrections")
    parser.add_argument("--simulation-budget", type=float, default=600)
    parser.add_argument("--wall-budget", type=float, default=300)
    parser.add_argument("--planner-timeout", type=float, default=30)
    parser.add_argument("--traffic", choices=("none", "fixed"), default="none")
    parser.add_argument("--output", type=Path, help="Optional JSON result path; must not already exist")
    args = parser.parse_args()
    if args.backend == "deepseek" and args.mode != "realtime":
        parser.error("Real API requests require --mode realtime --backend deepseek")
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
    for case in cases:
        result = run_case(case, mode=args.mode, backend=args.backend, seed=args.seed,
                          max_calls=args.max_calls, simulation_budget=args.simulation_budget,
                          wall_budget=args.wall_budget, planner_timeout=args.planner_timeout, traffic=args.traffic)
        results.append(result)
        print(f'{case["id"]}: {result["status"]}; oracle={result["oracle_success"]}; service={result["service_success"]}', flush=True)
    report = dict(dataset_version=data["version"], dataset_revision=data.get("revision"),
                  summary=summarize(results), results=results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as destination:
            json.dump(report, destination, ensure_ascii=False, indent=2, allow_nan=False)
            destination.write("\n")
    summary = report["summary"]
    print(f'{summary["successes"]}/{summary["runs"]} tasks satisfied; '
          f'{summary["regression_passes"]}/{summary["runs"]} expected outcomes')
    return 0 if all(r["regression_pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
