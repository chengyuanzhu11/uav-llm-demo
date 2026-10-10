from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uav_llm_demo.benchmark import run_benchmark, summarize, estimate_cost
from uav_llm_demo.run_log import configure_run_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="Run UAV LLM scenario benchmarks.")
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="Scenario id to run. Repeat to run a subset; default runs all.",
    )
    parser.add_argument("--backend", default="mock", choices=("mock", "deepseek"))
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--mode", choices=("virtual", "realtime"), default="virtual")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--planner-latency", type=float, default=0.0)
    parser.add_argument("--planner-timeout", type=float, default=30.0)
    parser.add_argument("--prices-json", type=Path, help="Optional per-million token rates: input_miss, input_hit, output, currency.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    prices = json.loads(args.prices_json.read_text(encoding="utf-8")) if args.prices_json else None
    if prices is not None:
        estimate_cost(dict(prompt_cache_miss_tokens=0, prompt_cache_hit_tokens=0, completion_tokens=0, usage_missing=0), prices)
    if args.seeds is not None and len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique within a batch.")
    journal = configure_run_logging()
    results = run_benchmark(
        args.scenarios,
        backend=args.backend,
        max_seconds=args.max_seconds, mode=args.mode, seeds=args.seeds,
        planner_latency_seconds=args.planner_latency, planner_timeout_seconds=args.planner_timeout,
        journal=journal,
    )
    for result in results:
        result["estimated_cost"] = estimate_cost(result["diagnostics"]["usage"], prices) if prices is not None else None
    journal.flush()
    rendered = json.dumps(dict(results=results, summary=summarize(results)), ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
