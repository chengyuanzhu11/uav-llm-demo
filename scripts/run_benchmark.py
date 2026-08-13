from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uav_llm_demo.benchmark import run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Run UAV LLM scenario benchmarks.")
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="Scenario id to run. Repeat to run a subset; default runs all.",
    )
    parser.add_argument("--backend", default="mock", choices=("mock", "deepseek"))
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = run_benchmark(
        args.scenarios,
        backend=args.backend,
        max_steps=args.max_steps,
    )
    rendered = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
