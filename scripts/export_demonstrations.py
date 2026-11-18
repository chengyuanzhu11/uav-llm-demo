"""Review and export evidence-backed decision samples without training."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uav_llm_demo.demonstration_data import (
    audit_episode, build_review, decision_candidates, export_dataset,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="List candidates; does not approve them")
    inspect.add_argument("episode")
    review = commands.add_parser("review", help="Review explicitly selected responses against execution")
    review.add_argument("episode")
    review.add_argument("--standard", required=True)
    review.add_argument("--execution", default="execution.json")
    review.add_argument("--selection", required=True, help="JSON object mapping response sequence to review reason")
    review.add_argument("--reviewer", required=True)
    review.add_argument("--visual-annotations", help="JSON list of reviewed visible-region polygons and image hashes")
    audit = commands.add_parser("audit", help="Recompute integrity and quality gates")
    audit.add_argument("episodes", nargs="+")
    export = commands.add_parser("export", help="Export approved responses with completion-only loss")
    export.add_argument("episodes", nargs="+")
    export.add_argument("--output", required=True, help="New directory, never overwrite a dataset")
    export.add_argument("--splits", help="JSON task_id to split mapping; default is all pilot")
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            rows = decision_candidates(args.episode)
            result = [dict(decision_id=row["decision_id"], response_sequence=row["response_sequence"],
                           stage=row["input"]["payload"].get("stage", "visual") if row["input"] else None,
                           locally_accepted=(row["result"] or {}).get("payload", {}).get("status") in {"applied", "completed"})
                      for row in rows]
        elif args.command == "review":
            selected = json.loads(Path(args.selection).read_text(encoding="utf-8"))
            if not isinstance(selected, dict):
                raise ValueError("Selection must map response sequence to review reason")
            result = build_review(args.episode,
                standard=json.loads(Path(args.standard).read_text(encoding="utf-8")),
                execution_path=args.execution, reviewer=args.reviewer,
                response_sequences=[int(sequence) for sequence in selected], target_reviews=selected,
                visual_annotations=json.loads(Path(args.visual_annotations).read_text(encoding="utf-8")) if args.visual_annotations else None)
        elif args.command == "audit":
            result = [{key: value for key, value in audit_episode(episode).items()
                       if key not in {"samples", "standard"}} for episode in args.episodes]
        else:
            splits = json.loads(Path(args.splits).read_text(encoding="utf-8")) if args.splits else None
            result = export_dataset(args.episodes, args.output, splits=splits)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command == "audit":
            return 0 if all(row["approved"] for row in result) else 1
        if args.command == "export":
            return 0 if result["decisions"] and result["approved_episodes"] == result["episodes"] else 1
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
