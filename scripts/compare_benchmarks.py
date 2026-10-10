import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uav_llm_demo.benchmark import compare_results


def main():
    parser = argparse.ArgumentParser(description='Compare matched benchmark seeds, reporting recoveries and regressions.')
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = compare_results(json.loads(args.before.read_text())['results'], json.loads(args.after.read_text())['results'])
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
