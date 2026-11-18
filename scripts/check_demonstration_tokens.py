"""Check exported prompt/completion boundaries with an existing serving tokenizer.

This performs tokenization only: no generation, parameter updates, or training.
Training must separately validate its own processor, collator and loss masks.
"""
import argparse
import base64
from copy import deepcopy
import json
import hashlib
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import requests


def materialize_images(value, root):
    if isinstance(value, list):
        return [materialize_images(item, root) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: materialize_images(item, root) for key, item in value.items()}
    if result.get("type") in {"image", "image_url"}:
        relative = result.get("image") if result["type"] == "image" else result["image_url"]["url"]
        if not isinstance(relative, str):
            raise ValueError("Exported image URL must be a dataset-relative path")
        path = (root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(root):
            raise ValueError("Image leaves exported dataset")
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp"}.get(path.suffix)
        if not mime:
            raise ValueError("Unsupported exported image format")
        result = {"type": "image_url", "image_url": {"url": "data:" + mime + ";base64," + base64.b64encode(path.read_bytes()).decode("ascii")}}
    return result


def check_dataset(dataset, *, tokenize_url, model, output, max_length=None):
    root, output = Path(dataset).resolve(), Path(output)
    if output.exists():
        raise FileExistsError("Use a new token-check output path")
    parsed = urlsplit(tokenize_url)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Tokenizer URL must not embed credentials or query parameters")
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}):
        raise ValueError("Tokenizer must use HTTPS or local loopback HTTP")
    audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    if not audit.get("decisions"):
        raise ValueError("Dataset has no approved decisions")
    references = audit.get("files")
    if not isinstance(references, list) or not references:
        raise ValueError("Export lacks its file-integrity manifest")
    for ref in references:
        relative = ref.get("path")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("Invalid export artifact path")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != ref.get("sha256"):
            raise ValueError("Exported artifact was changed after approval")
    listed = {ref["path"] for ref in references}
    actual = {str(path.relative_to(root)) for path in root.glob("*.jsonl")}
    if not actual or not actual.issubset(listed):
        raise ValueError("Unmanifested dataset file")
    rows, started = [], time.monotonic()
    headers = {}
    secret = os.getenv("TOKENIZER_API_KEY")
    if secret:
        headers["Authorization"] = "Bearer " + secret
    with requests.Session() as session:
        for path in sorted(root.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                sample = json.loads(line)
                if sample.get("loss_scope") != "completion_only" or len(sample.get("completion", [])) != 1:
                    raise ValueError("Expected exactly one completion-only target")
                prompt = materialize_images(deepcopy(sample["prompt"]), root)
                completion = sample["completion"]
                counts, tokens, limits = [], [], []
                for messages, generation in ((prompt, True), (prompt + completion, False)):
                    request = dict(model=model, messages=messages, tools=sample.get("tools", []),
                                   add_generation_prompt=generation)
                    response = session.post(tokenize_url, json=request, headers=headers,
                                            timeout=60, allow_redirects=False)
                    if response.status_code != 200:
                        raise ValueError("Tokenizer request failed with HTTP " + str(response.status_code))
                    body = response.json()
                    ids = body.get("tokens")
                    if not isinstance(ids, list) or not all(type(token) is int for token in ids) or body.get("count") != len(ids):
                        raise ValueError("Tokenizer returned inconsistent token counts")
                    counts.append(len(ids)); tokens.append(ids); limits.append(body.get("max_model_len"))
                prefix_ok = tokens[1][:counts[0]] == tokens[0]
                limit = max_length if max_length is not None else limits[1]
                if type(limit) is not int or limit <= 0:
                    raise ValueError("Supply --max-length if the server does not publish its context limit")
                rows.append(dict(id=sample["id"], stage=sample["metadata"]["stage"],
                    prompt_tokens=counts[0], total_tokens=counts[1], completion_tokens=counts[1]-counts[0],
                    prompt_is_prefix=prefix_ok, exceeds_limit=counts[1] > limit, checked_limit=limit,
                    images=len(sample.get("images", []))))
    if len(rows) != audit["decisions"]:
        raise ValueError("Dataset size differs from its export audit")
    result = dict(mode="serving_tokenizer_only", model=model, training_started=False,
        passed=bool(rows) and all(r["prompt_is_prefix"] and r["completion_tokens"] > 0 and not r["exceeds_limit"] for r in rows),
        decisions=len(rows), min_tokens=min(r["total_tokens"] for r in rows),
        max_tokens=max(r["total_tokens"] for r in rows), wall_seconds=time.monotonic()-started,
        note="No training ran. Validate the training processor/collator and actual label masks separately before SFT.", samples=rows)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("--tokenize-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int)
    args = parser.parse_args()
    if args.max_length is not None and args.max_length <= 0:
        parser.error("--max-length must be positive")
    result = check_dataset(args.dataset, tokenize_url=args.tokenize_url, model=args.model,
                           output=args.output, max_length=args.max_length)
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
