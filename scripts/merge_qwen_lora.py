#!/usr/bin/env python3
"""Create a verified independent Qwen3-VL checkpoint from language-only LoRA.

Run in the isolated training environment with exactly one allocated CUDA GPU.
Source files are never modified. Failed output is retained for diagnosis.
"""
from __future__ import annotations
import argparse
import gc
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_qwen_sft import (
    create_model, environment, event, model_fingerprint, prediction_probe, sha256, write_json,
)
from training.data import DemonstrationDataset, QwenCompletionCollator

LANGUAGE_PROJECTION = r".*language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
PROJECTION_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def adapter_provenance(adapter: Path, manifest: Path, base_model: Path) -> dict:
    """Require the same base checkpoint and only explicitly scoped LoRA tensors."""
    from safetensors import safe_open
    source = json.loads(manifest.read_text())
    signature = source.get("signature", {})
    if source.get("mode") != "train" or not signature.get("model"):
        raise ValueError("Expected the source formal training run_manifest.json")
    if signature["model"] != model_fingerprint(base_model):
        raise ValueError("Base model differs from the checkpoint used for training")
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text())
    targets = config.get("target_modules")
    if (config.get("peft_type") != "LORA" or config.get("bias") != "none"
            or config.get("modules_to_save") or config.get("use_dora")
            or not isinstance(targets, list) or not targets
            or any(not isinstance(name, str) or (name not in PROJECTION_NAMES
                    and not re.fullmatch(LANGUAGE_PROJECTION, name)) for name in targets)):
        raise ValueError("Only language-layer projection LoRA without extra saved modules is supported")
    tensor_path = adapter / "adapter_model.safetensors"
    with safe_open(str(tensor_path), framework="pt", device="cpu") as tensors:
        keys = list(tensors.keys())
        if not keys or any(not re.fullmatch(LANGUAGE_PROJECTION + r"\.lora_[AB]\.weight", name) for name in keys):
            raise ValueError("Adapter contains a tensor outside language-layer LoRA")
    return {
        "training_manifest_sha256": sha256(manifest),
        "source_signature": signature,
        "adapter_config_sha256": sha256(config_path),
        "adapter_model_sha256": sha256(tensor_path),
        "adapter_tensor_count": len(keys),
        "language_only_verified": True,
    }


def finite_probe(model, collator, row):
    import torch
    result = prediction_probe(model, collator, row)
    if any(not bool(torch.isfinite(value).all()) for value in result.values()):
        raise RuntimeError("Nonfinite model prediction: " + row["id"])
    return result


def prediction_differences(expected, actual, *, sample_id):
    import torch
    shift = actual["logits"] - expected["logits"]
    differences = shift.abs()
    centered = shift - shift.mean(dim=-1, keepdim=True)
    # Distribution comparisons are invariant to a common logit offset. Float64
    # keeps the small KL computation stable; these are detached CPU probes.
    before_logp = torch.log_softmax(expected["logits"].double(), dim=-1)
    after_logp = torch.log_softmax(actual["logits"].double(), dim=-1)
    forward = (before_logp.exp() * (before_logp - after_logp)).sum(-1).clamp_min(0)
    reverse = (after_logp.exp() * (after_logp - before_logp)).sum(-1).clamp_min(0)
    return {
        "id": sample_id,
        "loss_before": float(expected["loss"]),
        "loss_after": float(actual["loss"]),
        "loss_abs_diff": float((expected["loss"] - actual["loss"]).abs()),
        "max_logit_abs_diff": float(differences.max()),
        "mean_logit_abs_diff": float(differences.mean()),
        "logit_mean_shift_by_position": shift.mean(dim=-1).tolist(),
        "centered_max_logit_abs_diff": float(centered.abs().max()),
        "centered_mean_logit_abs_diff": float(centered.abs().mean()),
        "forward_kl_by_position": forward.tolist(),
        "reverse_kl_by_position": reverse.tolist(),
        "max_bidirectional_kl": float(torch.maximum(forward, reverse).max()),
        "probed_argmax_equal": bool(torch.equal(expected["logits"].argmax(-1), actual["logits"].argmax(-1))),
    }


def compare_merge(expected, actual, *, sample_id, max_kl, max_loss_diff):
    record = prediction_differences(expected, actual, sample_id=sample_id)
    record["passed"] = (record["max_bidirectional_kl"] <= max_kl
                        and record["loss_abs_diff"] <= max_loss_diff
                        and record["probed_argmax_equal"])
    return record


def compare_reload(expected, actual, *, sample_id):
    record = prediction_differences(expected, actual, sample_id=sample_id)
    record["passed"] = (record["max_logit_abs_diff"] <= 1e-3
                        and record["mean_logit_abs_diff"] <= 1e-4
                        and record["loss_abs_diff"] <= 1e-4)
    return record


def run(args, output):
    import torch
    from transformers import AutoConfig, AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration, set_seed
    installed = environment()
    provenance = adapter_provenance(args.adapter, args.manifest, args.base_model)
    config = provenance["source_signature"]["config"]
    dataset = DemonstrationDataset(args.dataset)
    if dataset.summary != provenance["source_signature"]["dataset"]:
        raise ValueError("Verification dataset differs from the audited training export")
    processor = AutoProcessor.from_pretrained(
        str(args.base_model), local_files_only=True, trust_remote_code=False,
        size={"shortest_edge": config["min_pixels"], "longest_edge": config["max_pixels"]},
    )
    collator = QwenCompletionCollator(processor, args.dataset, max_length=config["max_length"])
    text = next((row for row in dataset if not row["images"]), None)
    images = [row for row in dataset if row["images"]]
    if text is None or not images:
        raise ValueError("Verification requires actual text and image decisions")
    rows = [text, max(images, key=lambda row: len(row["images"]))]
    report = {
        "passed": False,
        "script_sha256": sha256(Path(__file__).resolve()),
        "provenance": provenance,
        "environment": installed,
        "samples": [collator.diagnostics(row) for row in rows],
        "merge_tolerances": {
            "max_bidirectional_kl_each_probed_position": args.max_kl_divergence,
            "max_full_target_mean_nll_abs_diff": args.max_loss_diff,
            "probed_argmax_must_match": True,
        },
        "scope": ("BF16 safe merge and saved-checkpoint numerical checks on one text and one multi-image "
                  "training example. Not an independent task evaluation or bitwise merge equivalence. "
                  "Logits cover the full vocabulary at first/last supervised positions; loss covers the entire target."),
    }
    write_json(output / "merge_report.json", report)
    set_seed(config["seed"])
    model = create_model(args.base_model, config, adapter=args.adapter, trainable=False)
    # PEFT may save short target names; inspect actual injected paths as well as
    # the saved tensor keys so a short name cannot silently wrap vision layers.
    loaded_lora = [name for name, _ in model.named_parameters() if "lora_" in name]
    if not loaded_lora or any(not re.fullmatch(
            LANGUAGE_PROJECTION + r"\.lora_[AB](?:\.[^.]+)?\.weight", name) for name in loaded_lora):
        raise RuntimeError("Loaded adapter injected LoRA outside language projection layers")
    report["loaded_language_lora_tensors"] = len(loaded_lora)
    before = [finite_probe(model, collator, row) for row in rows]
    event("adapter_probes_completed", samples=[row["id"] for row in rows])
    merged = model.merge_and_unload(safe_merge=True)
    if any("lora_" in name for name, _ in merged.named_parameters()):
        raise RuntimeError("LoRA parameters remain after merge_and_unload")
    after = [finite_probe(merged, collator, row) for row in rows]
    report["merge_checks"] = [compare_merge(a, b, sample_id=row["id"],
        max_kl=args.max_kl_divergence, max_loss_diff=args.max_loss_diff)
        for row, a, b in zip(rows, before, after)]
    write_json(output / "merge_report.json", report)
    if not all(item["passed"] for item in report["merge_checks"]):
        raise RuntimeError("BF16 merge difference exceeds declared tolerance; no checkpoint exported")
    # Restore deployment defaults from the base, not the training probe config.
    # Training pixel limits must not silently change the inference comparison.
    serving_config = AutoConfig.from_pretrained(str(args.base_model), local_files_only=True)
    if hasattr(serving_config, "use_cache"):
        merged.config.use_cache = serving_config.use_cache
    elif hasattr(merged.config, "use_cache"):
        delattr(merged.config, "use_cache")
    merged.config.text_config.use_cache = serving_config.text_config.use_cache
    serving_generation = GenerationConfig.from_pretrained(str(args.base_model), local_files_only=True)
    merged.generation_config = serving_generation
    serving_processor = AutoProcessor.from_pretrained(str(args.base_model), local_files_only=True,
                                                       trust_remote_code=False)
    merged.save_pretrained(str(output), safe_serialization=True, max_shard_size="4GB")
    serving_processor.save_pretrained(str(output))
    event("merged_checkpoint_saved", output=str(output))
    del merged, model
    gc.collect(); torch.cuda.empty_cache()
    reloaded = Qwen3VLForConditionalGeneration.from_pretrained(
        str(output), local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")
    reloaded.config.use_cache = False
    reloaded.config.text_config.use_cache = False
    deployment_processor = AutoProcessor.from_pretrained(str(output), local_files_only=True,
                                                         trust_remote_code=False)
    if (deployment_processor.image_processor.size != serving_processor.image_processor.size
            or deployment_processor.chat_template != serving_processor.chat_template
            or deployment_processor.tokenizer.special_tokens_map != serving_processor.tokenizer.special_tokens_map
            or reloaded.generation_config.to_dict() != serving_generation.to_dict()):
        raise RuntimeError("Deployment preprocessing, template or generation defaults changed")
    report["deployment_defaults_preserved"] = True
    report["serving_image_size_bounds"] = dict(deployment_processor.image_processor.size)
    # Verification repeats the same training-sized inputs, while deployment
    # artifacts keep the base model's original image limits above.
    restored_processor = AutoProcessor.from_pretrained(str(output), local_files_only=True,
        trust_remote_code=False, size={"shortest_edge": config["min_pixels"],
                                      "longest_edge": config["max_pixels"]})
    restored_collator = QwenCompletionCollator(restored_processor, args.dataset,
                                             max_length=config["max_length"])
    restored = [finite_probe(reloaded, restored_collator, row) for row in rows]
    report["reload_checks"] = [compare_reload(a, b, sample_id=row["id"])
                               for row, a, b in zip(rows, after, restored)]
    report["reload_tolerances"] = dict(max_logit_abs_diff=1e-3, mean_logit_abs_diff=1e-4,
                                       max_loss_abs_diff=1e-4)
    write_json(output / "merge_report.json", report)
    if not all(item["passed"] for item in report["reload_checks"]):
        raise RuntimeError("Merged checkpoint reload changed predictions")
    for name, parameter in reloaded.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise RuntimeError("Nonfinite saved parameter: " + name)
    report["checkpoint_files"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "merge_report.json"
    }
    report["passed"] = True
    report["all_saved_parameters_finite"] = True
    report["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    write_json(output / "merge_report.json", report)
    event("merge_verification_passed", output=str(output), checks=report["merge_checks"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Source run_manifest.json; defaults to adapter parent")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-kl-divergence", type=float, default=.01,
                        help="Fixed upper bound on each probed position's forward and reverse KL")
    parser.add_argument("--max-loss-diff", type=float, default=.05)
    args = parser.parse_args()
    for name in ("max_kl_divergence", "max_loss_diff"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error("Numerical tolerances must be positive and finite")
    for name in ("base_model", "adapter", "dataset"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    args.manifest = (args.manifest or args.adapter.parent / "run_manifest.json").resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    started = time.monotonic()
    try:
        run(args, output)
    except Exception as exc:
        write_json(output / "merge_failure.json", {
            "type": type(exc).__name__, "message": str(exc),
            "wall_seconds": time.monotonic() - started,
        })
        raise
    write_json(output / "merge_duration.json", {"wall_seconds": time.monotonic() - started})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

