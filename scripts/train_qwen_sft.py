#!/usr/bin/env python3
"""Offline Qwen3-VL language-only LoRA SFT, gated by a real GPU smoke run.

Examples (separate Python 3.11 training environment):
  CUDA_VISIBLE_DEVICES=0 python scripts/train_qwen_sft.py smoke \
    --dataset /data/pilot --model /models/Qwen3-VL-4B-Instruct --output /runs/smoke-001
  CUDA_VISIBLE_DEVICES=0 python scripts/train_qwen_sft.py train \
    --dataset /data/pilot --model /models/Qwen3-VL-4B-Instruct --output /runs/sft-001 \
    --smoke-report /runs/smoke-001/smoke_report.json

Existing output directories are refused. All examples are pilot training data:
training loss and smoke checks are not held-out evaluation or flight validation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def event(name: str, **values: Any) -> None:
    print(json.dumps({'event': name, **values}, ensure_ascii=False, allow_nan=False), flush=True)


def config_load(path: Path) -> dict:
    config = json.loads(path.read_text())
    expected = {
        'seed', 'max_length', 'max_pixels', 'min_pixels', 'num_train_epochs', 'per_device_train_batch_size',
        'gradient_accumulation_steps', 'learning_rate', 'warmup_ratio', 'weight_decay',
        'max_grad_norm', 'lora_r', 'lora_alpha', 'lora_dropout', 'smoke_steps',
        'smoke_interrupt_step', 'reload_atol', 'reload_rtol',
    }
    if set(config) != expected:
        raise ValueError(f'Config keys differ: missing={sorted(expected-set(config))}, extra={sorted(set(config)-expected)}')
    for key in expected:
        if isinstance(config[key], bool) or not isinstance(config[key], (int, float)) or not math.isfinite(config[key]):
            raise ValueError(f'Config {key} must be a finite number')
    for key in ('seed', 'max_length', 'max_pixels', 'min_pixels', 'per_device_train_batch_size', 'gradient_accumulation_steps', 'lora_r', 'lora_alpha', 'smoke_steps', 'smoke_interrupt_step'):
        if not isinstance(config[key], int) or config[key] < (0 if key == 'seed' else 1):
            raise ValueError(f'Config {key} must be an integer in range')
    if config['min_pixels'] > config['max_pixels']:
        raise ValueError('min_pixels must not exceed max_pixels')
    if config['per_device_train_batch_size'] != 1:
        raise ValueError('This validated pilot uses microbatch 1; mixed-batch training requires a new validation path')
    if not 0 < config['smoke_interrupt_step'] < config['smoke_steps']:
        raise ValueError('Smoke interruption must precede the final step')
    for key in ('num_train_epochs', 'learning_rate', 'max_grad_norm', 'reload_atol', 'reload_rtol'):
        if config[key] <= 0:
            raise ValueError(f'Config {key} must be positive')
    if not 0 <= config['lora_dropout'] < 1 or not 0 <= config['warmup_ratio'] < 1 or config['weight_decay'] < 0:
        raise ValueError('Invalid regularization or warmup settings')
    return config


def model_fingerprint(directory: Path) -> dict:
    files = sorted({*directory.glob('*.json'), *directory.glob('*.jinja'), *directory.glob('*.safetensors'), *directory.glob('*.model'), *directory.glob('*.txt')})
    if not (directory / 'config.json').is_file() or not list(directory.glob('*.safetensors')):
        raise ValueError('Model must be a complete local Transformers safetensors checkpoint')
    return {path.name: {'sha256': sha256(path), 'bytes': path.stat().st_size} for path in files if path.is_file()}


def environment() -> dict:
    import torch
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or ',' in visible or torch.cuda.device_count() != 1:
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to exactly one allocated GPU; automatic GPU selection is disabled')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU with BF16 support is required')
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'transformers', 'peft', 'accelerate', 'Pillow', 'safetensors')}
    return {
        'python': platform.python_version(), 'packages': versions,
        'cuda_visible_devices': visible, 'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0), 'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory,
    }


def state_digest(value: Any) -> str:
    """Stable digest of optimizer/scheduler or adapter tensors, including values."""
    import torch
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            cpu = item.detach().cpu().contiguous()
            digest.update(str((str(cpu.dtype), tuple(cpu.shape))).encode())
            digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                digest.update(repr(key).encode()); visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            for child in item: visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def create_model(model_path: Path, config: dict, adapter: Path | None = None, trainable: bool = True):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation='sdpa',
    )
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    if adapter is None:
        targets = [name for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear)
                   and re.fullmatch(r'.*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))', name)]
        if not targets:
            raise RuntimeError('No Qwen3-VL language projection modules found; refusing broad LoRA targeting')
        model = get_peft_model(model, LoraConfig(
            task_type='CAUSAL_LM', r=config['lora_r'], lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'], target_modules=targets, bias='none',
        ))
    else:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable, local_files_only=True)
    if trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.to('cuda')
    return model


def trainable_summary(model) -> dict:
    parameters = list(model.named_parameters())
    active = [(name, parameter) for name, parameter in parameters if parameter.requires_grad]
    if not active or any('lora_' not in name or 'language_model.layers.' not in name for name, _ in active):
        raise RuntimeError('Only language-layer LoRA parameters may be trainable')
    vision = [(name, parameter) for name, parameter in parameters if '.visual.' in name or '.merger.' in name]
    if not vision or any(parameter.requires_grad for _, parameter in vision):
        raise RuntimeError('Vision encoder/merger freezing could not be verified')
    return {
        'trainable_parameters': sum(parameter.numel() for _, parameter in active),
        'total_parameters': sum(parameter.numel() for _, parameter in parameters),
        'trainable_tensors': len(active), 'frozen_visual_tensors': len(vision),
        'trainable_names': [name for name, _ in active],
    }


def on_device(inputs: dict) -> dict:
    import torch
    return {key: value.to('cuda') if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}


def gradient_probe(model, collator, record: dict) -> dict:
    import torch
    model.train()
    model.zero_grad(set_to_none=True)
    batch = on_device(collator([record]))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(**batch)
    loss = output.loss
    if not torch.isfinite(loss):
        raise RuntimeError(f'Nonfinite loss for {record["id"]}')
    loss.backward()
    nonzero = 0
    squared_norm = 0.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad and parameter.grad is not None:
            raise RuntimeError(f'Frozen parameter unexpectedly received a gradient: {name}')
        if parameter.grad is not None:
            grad = parameter.grad.detach().float()
            if not torch.isfinite(grad).all():
                raise RuntimeError(f'Nonfinite gradient: {name}')
            norm = float(grad.norm())
            squared_norm += norm * norm
            nonzero += int(norm > 0)
    if not nonzero:
        raise RuntimeError('No trainable parameter received a nonzero gradient')
    result = {'id': record['id'], 'images': len(record['images']), 'loss': float(loss.detach()),
              'nonzero_gradient_tensors': nonzero, 'gradient_l2': math.sqrt(squared_norm),
              'frozen_gradients_absent': True, 'input_tokens': int(batch['input_ids'].shape[-1]),
              'supervised_tokens': int((batch['labels'] != -100).sum())}
    model.zero_grad(set_to_none=True)
    del output, loss, batch
    torch.cuda.empty_cache()
    return result


def prediction_probe(model, collator, record: dict):
    import torch
    model.eval()
    batch = on_device(collator([record]))
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(**batch)
    # Entire vocabulary at first and last supervised token, plus full completion loss.
    positions = (batch['labels'][0] != -100).nonzero().flatten()
    indexes = torch.unique(torch.stack((positions[0] - 1, positions[-1] - 1)))
    result = {'loss': output.loss.detach().float().cpu(),
              'logits': output.logits[0, indexes].detach().float().cpu()}
    del output, batch
    return result


def training_args(config: dict, output: Path, smoke: bool):
    from transformers import TrainingArguments
    return TrainingArguments(
        output_dir=str(output), overwrite_output_dir=False,
        num_train_epochs=config['num_train_epochs'],
        max_steps=config['smoke_steps'] if smoke else -1,
        per_device_train_batch_size=config['per_device_train_batch_size'],
        gradient_accumulation_steps=1 if smoke else config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'], warmup_ratio=config['warmup_ratio'],
        weight_decay=config['weight_decay'], max_grad_norm=config['max_grad_norm'],
        lr_scheduler_type='cosine', optim='adamw_torch', bf16=True, tf32=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        dataloader_num_workers=0, dataloader_pin_memory=True,
        remove_unused_columns=False, label_names=['labels'], report_to=[],
        logging_steps=1, logging_nan_inf_filter=False,
        eval_strategy='no', save_strategy='steps' if smoke else 'epoch',
        save_steps=config['smoke_interrupt_step'] if smoke else 500,
        save_total_limit=3, save_safetensors=True, seed=config['seed'], data_seed=config['seed'],
        disable_tqdm=True,
    )


def make_trainer(model, processor, dataset, collator, config, output, smoke, callbacks=()):
    from transformers import Trainer
    trainer = Trainer(model=model, args=training_args(config, output, smoke),
                      train_dataset=dataset, data_collator=collator, processing_class=processor,
                      callbacks=list(callbacks))
    # Per-decision completion loss; do not pass accumulated token counts through PEFT.
    trainer.model_accepts_loss_kwargs = False
    return trainer


def callbacks_for_run(output: Path, stop_at: int | None = None, expected_checkpoint: Path | None = None):
    import torch
    from transformers import TrainerCallback
    class EvidenceCallback(TrainerCallback):
        def __init__(self):
            self.resume_verified = False
        def on_train_begin(self, args, state, control, **kwargs):
            if expected_checkpoint is not None:
                saved_state = json.loads((expected_checkpoint / 'trainer_state.json').read_text())
                expected_optimizer = torch.load(expected_checkpoint / 'optimizer.pt', map_location='cpu', weights_only=True)
                expected_scheduler = torch.load(expected_checkpoint / 'scheduler.pt', map_location='cpu', weights_only=True)
                evidence = {
                    'global_step': state.global_step,
                    'expected_global_step': saved_state['global_step'],
                    'optimizer_equal': state_digest(kwargs['optimizer'].state_dict()) == state_digest(expected_optimizer),
                    'scheduler_equal': state_digest(kwargs['lr_scheduler'].state_dict()) == state_digest(expected_scheduler),
                }
                if state.global_step != saved_state['global_step'] or not evidence['optimizer_equal'] or not evidence['scheduler_equal']:
                    raise RuntimeError(f'Full checkpoint resume verification failed: {evidence}')
                self.resume_verified = True
                write_json(output / 'resume_evidence.json', evidence)
                event('checkpoint_resume_verified', **evidence)
        def on_step_end(self, args, state, control, **kwargs):
            if stop_at is not None and state.global_step == stop_at:
                control.should_save = True
                control.should_training_stop = True
            return control
        def on_log(self, args, state, control, logs=None, **kwargs):
            values = {key: value for key, value in (logs or {}).items() if isinstance(value, (int, float))}
            if any(not math.isfinite(value) for value in values.values()):
                raise RuntimeError('Training emitted nonfinite loss/gradient/learning-rate')
            write_json(output / 'progress.json', {'global_step': state.global_step, 'epoch': state.epoch, 'metrics': values})
            event('training', global_step=state.global_step, **values)
    return EvidenceCallback()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def validate_resume_checkpoint(checkpoint: Path, signature: dict) -> dict:
    manifest = json.loads((checkpoint.parent / 'run_manifest.json').read_text())
    if manifest.get('mode') != 'train' or manifest.get('signature') != signature:
        raise ValueError('Resume checkpoint must belong to a formal run with identical data/model/code/config/environment')
    required = ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt',
                'scheduler.pt', 'trainer_state.json', 'rng_state.pth')
    if any(not (checkpoint / name).is_file() for name in required):
        raise ValueError('Resume requires a complete Trainer checkpoint, not a standalone adapter')
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    if not state.get('global_step') or float(state.get('epoch') or 0) >= signature['config']['num_train_epochs']:
        raise ValueError('Checkpoint has no optimizer step or has already completed the configured epochs')
    return {'checkpoint': str(checkpoint), 'global_step': state['global_step'], 'epoch': state.get('epoch'),
            'files': {name: sha256(checkpoint / name) for name in required}}


def prepare(args, config, output):
    from transformers import AutoProcessor
    from training.data import DemonstrationDataset, QwenCompletionCollator
    dataset = DemonstrationDataset(args.dataset, split=args.split)
    source_files = ['scripts/train_qwen_sft.py', 'training/data.py', 'requirements-train.txt']
    signature = {
        'dataset': dataset.summary, 'config': config,
        'source': {name: sha256(ROOT / name) for name in source_files},
        'model': model_fingerprint(args.model),
        'environment': environment(),
    }
    # Physical GPU identifier is operational, not part of mathematical compatibility.
    signature['environment'].pop('cuda_visible_devices')
    resume_evidence = validate_resume_checkpoint(args.resume_from_checkpoint, signature) if args.resume_from_checkpoint else None
    write_json(output / 'run_manifest.json', {
        'mode': args.mode, 'signature': signature, 'git_revision': git_revision(),
        'resumed_from': resume_evidence,
        'model_path': str(args.model), 'dataset_path': str(args.dataset),
        'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
        'evaluation_scope': f'Only {args.split} records used for optimization; other splits excluded.',
    })
    if args.mode == 'train':
        if args.smoke_report is None:
            raise ValueError('Formal training requires --smoke-report from this exact data/model/code/config')
        smoke = json.loads(args.smoke_report.read_text())
        if smoke.get('passed') is not True or smoke.get('signature') != signature:
            raise ValueError('Smoke report is failed or stale: data, base model, source, config or environment changed')
    processor = AutoProcessor.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=False,
        size={'shortest_edge': config['min_pixels'], 'longest_edge': config['max_pixels']},
    )
    effective_image_size = dict(processor.image_processor.size)
    if effective_image_size != {'shortest_edge': config['min_pixels'], 'longest_edge': config['max_pixels']}:
        raise RuntimeError(f'Image processor ignored requested pixel bounds: {effective_image_size}')
    write_json(output / 'processor_settings.json', {'image_size_bounds': effective_image_size,
                                                 'patch_size': processor.image_processor.patch_size,
                                                 'merge_size': processor.image_processor.merge_size})
    collator = QwenCompletionCollator(processor, args.dataset, max_length=config['max_length'])
    diagnostics = []
    token_lengths = []
    for index, row in enumerate(dataset):
        encoded, diagnostic = collator.encode(row)
        if not (encoded['labels'] != -100).any():
            raise RuntimeError(f'No supervised completion tokens: {row["id"]}')
        diagnostics.append(diagnostic)
        token_lengths.append(int(encoded['input_ids'].shape[-1]))
        if (index + 1) % 20 == 0:
            event('data_preflight', checked=index + 1, total=len(dataset))
    write_json(output / 'data_preflight.json', {'passed': True, 'count': len(dataset), 'samples': diagnostics})
    visual_indices = [index for index, row in enumerate(dataset) if row['images']]
    text_indices = [index for index, row in enumerate(dataset) if not row['images']]
    if not visual_indices or not text_indices:
        raise RuntimeError('Smoke requires both actual text and image examples')
    # Cover the longest text and the most image-rich observation in diagnostics.
    text_index = max(text_indices, key=lambda index: token_lengths[index])
    visual_index = max(visual_indices, key=lambda index: len(dataset[index]['images']))
    return dataset, processor, collator, signature, [text_index, visual_index]


def run_smoke(args, config, output, dataset, processor, collator, signature, indices):
    import torch
    from peft import get_peft_model_state_dict
    from torch.utils.data import Subset
    from transformers import set_seed
    set_seed(config['seed'])
    model = create_model(args.model, config)
    parameter_summary = trainable_summary(model)
    write_json(output / 'trainable_parameters.json', parameter_summary)
    gradients = [gradient_probe(model, collator, dataset[index]) for index in indices]
    write_json(output / 'gradient_checks.json', gradients)
    event('text_and_image_gradients_verified')
    initial_adapter_hash = state_digest(get_peft_model_state_dict(model))
    subset = Subset(dataset, indices)
    first_callback = callbacks_for_run(output, stop_at=config['smoke_interrupt_step'])
    trainer = make_trainer(model, processor, subset, collator, config, output, True, [first_callback])
    trainer.train()
    checkpoint = output / f'checkpoint-{config["smoke_interrupt_step"]}'
    for name in ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json', 'rng_state.pth'):
        if not (checkpoint / name).is_file():
            raise RuntimeError(f'Incomplete full training checkpoint: {name}')
    if trainer.state.global_step != config['smoke_interrupt_step']:
        raise RuntimeError('Smoke interruption did not occur at requested step')
    del trainer, model
    gc.collect(); torch.cuda.empty_cache()
    # Fresh model and Trainer: prove optimizer/scheduler restore, not only continuation in memory.
    model = create_model(args.model, config)
    resumed_callback = callbacks_for_run(output, expected_checkpoint=checkpoint)
    trainer = make_trainer(model, processor, subset, collator, config, output, True, [resumed_callback])
    trainer.train(resume_from_checkpoint=str(checkpoint))
    if not resumed_callback.resume_verified or trainer.state.global_step != config['smoke_steps']:
        raise RuntimeError('Smoke checkpoint resume did not complete the intended optimizer steps')
    trained_hash = state_digest(get_peft_model_state_dict(model))
    if trained_hash == initial_adapter_hash:
        raise RuntimeError('Optimizer steps did not change adapter weights')
    adapter = output / 'adapter'
    trainer.save_model(str(adapter)); processor.save_pretrained(str(adapter)); trainer.save_state()
    before = [prediction_probe(model, collator, dataset[index]) for index in indices]
    del trainer, model
    gc.collect(); torch.cuda.empty_cache()
    model = create_model(args.model, config, adapter=adapter, trainable=False)
    reloaded_hash = state_digest(get_peft_model_state_dict(model))
    if trained_hash != reloaded_hash:
        raise RuntimeError('Saved adapter tensor values changed during reload')
    reload_checks = []
    for index, expected in zip(indices, before):
        actual = prediction_probe(model, collator, dataset[index])
        checks = {key: torch.allclose(expected[key], actual[key], atol=config['reload_atol'], rtol=config['reload_rtol']) for key in expected}
        if not all(checks.values()):
            raise RuntimeError(f'Adapter output equivalence failed for {dataset[index]["id"]}: {checks}')
        reload_checks.append({'id': dataset[index]['id'], 'loss_equal': checks['loss'], 'logits_equal': checks['logits'],
                              'max_logit_abs_diff': float((expected['logits'] - actual['logits']).abs().max()),
                              'loss_abs_diff': float((expected['loss'] - actual['loss']).abs())})
    report = {'passed': True, 'signature': signature, 'optimizer_steps': config['smoke_steps'],
              'data_preflight_passed': True, 'gradients': gradients, 'checkpoint_resume_verified': True,
              'smoke_sample_ids': [dataset[index]['id'] for index in indices],
              'smoke_gradient_accumulation_steps': 1,
              'formal_gradient_accumulation_steps': config['gradient_accumulation_steps'],
              'rng_scope': 'RNG checkpoint file exists; bitwise uninterrupted-run equivalence is not claimed',
              'adapter_changed': True, 'adapter_reload_tensor_equal': True, 'reload_checks': reload_checks,
              'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
              'scope': 'Pipeline verification only; no held-out evaluation or flight capability claim.'}
    write_json(output / 'smoke_report.json', report)
    event('smoke_passed', report=str(output / 'smoke_report.json'))


def run_train(args, config, output, dataset, processor, collator, signature, indices):
    import torch
    from transformers import set_seed
    set_seed(config['seed'])
    model = create_model(args.model, config)
    write_json(output / 'trainable_parameters.json', trainable_summary(model))
    callback = callbacks_for_run(output, expected_checkpoint=args.resume_from_checkpoint)
    trainer = make_trainer(model, processor, dataset, collator, config, output, False, [callback])
    write_json(output / 'status.json', {'status': 'training', 'samples': len(dataset), 'epochs': config['num_train_epochs'],
                                      'smoke_report': str(args.smoke_report.resolve())})
    result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    final = output / 'adapter'
    trainer.save_model(str(final)); processor.save_pretrained(str(final)); trainer.save_state()
    report = {'status': 'completed', 'global_step': trainer.state.global_step, 'epoch': trainer.state.epoch,
              'metrics': result.metrics, 'samples': len(dataset), 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
              'adapter_path': str(final), 'smoke_report': str(args.smoke_report.resolve()),
              'scope': 'Pilot SFT training completed; task efficacy still requires independent execution evaluation.'}
    write_json(output / 'status.json', report)
    event('training_completed', **report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mode', choices=('smoke', 'train'))
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--split', choices=('pilot', 'train'), default='pilot',
                        help='Optimization split; validation/test are never training inputs')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/sft_pilot.json')
    parser.add_argument('--smoke-report', type=Path)
    parser.add_argument('--resume-from-checkpoint', type=Path, help='Formal run only; restores into a fresh output directory')
    args = parser.parse_args()
    args.dataset = args.dataset.resolve(strict=True); args.model = args.model.resolve(strict=True)
    if args.resume_from_checkpoint is not None:
        if args.mode != 'train':
            parser.error('--resume-from-checkpoint is only supported for formal train')
        args.resume_from_checkpoint = args.resume_from_checkpoint.resolve(strict=True)
    config = config_load(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    started = time.monotonic()
    try:
        event('run_started', mode=args.mode, output=str(output))
        prepared = prepare(args, config, output)
        if args.mode == 'smoke':
            run_smoke(args, config, output, *prepared)
        else:
            run_train(args, config, output, *prepared)
        write_json(output / 'duration.json', {'wall_seconds': time.monotonic() - started})
        return 0
    except Exception as exc:
        write_json(output / 'failure.json', {'type': type(exc).__name__, 'message': str(exc), 'wall_seconds': time.monotonic() - started})
        write_json(output / 'status.json', {'status': 'failed', 'type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
