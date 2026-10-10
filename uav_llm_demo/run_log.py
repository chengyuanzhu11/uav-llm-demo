"""Asynchronous, bounded JSONL journal. Only explicitly selected mission data enter it."""
import atexit
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from logging.handlers import RotatingFileHandler
from pathlib import Path
from queue import Queue, Full
from threading import Thread, Lock
import time
import uuid


class JournalWriter:
    def __init__(self, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.handler = RotatingFileHandler(destination, maxBytes=20_000_000, backupCount=5, encoding='utf-8')
        self.handler.setFormatter(logging.Formatter('%(message)s'))
        self.queue = Queue(maxsize=4096)
        self.dropped = 0
        self.errors = 0
        self.handler.handleError = self._handle_error
        self.thread = Thread(target=self._write, name='uav-run-journal', daemon=True)
        self.thread.start()
        atexit.register(self.close)

    def _handle_error(self, record):
        self.errors += 1

    def emit(self, record):
        try:
            self.queue.put_nowait(record)
        except Full:
            self.dropped += 1

    def _write(self):
        while True:
            record = self.queue.get()
            try:
                if record is None:
                    return
                # Encoding and disk I/O stay outside the control thread.
                message = json.dumps(record, ensure_ascii=True, allow_nan=False)
                self.handler.emit(logging.LogRecord('uav.run', logging.INFO, '', 0, message, (), None))
            except Exception:
                self.errors += 1
            finally:
                self.queue.task_done()

    def flush(self):
        self.queue.join()
        self.handler.flush()

    def close(self):
        if self.thread.is_alive():
            self.queue.put(None)
            self.thread.join(timeout=3)
            self.handler.close()


_writer = None


def configure_run_logging(destination=None):
    global _writer
    if _writer is None:
        _writer = JournalWriter(destination or Path(__file__).resolve().parents[1] / 'logs' / 'mission_runs.jsonl')
    return _writer


def source_fingerprint():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob('*.py')):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class RunTrace:
    def __init__(self, writer=None, secrets=()):
        self.secrets = tuple(s for s in secrets if isinstance(s, str) and s)
        self.run_id = uuid.uuid4().hex
        self.writer = writer if writer is not None else _writer
        self.started = time.monotonic()
        self.lock = Lock()
        self.sequence = 0
        self.samples = defaultdict(lambda: deque(maxlen=2048))
        self.counts = defaultdict(int)
        self.maxima = defaultdict(float)
        self.submitted_requests = 0
        self.usage = dict(requests=0, usage_missing=0, prompt_tokens=0, completion_tokens=0,
                          prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=0)
        self.last_interval = 0.0
        self.lag_seconds = 0.0
        self.missed_deadlines = 0

    def _redact(self, value):
        if isinstance(value, str):
            for secret in self.secrets:
                if len(secret) >= 8:
                    value = value.replace(secret, "[redacted]")
                elif value == secret:
                    value = "[redacted]"
            return value
        if isinstance(value, dict):
            return {key: self._redact(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._redact(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    def event(self, kind, simulation_time, **data):
        data = self._redact(data)
        with self.lock:
            self.sequence += 1
            record = dict(run_id=self.run_id, sequence=self.sequence, event=kind,
                          timestamp=datetime.now(timezone.utc).isoformat(),
                          wall_seconds=time.monotonic() - self.started, simulation_seconds=simulation_time, **data)
            if self.writer:
                self.writer.emit(record)

    def provider_started(self):
        with self.lock:
            self.submitted_requests += 1

    def usage_event(self, event):
        with self.lock:
            self.usage['requests'] += 1
            if any(event.get(key) is None for key in ('prompt_tokens', 'completion_tokens',
                                                      'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens')):
                self.usage['usage_missing'] += 1
            for key in ('prompt_tokens', 'completion_tokens', 'prompt_cache_hit_tokens', 'prompt_cache_miss_tokens'):
                if type(event.get(key)) is int:
                    self.usage[key] += event[key]
        # Request timestamps are wall time; do not invent a simultaneous sim timestamp.
        self.event('provider_request', None, request=event)

    @contextmanager
    def stage(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.duration(name, (time.perf_counter() - started) * 1000)

    def duration(self, name, milliseconds):
        with self.lock:
            self.samples[name].append(milliseconds)
            self.counts[name] += 1
            self.maxima[name] = max(self.maxima[name], milliseconds)

    def interval(self, elapsed, step):
        self.last_interval = elapsed
        self.lag_seconds += max(0.0, elapsed - step)
        if elapsed > 0.075:
            self.missed_deadlines += 1

    def snapshot(self):
        with self.lock:
            stages = {}
            for key, values in self.samples.items():
                ordered = sorted(values)
                stages[key] = dict(count=self.counts[key], max_ms=round(self.maxima[key], 3),
                                   p95_ms=round(ordered[min(len(ordered)-1, int((len(ordered)-1)*.95))], 3))
            return dict(run_id=self.run_id, wall_seconds=time.monotonic()-self.started,
                        timing=dict(stages=stages, percentile_window=2048,
                                    last_interval_seconds=self.last_interval, lag_seconds=self.lag_seconds,
                                    missed_deadlines=self.missed_deadlines), usage=dict(self.usage, unreported_requests=max(0, self.submitted_requests-self.usage['requests'])),
                        journal_dropped=self.writer.dropped if self.writer else 0,
                        journal_errors=self.writer.errors if self.writer else 0)
