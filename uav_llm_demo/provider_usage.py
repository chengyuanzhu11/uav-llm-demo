"""Credential-free request measurements; never persist messages or response text."""
from contextvars import ContextVar
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOGGER = logging.getLogger("uav_llm_demo.deepseek_usage")
PROMPT_VERSION = "static-prefix-v1"
USAGE_FIELDS = ("prompt_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
                "completion_tokens", "total_tokens")


REQUEST_TRACE = ContextVar("uav_request_trace", default=None)


@contextmanager
def request_trace(run_id, request_id, sink):
    token = REQUEST_TRACE.set((run_id, request_id, sink))
    try:
        yield
    finally:
        REQUEST_TRACE.reset(token)


def configure_usage_logging():
    if LOGGER.handlers:
        return
    destination = Path(__file__).resolve().parents[1] / "logs" / "deepseek_usage.log"
    destination.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(destination, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def prefix_metrics(messages, previous):
    common, changed = 0, None
    if previous is not None:
        for index, (a, b) in enumerate(zip(previous, messages)):
            if a["role"] != b["role"]:
                changed = index
                break
            x, y = a["content"], b["content"]
            common += next((i for i, pair in enumerate(zip(x, y)) if pair[0] != pair[1]), min(len(x), len(y)))
            if x != y:
                changed = index
                break
        else:
            if len(previous) != len(messages):
                changed = min(len(previous), len(messages))
    return dict(prompt_content_chars=sum(len(m["content"]) for m in messages),
                common_prefix_content_chars=common if previous is not None else None,
                first_changed_message=changed)


def record_request(body, *, http_status, elapsed_ms, model, metrics):
    usage = body.get("usage") if isinstance(body, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    values = {key: value if type(value := usage.get(key)) is int and value >= 0 else None
              for key in USAGE_FIELDS}
    total, hits, misses = (values[key] for key in USAGE_FIELDS[:3])
    consistent = all(value is not None for value in (total, hits, misses)) and total == hits + misses
    event = dict(timestamp=datetime.now(timezone.utc).isoformat(), prompt_version=PROMPT_VERSION,
                 model=model, http_status=http_status, elapsed_ms=elapsed_ms, **metrics, **values,
                 usage_consistent=consistent,
                 cache_hit_rate=hits / total if consistent and total else None)
    trace = REQUEST_TRACE.get()
    if trace:
        event.update(run_id=trace[0], request_id=trace[1])
        trace[2](event)
    LOGGER.info(json.dumps(event, ensure_ascii=True, allow_nan=False))
    return event
