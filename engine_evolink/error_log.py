"""Per-item failure log, like engine-replicate's _write_error_log: one
``<ts>-<stem>-<idx>_ERROR.log`` beside the would-be output."""
import json
import re
import traceback
from datetime import datetime
from pathlib import Path

from .datatypes import OutputFile

_SECRET_FIELD_RE = re.compile(
    r'("[^"]*(?:api[_-]?key|token|secret|password|authorization)[^"]*"\s*:\s*)"[^"]*"',
    re.IGNORECASE,
)


def _mask(value) -> str:
    text = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    return _SECRET_FIELD_RE.sub(r'\g<1>"***MASKED***"', text)


def _write(engine, item, idx, total, expected, stage, msg, prompt, payload, made, error) -> Path | None:
    """Best-effort: never raises."""
    try:
        path = expected.with_name(f"{expected.stem}_ERROR.log")
        path.parent.mkdir(parents=True, exist_ok=True)
        task = made.get("task") or {}
        reason = task.get("error") or {}
        lines = [
            f"{path.name} — generation failure",
            f"when: {datetime.now().astimezone().isoformat()}",
            f"item: {idx + 1} of {total}",
            f"source: {Path(item.path).resolve()}",
            f"expected: {expected.resolve()}",
            f"platform: {engine.PLATFORM}",
            f"endpoint: {engine._profile.get('endpoint', '')}",
            "profile:",
            _mask(engine._profile),
            f"prompt: {prompt or '(empty)'}",
            f"error stage: {stage}",
            f"evolink task id: {made.get('task_id', 'n/a')}",
            f"evolink task status: {task.get('status', 'n/a')}",
            f"evolink reason: {reason.get('message') or reason.get('code') or 'n/a'}",
            f"error: {msg or 'n/a'}",
            "traceback:",
            "".join(traceback.format_exception(error)).rstrip() if error else "n/a",
            "api payload:",
            _mask(payload),
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path
    except Exception:
        return None


def error_result(engine, item, idx, total, media_type, expected, stage, msg,
                 prompt="", payload=None, made=None, error=None) -> OutputFile:
    log = _write(engine, item, idx, total, expected, stage, msg, prompt, payload, made or {}, error)
    return OutputFile(
        source_path=item.path,
        status="error",
        error_msg=msg,
        media_type=media_type,
        expected_path=expected,
        metadata={"error_log": str(log)} if log else {},
    )
