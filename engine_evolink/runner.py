"""N-in-flight execution — mirrors the gold engine's runner.

`run_batch` caps jobs in flight structurally: at most the profile's
`concurrency` futures are submitted at any moment. Per-item behaviour
reuses this engine's existing machinery verbatim (the sequential body's
chat and task branches) — same retry, polling and error semantics.
Results are written by input index; a failing item never disturbs jobs
still in flight (each failure is converted inside run_item by this
engine's own handler). The parallel path adds a lock-guarded circuit breaker
that stops new submissions after fatal errors or repeated transient failures;
never-started items remain absent, matching the reference runner.
"""

import concurrent.futures
import threading

from .datatypes import OutputFile
from .error_log import error_result
from .retry_policy import RetryingSession, classify_error


class BatchState:
    """Thread-safe stop flag and rate-limit circuit for one parallel batch."""

    def __init__(self, circuit_max: int):
        self.circuit_max = circuit_max
        self.consecutive_conflicts = 0
        self._lock = threading.Lock()
        self._stop = False

    def is_stopped(self) -> bool:
        with self._lock:
            return self._stop

    def halt(self):
        with self._lock:
            self._stop = True

    def note_success(self):
        with self._lock:
            if not self._stop:
                self.consecutive_conflicts = 0

    def note_rate_limited(self) -> tuple[bool, int]:
        with self._lock:
            self.consecutive_conflicts += 1
            count = self.consecutive_conflicts
            return count >= self.circuit_max, count


def _record_failure(engine, state, exc: Exception, prefix: str):
    kind = classify_error(exc)
    if kind == "fatal":
        state.halt()
        engine._emit(
            f"{prefix} 🛑 Fatal provider error — aborting batch. ({exc})",
            level="error",
        )
    elif kind == "retryable":
        tripped, count = state.note_rate_limited()
        if tripped:
            state.halt()
            engine._emit(
                f"{prefix} 🛑 Circuit breaker tripped after {count} consecutive "
                "rate-limited failures — aborting batch.",
                level="error",
            )
    else:
        state.note_success()


def run_item(engine, session, item, idx, total, state) -> list[OutputFile]:
    from engine_evolink.engine import _CHAT_MEDIA_TYPES

    stem = item.path.stem
    current = idx + 1
    prefix = f"[{current}/{total}] {item.path.name}"
    media_type = engine._profile.get("media_type", "") or "image"
    rel_dir = str(item.metadata.get("relative_dir", "") or "")
    dest_dir = engine._output_dir / rel_dir if rel_dir else engine._output_dir
    ts = engine._now_string()
    ext = engine._default_extension(media_type)
    expected = dest_dir / f"{ts}-{stem}-{idx}{ext}"

    prompt = f"{engine._prefix}{item.prompt}{engine._suffix}".strip()
    if not prompt:
        return [error_result(
            engine, item, idx, total, media_type, expected, "empty_prompt",
            "Empty prompt after applying prefix/suffix", prompt=prompt,
        )]

    engine._emit(f"{prefix} 📝 Prompt: {prompt}")

    payload = None
    made: dict = {}
    try:
        if media_type in _CHAT_MEDIA_TYPES:
            engine._emit(f"{prefix} 📡 Calling {engine.PROVIDER_NAME} chat API...")
            content = engine._run_chat(
                session,
                engine._profile["endpoint"],
                dict(engine._profile.get("parameters", {})),
                prompt,
                item,
                media_type,
            )
            dest = dest_dir / f"{ts}-{stem}-{idx}.txt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            engine._emit(
                f"{prefix} 💾 Saved: {dest.name}",
                current=current,
                total=total,
                saved_path=dest,
            )
            state.note_success()
            return [
                OutputFile(
                    source_path=item.path,
                    path=dest,
                    status="ok",
                    media_type=media_type,
                )
            ]

        engine._emit(f"{prefix} 📡 Calling {engine.PROVIDER_NAME} API...")
        payload = engine._build_task_payload(
            dict(engine._profile.get("parameters", {})),
            prompt,
            item,
            engine._profile["endpoint"],
        )
        saved = engine._run_task(
            session, payload, media_type, stem, idx, prefix, current, total, rel_dir, expected, made
        )
        if not saved:
            return [error_result(
                engine, item, idx, total, media_type, expected, "no_output",
                "No output returned from Evolink.AI", prompt=prompt, payload=payload, made=made,
            )]
        state.note_success()
        return [
            OutputFile(
                source_path=item.path,
                path=saved_path,
                status="ok",
                media_type=media_type,
                metadata={"api_payload": payload},
            )
            for saved_path in saved
        ]
    except Exception as exc:
        _record_failure(engine, state, exc, prefix)
        engine._emit(f"{prefix} Error: {exc}", level="error", current=current, total=total)
        return [error_result(
            engine, item, idx, total, media_type, expected, "provider_call",
            str(exc), prompt=prompt, payload=payload, made=made, error=exc,
        )]


def run_batch(engine, inputs) -> list[OutputFile]:
    import requests

    engine._output_dir.mkdir(parents=True, exist_ok=True)
    state = BatchState(int(engine._profile.get("circuit_breaker_max", 3)))
    results: list[list[OutputFile] | None] = [None] * len(inputs)
    inflight_cap = max(1, engine._concurrency)
    total = len(inputs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=inflight_cap) as pool:
        pending: dict[concurrent.futures.Future, int] = {}
        next_idx = 0
        while pending or (next_idx < total and not state.is_stopped()):
            while (
                not state.is_stopped()
                and next_idx < total
                and len(pending) < inflight_cap
            ):
                item = inputs[next_idx]
                prefix = f"[{next_idx + 1}/{total}] {item.path.name}"
                session = requests.Session()
                session.headers["Authorization"] = (
                    f"Bearer {engine._resolve_api_key()}"
                )
                session = RetryingSession(session, engine, prefix)
                future = pool.submit(
                    run_item, engine, session, item, next_idx, total, state
                )
                pending[future] = next_idx
                next_idx += 1
            if not pending:
                break
            done, _ = concurrent.futures.wait(
                pending, return_when="FIRST_COMPLETED"
            )
            for future in done:
                results[pending.pop(future)] = future.result()

    return [out for item in results if item is not None for out in item]
