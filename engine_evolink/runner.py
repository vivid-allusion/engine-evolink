"""N-in-flight execution — mirrors the gold engine's runner.

`run_batch` caps jobs in flight structurally: at most the profile's
`concurrency` futures are submitted at any moment. Per-item behaviour
reuses this engine's existing machinery verbatim (the sequential body's
chat and task branches) — same retry, polling and error semantics.
Results are written by input index; a failing item never disturbs jobs
still in flight; the stop flag exists for the mirror-shape contract
(this engine's dispatcher has no breaker to trip from its per-item code).
"""

import concurrent.futures
import threading

from .datatypes import OutputFile


class BatchState:
    def __init__(self):
        self._lock = threading.Lock()
        self._stop = False

    def is_stopped(self) -> bool:
        with self._lock:
            return self._stop

    def halt(self):
        with self._lock:
            self._stop = True

    def note_success(self):
        pass


def run_item(engine, session, item, idx, total, state) -> list[OutputFile]:
    from engine_evolink.engine import _CHAT_MEDIA_TYPES

    stem = item.path.stem
    current = idx + 1
    prefix = f"[{current}/{total}]"
    media_type = engine._profile.get("media_type", "") or "image"
    rel_dir = str(item.metadata.get("relative_dir", "") or "")
    dest_dir = engine._output_dir / rel_dir if rel_dir else engine._output_dir
    ts = engine._now_string()
    ext = engine._default_extension(media_type)
    expected = dest_dir / f"{ts}-{stem}-{idx}{ext}"

    prompt = f"{engine._prefix}{item.prompt}{engine._suffix}".strip()
    if not prompt:
        return [
            OutputFile(
                source_path=item.path,
                status="error",
                error_msg="Empty prompt after applying prefix/suffix",
                media_type=media_type,
                expected_path=expected,
            )
        ]

    engine._emit(f"{prefix} 📝 Prompt: {prompt}")

    try:
        if media_type in _CHAT_MEDIA_TYPES:
            engine._emit(f"{prefix} 📡 Calling {engine.PROVIDER_NAME} chat API...")
            content = engine._run_chat(session, engine._profile["endpoint"], dict(engine._profile.get("parameters", {})), prompt, item, media_type)
            dest = dest_dir / f"{ts}-{stem}-{idx}.txt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            engine._emit(f"{prefix} 💾 Saved: {dest.name}", current=current, total=total, saved_path=dest)
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
        payload = engine._build_task_payload(dict(engine._profile.get("parameters", {})), prompt, item, engine._profile["endpoint"])
        saved = engine._run_task(session, payload, media_type, stem, idx, prefix, current, total, rel_dir, expected)
        if not saved:
            return [
                OutputFile(
                    source_path=item.path,
                    status="error",
                    error_msg="No output returned from Evolink.AI",
                    media_type=media_type,
                    expected_path=expected,
                )
            ]
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
        engine._emit(f"{prefix} Error: {exc}", level="error", current=current, total=total)
        return [
            OutputFile(
                source_path=item.path,
                status="error",
                error_msg=str(exc),
                media_type=media_type,
                expected_path=expected,
            )
        ]


def run_batch(engine, inputs) -> list[OutputFile]:
    import requests

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {engine._resolve_api_key()}"

    state = BatchState()
    results: list[list[OutputFile] | None] = [None] * len(inputs)
    inflight_cap = max(1, engine._concurrency)

    with concurrent.futures.ThreadPoolExecutor(max_workers=inflight_cap) as pool:
        pending: dict[concurrent.futures.Future, int] = {}
        next_idx = 0
        while pending or (next_idx < len(inputs) and not state.is_stopped()):
            while (
                not state.is_stopped()
                and next_idx < len(inputs)
                and len(pending) < inflight_cap
            ):
                future = pool.submit(
                    run_item, engine, session, inputs[next_idx], next_idx, len(inputs), state
                )
                pending[future] = next_idx
                next_idx += 1
            if not pending:
                break
            done, _ = concurrent.futures.wait(pending, return_when="FIRST_COMPLETED")
            for future in done:
                results[pending.pop(future)] = future.result()

    return [out for item in results if item is not None for out in item]
