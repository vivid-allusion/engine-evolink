"""N-in-flight proofs — mirrored from the gold engine's stub suite (phase_3).

Both legs share one output-name rule (ts-stem-idx built by the engine's
own _now_string/_default_extension helpers), so the concurrency-1 identity
check compares real equals. Stub-only: no real poll loop, no network.
"""

import sys
import threading
import time
from datetime import datetime as _dt
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine_evolink.runner as runner_module  # noqa: E402


def _stub_session(raise_):
    class _Resp:
        ok = False
        status_code = 500

    session = MagicMock()
    session.post.return_value = _Resp.__new__(_Resp)
    return session

from engine_evolink import Engine, EngineError, InputFile  # noqa: E402


class _Frozen(_dt):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 18, 12, 0, 0)


def inputs_stub(n=3):
    return [InputFile(path=Path(f"b{i}.md"), prompt=f"p{i}") for i in range(n)]


def base_profile(**extra):
    return {"endpoint": "test/model", "media_type": "image", **extra}


def _dest(engine, item, idx):
    ts = engine._now_string()
    ext = engine._default_extension(engine._profile.get("media_type", "image"))
    return engine._output_dir / f"{ts}-{item.path.stem}-{idx}{ext}"


def _saved_paths(engine, items):
    return [_dest(engine, item, idx) for idx, item in enumerate(items)]


def run_engine(profile, out_dir, behavior, n=3, frozen=True):
    """behavior(session_post=None) receives the probe; returns saved paths.

    The parallel branch swaps `runner.run_item` for `watching` (a wrapped
    run_item replacement); the sequential route gets the same patch on the
    engine's _run_task/_run_chat entry — one naming rule in both.
    """
    mock_requests = MagicMock()
    mock_session = MagicMock()
    mock_requests.Session.return_value = mock_session
    recorder = []
    watch = {"cur": 0, "max": 0, "lock": threading.Lock()}

    def saved_outputs(engine_args):
        from engine_evolink.datatypes import OutputFile

        engine, items = engine_args
        return [
            OutputFile(
                source_path=item.path,
                path=dest,
                status="ok",
                media_type="image",
            )
            for item, dest in zip(items, _saved_paths(engine, items))
        ]

    seq_items = inputs_stub(n)
    par_items = seq_items

    def same_paths(engine, items):
        return _saved_paths(engine, items)

    real_run_item = runner_module.run_item
    def watching(engine, session, item, idx, total, state):
        with watch["lock"]:
            watch["cur"] += 1
            watch["max"] = max(watch["max"], watch["cur"])
        try:
            time.sleep(0.02 + (hash(threading.current_thread()) % 11) * 0.002)
            raise_ = behavior(index=idx, item=item)
            if raise_ is not None:
                session = _stub_session(raise_)
                return real_run_item(engine, session, item, idx, total, state)
            from engine_evolink.datatypes import OutputFile

            return [
                OutputFile(
                    source_path=item.path,
                    path=_dest(engine, item, idx),
                    status="ok",
                    media_type="image",
                )
            ]
        finally:
            with watch["lock"]:
                watch["cur"] -= 1

    patches = [
        patch.dict("os.environ", {"EVOLINK_API_KEY": "test_key"}),
        patch.dict("sys.modules", {"requests": mock_requests}),
    ]
    if frozen:
        patches.append(patch("engine_evolink.engine.datetime", _Frozen))
    with patches[0], patches[1]:
        ctx = patches[2] if frozen else _NullCtx()
        with ctx:
            engine = Engine(profile, out_dir, on_progress=recorder.append)
            items = inputs_stub(n)

            def seq_task(session, payload, media_type, stem, idx, prefix, current, total, rel_dir, expected):
                raise_ = behavior(index=idx, item=items[idx])
                if raise_ is not None:
                    raise raise_
                return [_dest(engine, items[idx], idx)]

            engine._run_task = seq_task
            engine._run_chat = lambda session, *args, **kwargs: "chat answer"

            seq_results = engine.run(inputs_stub(n))
            with patch("engine_evolink.runner.run_item", watching):
                par_results = engine.run(items)
    return seq_results, par_results, watch


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestParallelRunner:
    def test_result_set_and_order_identical_to_own_concurrency_one(self, tmp_path):
        seq_results, par_results, _ = run_engine(
            base_profile(), tmp_path, lambda index, item: None
        )
        assert [r.status for r in par_results] == [r.status for r in seq_results]
        assert [r.source_path for r in par_results] == [
            r.source_path for r in seq_results
        ]
        assert [r.path.name for r in par_results] == [r.path.name for r in seq_results]
        assert len(par_results) == 3

    def test_more_than_one_job_in_flight(self, tmp_path):
        _, _, watch = run_engine(
            base_profile(concurrency=3), tmp_path, lambda index, item: None
        )
        assert watch["max"] >= 2

    def test_never_more_than_n_in_flight(self, tmp_path):
        _, _, watch = run_engine(
            base_profile(concurrency=2), tmp_path, lambda index, item: None
        )
        assert watch["max"] <= 2

    def test_failing_item_leaves_in_flight_siblings_intact(self, tmp_path):
        def behavior(index, item):
            if item.prompt == "p1":
                return RuntimeError("evolink blew up")
            return None

        seq_results, par_results, _ = run_engine(
            base_profile(concurrency=3), tmp_path, behavior
        )
        assert [r.status for r in par_results] == ["ok", "error", "ok"]
        assert "evolink blew up" in par_results[1].error_msg
        assert [r.status for r in seq_results] == ["ok", "error", "ok"]

    def test_invalid_concurrency_raises_loudly(self, tmp_path):
        with patch.dict("os.environ", {"EVOLINK_API_KEY": "test_key"}):
            engine = Engine(base_profile(concurrency=1.5), tmp_path)
            with pytest.raises(EngineError, match="concurrency"):
                engine.run([])
