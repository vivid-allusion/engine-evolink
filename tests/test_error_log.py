from pathlib import Path
from unittest.mock import MagicMock, patch

from engine_evolink import Engine, InputFile


class _Resp:
    ok = True
    status_code = 200

    def __init__(self, body):
        self._b = body

    def json(self):
        return self._b


def _session(task):
    s = MagicMock()
    s.post.return_value = _Resp({"id": "task-9"})
    s.get.return_value = _Resp(task)
    return s


def _run(tmp_path, concurrency, prompt="a fox", task=None):
    profile = {"endpoint": "m", "media_type": "video", "poll_interval": 0, "concurrency": concurrency,
               "parameters": {"api_key": "sk-secret"}}
    src = tmp_path / "S01.md"
    src.write_text(prompt)
    eng = Engine(profile, tmp_path / "out", api_key="k")
    sess = _session(task or {})
    fake = MagicMock()
    fake.Session.return_value = sess
    with patch.dict("sys.modules", {"requests": fake}):
        return eng.run([InputFile(path=src, prompt=prompt)])[0]


def test_failed_task_logs_reason_and_task_id(tmp_path):
    for conc in (1, 2):
        (tmp_path / str(conc)).mkdir()
        r = _run(tmp_path / str(conc), conc, task={"status": "failed", "error": {"message": "moderated"}})
        log = Path(r.metadata["error_log"])
        text = log.read_text()
        assert log.name.endswith("-0_ERROR.log") and log.parent == tmp_path / str(conc) / "out"
        assert "task-9" in text and "moderated" in text and "S01.md" in text and "item: 1 of 1" in text
        assert "sk-secret" not in text


def test_empty_prompt_logs(tmp_path):
    r = _run(tmp_path, 1, prompt="")
    assert "empty_prompt" in Path(r.metadata["error_log"]).read_text()


def test_log_write_failure_never_raises(tmp_path):
    with patch("engine_evolink.error_log._mask", side_effect=OSError):
        r = _run(tmp_path, 1, prompt="")
    assert r.status == "error" and r.metadata == {}


def test_failed_chat_item_logs_masked_request(tmp_path):
    profile = {"endpoint": "chat-m", "media_type": "text", "parameters": {"api_key": "sk-secret"}}
    src = tmp_path / "S01.md"
    src.write_text("hello")
    sess = MagicMock()
    sess.post.return_value = _Resp({"choices": []})
    fake = MagicMock()
    fake.Session.return_value = sess
    with patch.dict("sys.modules", {"requests": fake}):
        r = Engine(profile, tmp_path / "out", api_key="k").run([InputFile(path=src, prompt="hello")])[0]
    text = Path(r.metadata["error_log"]).read_text()
    assert '"model": "chat-m"' in text and "hello" in text and "***MASKED***" in text
    assert "sk-secret" not in text
