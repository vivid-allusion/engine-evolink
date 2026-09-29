import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine_evolink.metadata as meta  # noqa: E402
from engine_evolink import Engine, EngineError, InputFile, OutputFile, ProgressEvent  # noqa: E402


class _FakeStream:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _task_response(payload: dict) -> MagicMock:
    resp = MagicMock(ok=True)
    resp.json.return_value = payload
    return resp


@contextmanager
def _requests_mock():
    mock_requests = MagicMock()
    mock_session = MagicMock()
    mock_requests.Session.return_value = mock_session
    with patch.dict("sys.modules", {"requests": mock_requests}):
        yield mock_requests, mock_session


def _completed_task(urls: list[str]) -> dict:
    return {"status": "completed", "progress": 100, "results": urls}


class TestDatatypes:
    def test_inputfile_defaults(self):
        f = InputFile(path=Path("test.md"), prompt="hello")
        assert f.path == Path("test.md")
        assert f.prompt == "hello"
        assert f.reference_urls == []
        assert f.metadata == {}

    def test_outputfile_defaults(self):
        o = OutputFile(source_path=Path("test.md"))
        assert o.source_path == Path("test.md")
        assert o.path is None
        assert o.status == "ok"
        assert o.error_msg == ""
        assert o.media_type == ""
        assert o.metadata == {}
        assert o.expected_path is None

    def test_outputfile_error_status(self):
        o = OutputFile(
            source_path=Path("test.md"),
            status="error",
            error_msg="timeout",
            media_type="image",
        )
        assert o.status == "error"
        assert o.error_msg == "timeout"
        assert o.media_type == "image"

    def test_progress_event_defaults(self):
        e = ProgressEvent(message="processing")
        assert e.message == "processing"
        assert e.level == "info"

    def test_engine_error_is_exception(self):
        with pytest.raises(EngineError):
            raise EngineError("test")


class TestMetadata:
    def test_provider_name(self):
        assert meta.PROVIDER_NAME == "Evolink.AI"

    def test_platform(self):
        assert meta.PLATFORM == "evolink"

    def test_api_key_env_var(self):
        assert meta.API_KEY_ENV_VAR == "EVOLINK_API_KEY"

    def test_provider_homepage(self):
        assert meta.PROVIDER_HOMEPAGE == "https://evolink.ai"

    def test_metadata_matches_engine_class(self):
        assert Engine.PLATFORM == meta.PLATFORM
        assert Engine.PROVIDER_NAME == meta.PROVIDER_NAME
        assert Engine.API_KEY_ENV_VAR == meta.API_KEY_ENV_VAR
        assert Engine.API_KEY_PATTERN == meta.API_KEY_PATTERN


class TestEngineInitPurity:
    def test_init_stores_attributes(self):
        profile = {"endpoint": "test/model", "media_type": "image"}
        engine = Engine(profile, "/tmp/out")
        assert engine._profile == profile
        assert engine._output_dir == Path("/tmp/out")
        assert engine._api_key is None
        assert engine._on_progress is None
        assert engine._prefix == ""
        assert engine._suffix == ""
        assert engine._base_url == "https://api.evolink.ai"
        assert engine._poll_interval == 2.0
        assert engine._timeout == 0.0

    def test_init_extracts_prefix_suffix(self):
        profile = {
            "endpoint": "test/model",
            "prompt_prefix": "Turn this into ",
            "prompt_suffix": " in oil painting style",
        }
        engine = Engine(profile, "/tmp/out")
        assert engine._prefix == "Turn this into "
        assert engine._suffix == " in oil painting style"

    def test_init_base_url_override(self):
        engine = Engine({"endpoint": "m", "base_url": "https://direct.evolink.ai/"}, "/tmp/out")
        assert engine._base_url == "https://direct.evolink.ai"

    def test_init_poll_and_timeout_from_profile(self):
        engine = Engine({"endpoint": "m", "poll_interval": 5, "timeout_seconds": "600"}, "/tmp/out")
        assert engine._poll_interval == 5.0
        assert engine._timeout == 600.0


class TestEnginePreflight:
    def test_missing_endpoint_raises(self):
        engine = Engine({"media_type": "image"}, "/tmp/out")
        with pytest.raises(EngineError, match="Missing or empty 'endpoint'"):
            engine.run([])

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_raises(self):
        with patch.dict("sys.modules", {"requests": MagicMock()}):
            engine = Engine({"endpoint": "test/model"}, "/tmp/out")
            with pytest.raises(EngineError, match="not set"):
                engine.run([])

    def test_requests_not_installed_raises(self):
        with patch.dict("sys.modules", {"requests": None}):
            engine = Engine({"endpoint": "test/model"}, "/tmp/out")
            with pytest.raises(EngineError, match="requests not installed"):
                engine.run([])


class TestEngineImageTask:
    def _run(
        self,
        tmp_path,
        inputs,
        profile=None,
        on_progress=None,
    ):
        with _requests_mock() as (mock_requests, mock_session):
            mock_session.post.return_value = _task_response({"id": "task-1"})
            mock_session.get.return_value = _task_response(_completed_task(["https://x/out.png"]))
            full_profile = {"endpoint": "test/model", "media_type": "image"}
            full_profile.update(profile or {})
            with patch.dict("os.environ", {"EVOLINK_API_KEY": "test-key"}):
                with patch("time.sleep"):
                    with patch("urllib.request.urlopen", return_value=_FakeStream(b"\x89PNG\r\n\x1a\npngdata")):
                        engine = Engine(full_profile, tmp_path, on_progress=on_progress)
                        results = engine.run(inputs)
                        return results, mock_session

    def test_empty_inputs(self, tmp_path):
        results, _ = self._run(tmp_path, [])
        assert results == []

    def test_run_calls_progress_callback(self, tmp_path):
        progress_calls = []
        results, _ = self._run(
            tmp_path,
            [InputFile(path=Path("b.md"), prompt="test")],
            on_progress=progress_calls.append,
        )
        assert len(progress_calls) >= 1
        assert any("Calling Evolink.AI API" in c.message for c in progress_calls)

    def test_applies_prefix_suffix(self, tmp_path):
        results, session = self._run(
            tmp_path,
            [InputFile(path=Path("b.md"), prompt="hello")],
            profile={"prompt_prefix": "PREFIX: ", "prompt_suffix": " :SUFFIX"},
        )
        prompt_sent = session.post.call_args.kwargs["json"]["prompt"]
        assert prompt_sent == "PREFIX: hello :SUFFIX"

    def test_empty_prompt_after_prefix_suffix_is_error(self, tmp_path):
        results, session = self._run(tmp_path, [InputFile(path=Path("b.md"), prompt="  ")])
        assert len(results) == 1
        assert results[0].status == "error"
        assert "Empty prompt" in results[0].error_msg
        session.post.assert_not_called()

    def test_per_markdown_file_error_returns_error_outputfile(self, tmp_path):
        with _requests_mock() as (mock_requests, mock_session):
            mock_session.post.side_effect = RuntimeError("API timeout")
            full_profile = {"endpoint": "test/model", "media_type": "image"}
            with patch.dict("os.environ", {"EVOLINK_API_KEY": "test-key"}):
                engine = Engine(full_profile, tmp_path)
                results = engine.run([InputFile(path=Path("b.md"), prompt="test")])
        assert len(results) == 1
        assert results[0].status == "error"
        assert "API timeout" in results[0].error_msg

    def test_partial_success_mixed_batch(self, tmp_path):
        with _requests_mock() as (mock_requests, mock_session):
            ok_resp = _task_response({"id": "task-1"})
            mock_session.post.side_effect = [
                ok_resp,
                RuntimeError("fail"),
                ok_resp,
            ]
            mock_session.get.return_value = _task_response(_completed_task(["https://x/out.png"]))
            full_profile = {"endpoint": "test/model", "media_type": "image"}
            with patch.dict("os.environ", {"EVOLINK_API_KEY": "test-key"}):
                with patch("time.sleep"):
                    with patch("urllib.request.urlopen", return_value=_FakeStream(b"\x89PNG\r\n\x1a\npngdata")):
                        engine = Engine(full_profile, tmp_path)
                        markdown_files = [InputFile(path=Path(f"b{i}.md"), prompt="test") for i in range(3)]
                        results = engine.run(markdown_files)
        statuses = [r.status for r in results]
        assert statuses.count("ok") == 2
        assert statuses.count("error") == 1

    def test_completed_task_downloads_results(self, tmp_path):
        results, _ = self._run(tmp_path, [InputFile(path=Path("b.md"), prompt="test")])
        assert len(results) == 1
        assert results[0].status == "ok"
        assert results[0].path is not None
        assert results[0].path.suffix == ".png"
        assert results[0].path.read_bytes() == b"\x89PNG\r\n\x1a\npngdata"

    def test_task_poll_emits_progress(self, tmp_path):
        progress_calls = []
        with _requests_mock() as (mock_requests, mock_session):
            mock_session.post.return_value = _task_response({"id": "task-1"})
            mock_session.get.side_effect = [
                _task_response({"status": "processing", "progress": 42}),
                _task_response(_completed_task(["https://x/out.png"])),
            ]
            full_profile = {"endpoint": "test/model", "media_type": "image"}
            with patch.dict("os.environ", {"EVOLINK_API_KEY": "test-key"}):
                with patch("time.sleep"):
                    with patch("urllib.request.urlopen", return_value=_FakeStream(b"\x89PNG\r\n\x1a\npngdata")):
                        engine = Engine(full_profile, tmp_path, on_progress=progress_calls.append)
                        engine.run([InputFile(path=Path("b.md"), prompt="test")])
        messages = [c.message for c in progress_calls]
        assert any("Task created: task-1" in m for m in messages)
        assert any("processing (42%)" in m for m in messages)
        assert any("Task completed" in m for m in messages)

    def test_failed_task_returns_error(self, tmp_path):
        with _requests_mock() as (mock_requests, mock_session):
            mock_session.post.return_value = _task_response({"id": "task-1"})
            mock_session.get.return_value = _task_response(
                {
                    "status": "failed",
                    "error": {"code": "content_policy_violation", "message": "Blocked by safety filters"},
                }
            )
            full_profile = {"endpoint": "test/model", "media_type": "image"}
            with patch.dict("os.environ", {"EVOLINK_API_KEY": "test-key"}):
                with patch("time.sleep"):
                    engine = Engine(full_profile, tmp_path)
                    results = engine.run([InputFile(path=Path("b.md"), prompt="test")])
        assert results[0].status == "error"
        assert "Blocked by safety filters" in results[0].error_msg

    def test_empty_results_returns_error(self, tmp_path):
        with _requests_mock() as (mock_requests, mock_session):
            mock_session.post.return_value = _task_response({"id": "task-1"})
            mock_session.get.return_value = _task_response(_completed_task([]))
            full_profile = {"endpoint": "test/model", "media_type": "image"}
            with patch.dict("os.environ", {"EVOLINK_API_KEY": "test-key"}):
                with patch("time.sleep"):
                    engine = Engine(full_profile, tmp_path)
                    results = engine.run([InputFile(path=Path("b.md"), prompt="test")])
        assert results[0].status == "error"
        assert "No output" in results[0].error_msg

    def test_mirrors_relative_dir(self, tmp_path):
        results, _ = self._run(
            tmp_path,
            [InputFile(path=Path("b.md"), prompt="test", metadata={"relative_dir": "scene1/nested"})],
        )
        assert results[0].status == "ok"
        assert results[0].path.parent == tmp_path / "scene1" / "nested"

    def test_reference_image_urls_in_payload(self, tmp_path):
        results, session = self._run(
            tmp_path,
            [
                InputFile(
                    path=Path("b.md"),
                    prompt="test",
                    reference_urls=["https://ref.com/a.jpg", "https://ref.com/b.jpg"],
                )
            ],
        )
        payload = session.post.call_args.kwargs["json"]
        assert payload["image_urls"] == ["https://ref.com/a.jpg", "https://ref.com/b.jpg"]

    def test_start_image_models_use_image_start_and_end(self, tmp_path):
        results, session = self._run(
            tmp_path,
            [
                InputFile(
                    path=Path("b.md"),
                    prompt="test",
                    reference_urls=["https://ref.com/first.jpg", "https://ref.com/last.jpg"],
                )
            ],
            profile={"endpoint": "kling-v3-image-to-video", "media_type": "video"},
        )
        payload = session.post.call_args.kwargs["json"]
        assert payload["image_start"] == "https://ref.com/first.jpg"
        assert payload["image_end"] == "https://ref.com/last.jpg"
        assert "image_urls" not in payload

    def test_reference_param_profile_override(self, tmp_path):
        results, session = self._run(
            tmp_path,
            [InputFile(path=Path("b.md"), prompt="test", reference_urls=["https://ref.com/a.jpg"])],
            profile={"reference_param": "custom_ref"},
        )
        payload = session.post.call_args.kwargs["json"]
        assert payload["custom_ref"] == ["https://ref.com/a.jpg"]

    def test_nested_params_split_into_objects(self, tmp_path):
        results, session = self._run(
            tmp_path,
            [InputFile(path=Path("b.md"), prompt="test")],
            profile={"parameters": {"quality": "2K", "model_params.web_search": True, "prompt_prefix": "x"}},
        )
        payload = session.post.call_args.kwargs["json"]
        assert payload["quality"] == "2K"
        assert payload["model_params"] == {"web_search": True}
        assert "prompt_prefix" not in payload

    def test_post_http_error_surfaces_message(self, tmp_path):
        with _requests_mock() as (mock_requests, mock_session):
            error_resp = MagicMock(ok=False, status_code=402)
            error_resp.json.return_value = {"error": {"message": "Insufficient quota"}}
            mock_session.post.return_value = error_resp
            full_profile = {"endpoint": "test/model", "media_type": "image"}
            with patch.dict("os.environ", {"EVOLINK_API_KEY": "test-key"}):
                engine = Engine(full_profile, tmp_path)
                results = engine.run([InputFile(path=Path("b.md"), prompt="test")])
        assert results[0].status == "error"
        assert "Insufficient quota" in results[0].error_msg


class TestEngineChat:
    def _run_chat(self, tmp_path, media_type, profile=None, on_progress=None):
        with _requests_mock() as (mock_requests, mock_session):
            chat_resp = MagicMock(ok=True)
            chat_resp.json.return_value = {"choices": [{"message": {"content": "hello world"}}]}
            mock_session.post.return_value = chat_resp
            full_profile = {"endpoint": "deepseek-v4-flash", "media_type": media_type}
            full_profile.update(profile or {})
            with patch.dict("os.environ", {"EVOLINK_API_KEY": "test-key"}):
                engine = Engine(full_profile, tmp_path, on_progress=on_progress)
                results = engine.run([InputFile(path=Path("b.md"), prompt="test")])
                return results, mock_session

    def test_text_writes_txt_file(self, tmp_path):
        results, session = self._run_chat(tmp_path, "text")
        assert results[0].status == "ok"
        assert results[0].path.suffix == ".txt"
        assert results[0].path.read_text(encoding="utf-8") == "hello world"
        payload = session.post.call_args.kwargs["json"]
        assert payload["model"] == "deepseek-v4-flash"
        assert payload["messages"] == [{"role": "user", "content": "test"}]
        assert session.post.call_args.args[0].endswith("/v1/chat/completions")

    def test_chat_uses_direct_base_url(self, tmp_path):
        results, session = self._run_chat(tmp_path, "text", profile={"base_url": "https://direct.evolink.ai"})
        assert session.post.call_args.args[0] == "https://direct.evolink.ai/v1/chat/completions"

    def test_vision_builds_image_content_blocks(self, tmp_path):
        with _requests_mock() as (mock_requests, mock_session):
            chat_resp = MagicMock(ok=True)
            chat_resp.json.return_value = {"choices": [{"message": {"content": "hello world"}}]}
            mock_session.post.return_value = chat_resp
            full_profile = {"endpoint": "deepseek-v4-flash-vision-exp", "media_type": "vision"}
            with patch.dict("os.environ", {"EVOLINK_API_KEY": "test-key"}):
                engine = Engine(full_profile, tmp_path)
                results = engine.run(
                    [
                        InputFile(
                            path=Path("b.md"),
                            prompt="test",
                            reference_urls=["https://ref.com/a.jpg"],
                        )
                    ]
                )
        messages = mock_session.post.call_args.kwargs["json"]["messages"]
        content = messages[0]["content"]
        assert content == [
            {"type": "text", "text": "test"},
            {"type": "image_url", "image_url": {"url": "https://ref.com/a.jpg"}},
        ]
        assert results[0].status == "ok"

    def test_vision_without_reference_urls_is_plain_text(self, tmp_path):
        results, session = self._run_chat(tmp_path, "vision")
        content = session.post.call_args.kwargs["json"]["messages"][0]["content"]
        assert content == "test"

    def test_chat_http_error_surfaces_message(self, tmp_path):
        with _requests_mock() as (mock_requests, mock_session):
            error_resp = MagicMock(ok=False, status_code=429)
            error_resp.json.return_value = {"error": {"message": "Rate limit exceeded"}}
            mock_session.post.return_value = error_resp
            full_profile = {"endpoint": "deepseek-v4-flash", "media_type": "text"}
            with patch.dict("os.environ", {"EVOLINK_API_KEY": "test-key"}):
                engine = Engine(full_profile, tmp_path)
                results = engine.run([InputFile(path=Path("b.md"), prompt="test")])
        assert results[0].status == "error"
        assert "Rate limit exceeded" in results[0].error_msg

    def test_chat_empty_content_returns_error(self, tmp_path):
        with _requests_mock() as (mock_requests, mock_session):
            chat_resp = MagicMock(ok=True)
            chat_resp.json.return_value = {"choices": [{"message": {"content": ""}}]}
            mock_session.post.return_value = chat_resp
            full_profile = {"endpoint": "deepseek-v4-flash", "media_type": "text"}
            with patch.dict("os.environ", {"EVOLINK_API_KEY": "test-key"}):
                engine = Engine(full_profile, tmp_path)
                results = engine.run([InputFile(path=Path("b.md"), prompt="test")])
        assert results[0].status == "error"
        assert "Empty content" in results[0].error_msg


class TestStreamSaveExtension:
    def _run_stream(self, tmp_path, data: bytes, profile: dict | None = None):
        with _requests_mock() as (mock_requests, mock_session):
            mock_session.post.return_value = _task_response({"id": "task-1"})
            mock_session.get.return_value = _task_response(_completed_task(["https://x/out.bin"]))
            full_profile = {"endpoint": "test/model", "media_type": "image"}
            full_profile.update(profile or {})
            with patch.dict("os.environ", {"EVOLINK_API_KEY": "test-key"}):
                with patch("time.sleep"):
                    with patch("urllib.request.urlopen", return_value=_FakeStream(data)):
                        engine = Engine(full_profile, tmp_path)
                        return engine.run([InputFile(path=Path("b.md"), prompt="test")])

    def test_jpeg_bytes_saved_with_jpg_extension(self, tmp_path):
        results = self._run_stream(tmp_path, b"\xff\xd8\xff" + b"jpegdata")
        assert results[0].status == "ok"
        assert results[0].path.suffix == ".jpg"

    def test_png_bytes_saved_with_png_extension(self, tmp_path):
        results = self._run_stream(tmp_path, b"\x89PNG\r\n\x1a\n" + b"pngdata")
        assert results[0].path.suffix == ".png"

    def test_gif_bytes_saved_with_gif_extension(self, tmp_path):
        results = self._run_stream(tmp_path, b"GIF89a" + b"gifdata")
        assert results[0].path.suffix == ".gif"

    def test_webp_bytes_saved_with_webp_extension(self, tmp_path):
        results = self._run_stream(tmp_path, b"RIFF\x00\x00\x00\x00WEBP" + b"data")
        assert results[0].path.suffix == ".webp"

    def test_mp4_bytes_saved_with_mp4_extension(self, tmp_path):
        results = self._run_stream(tmp_path, b"\x00\x00\x00\x18ftyp" + b"mp4data")
        assert results[0].path.suffix == ".mp4"

    def test_mp3_bytes_saved_with_mp3_extension(self, tmp_path):
        results = self._run_stream(tmp_path, b"ID3" + b"mp3data", profile={"media_type": "audio"})
        assert results[0].path.suffix == ".mp3"

    def test_unknown_bytes_use_output_format_param(self, tmp_path):
        results = self._run_stream(
            tmp_path,
            b"\xde\xad\xbe\xef",
            profile={"parameters": {"output_format": "jpg"}},
        )
        assert results[0].path.suffix == ".jpg"

    def test_unknown_bytes_fall_back_to_media_type_default(self, tmp_path):
        results = self._run_stream(tmp_path, b"\xde\xad\xbe\xef")
        assert results[0].path.suffix == ".png"


class TestExpectedPath:
    def _run_error(self, tmp_path, profile: dict | None = None):
        with _requests_mock() as (mock_requests, mock_session):
            mock_session.post.side_effect = RuntimeError("API timeout")
            full_profile = {"endpoint": "test/model", "media_type": "image"}
            full_profile.update(profile or {})
            with patch.dict("os.environ", {"EVOLINK_API_KEY": "test-key"}):
                engine = Engine(full_profile, tmp_path)
                return engine.run([InputFile(path=Path("b.md"), prompt="test")])

    def test_exception_error_carries_expected_path(self, tmp_path):
        results = self._run_error(tmp_path)
        assert results[0].status == "error"
        assert results[0].expected_path is not None
        assert results[0].expected_path.parent == tmp_path
        assert results[0].expected_path.name.startswith("2")
        assert results[0].expected_path.stem.endswith("-b-0")

    def test_expected_path_uses_output_format_extension(self, tmp_path):
        results = self._run_error(tmp_path, profile={"parameters": {"output_format": "jpg"}})
        assert results[0].expected_path.suffix == ".jpg"

    def test_empty_prompt_error_carries_expected_path(self, tmp_path):
        with _requests_mock() as (mock_requests, mock_session):
            full_profile = {"endpoint": "test/model", "media_type": "image"}
            with patch.dict("os.environ", {"EVOLINK_API_KEY": "test-key"}):
                engine = Engine(full_profile, tmp_path)
                results = engine.run([InputFile(path=Path("b.md"), prompt="  ")])
        assert results[0].status == "error"
        assert results[0].expected_path is not None
        assert results[0].expected_path.stem.endswith("-b-0")


class TestImports:
    def test_init_exports_all_names(self):
        from engine_evolink import (
            Engine,
            EngineError,
            InputFile,
            OutputFile,
            ProgressEvent,
        )

        assert Engine is not None
        assert InputFile is not None
        assert OutputFile is not None
        assert ProgressEvent is not None
        assert EngineError is not None

    def test_list_standby_profiles_empty(self):
        from engine_evolink import list_standby_profiles

        assert list_standby_profiles() == []
