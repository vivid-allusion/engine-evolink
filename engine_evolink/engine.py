import importlib
import os
import time
import urllib.request
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from .datatypes import EngineError, InputFile, OutputFile, ProgressEvent

_TASK_PATHS = {
    "image": "/v1/images/generations",
    "video": "/v1/videos/generations",
    "audio": "/v1/audios/generations",
}

_START_IMAGE_MODELS = {
    "kling-v3-image-to-video",
    "wan3.0-prime-image-to-video",
    "wan3.0-image-to-video",
    "wan2.7-image-to-video",
    "wan2.6-image-to-video",
    "wan2.5-image-to-video",
}

_CHAT_MEDIA_TYPES = {"text", "vision"}

_DEFAULT_EXTENSIONS = {"image": ".png", "video": ".mp4", "audio": ".mp3", "text": ".txt", "vision": ".txt"}


class Engine:
    PLATFORM: str = "evolink"
    PROVIDER_NAME: str = "Evolink.AI"
    PROVIDER_HOMEPAGE: str = "https://evolink.ai"
    API_KEY_ENV_VAR: str = "EVOLINK_API_KEY"
    API_KEY_PATTERN: str = r".+"
    BASE_URL: str = "https://api.evolink.ai"

    def __init__(
        self,
        profile: dict,
        output_dir: str | Path,
        api_key: str | None = None,
        on_progress: Callable[[str], None] | None = None,
    ):
        self._profile = profile
        self._output_dir = Path(output_dir)
        self._api_key = api_key
        self._on_progress = on_progress
        self._prefix = profile.get("prompt_prefix", "")
        self._suffix = profile.get("prompt_suffix", "")
        self._base_url = (
            profile.get("base_url") or profile.get("api_base_url") or self.BASE_URL
        ).rstrip("/")
        try:
            self._poll_interval = float(profile.get("poll_interval", 2) or 2)
        except (TypeError, ValueError):
            self._poll_interval = 2.0
        try:
            self._timeout = float(profile.get("timeout_seconds") or 0)
        except (TypeError, ValueError):
            self._timeout = 0.0

    def run(self, inputs: list[InputFile]) -> list[OutputFile]:
        self._validate_preflight()

        import requests

        session = requests.Session()
        session.headers["Authorization"] = f"Bearer {self._resolve_api_key()}"

        endpoint = self._profile["endpoint"]
        params = dict(self._profile.get("parameters", {}))
        media_type = self._profile.get("media_type", "") or "image"
        self._output_dir.mkdir(parents=True, exist_ok=True)

        results: list[OutputFile] = []
        total = len(inputs)

        for idx, item in enumerate(inputs):
            stem = item.path.stem
            current = idx + 1
            prefix = f"[{current}/{total}]"
            rel_dir = str(item.metadata.get("relative_dir", "") or "")
            dest_dir = self._output_dir / rel_dir if rel_dir else self._output_dir
            ts = datetime.now().strftime("%y%m%d_%H%M%S")
            ext = self._default_extension(media_type)
            expected = dest_dir / f"{ts}-{stem}-{idx}{ext}"

            prompt = f"{self._prefix}{item.prompt}{self._suffix}".strip()
            if not prompt:
                results.append(
                    OutputFile(
                        source_path=item.path,
                        status="error",
                        error_msg="Empty prompt after applying prefix/suffix",
                        media_type=media_type,
                        expected_path=expected,
                    )
                )
                continue

            self._emit(f"{prefix} 📝 Prompt: {prompt}")

            try:
                if media_type in _CHAT_MEDIA_TYPES:
                    self._emit(f"{prefix} 📡 Calling Evolink.AI chat API...")
                    content = self._run_chat(session, endpoint, params, prompt, item, media_type)
                    dest = dest_dir / f"{ts}-{stem}-{idx}.txt"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(content, encoding="utf-8")
                    self._emit(f"{prefix} 💾 Saved: {dest.name}", current=current, total=total, saved_path=dest)
                    results.append(
                        OutputFile(
                            source_path=item.path,
                            path=dest,
                            status="ok",
                            media_type=media_type,
                        )
                    )
                    continue

                self._emit(f"{prefix} 📡 Calling Evolink.AI API...")
                payload = self._build_task_payload(params, prompt, item, endpoint)
                saved = self._run_task(session, payload, media_type, stem, idx, prefix, current, total, rel_dir, expected)
                if not saved:
                    results.append(
                        OutputFile(
                            source_path=item.path,
                            status="error",
                            error_msg="No output returned from Evolink.AI",
                            media_type=media_type,
                            expected_path=expected,
                        )
                    )
                else:
                    for saved_path in saved:
                        results.append(
                            OutputFile(
                                source_path=item.path,
                                path=saved_path,
                                status="ok",
                                media_type=media_type,
                                metadata={"api_payload": payload},
                            )
                        )
                    continue

            except Exception as exc:
                self._emit(f"{prefix} Error: {exc}", level="error", current=current, total=total)
                results.append(
                    OutputFile(
                        source_path=item.path,
                        status="error",
                        error_msg=str(exc),
                        media_type=media_type,
                        expected_path=expected,
                    )
                )

        return results

    def _run_task(
        self,
        session,
        payload: dict,
        media_type: str,
        stem: str,
        idx: int,
        prefix: str,
        current: int,
        total: int,
        rel_dir: str,
        expected: Path,
    ) -> list[Path]:
        task_path = _TASK_PATHS.get(media_type, _TASK_PATHS["image"])
        response = session.post(f"{self._base_url}{task_path}", json=payload, timeout=60)
        if not response.ok:
            raise RuntimeError(self._error_text(response, f"Evolink.AI HTTP {response.status_code}"))
        body = response.json()
        task_id = body.get("id") or body.get("task_id")
        if not task_id:
            raise RuntimeError("No task ID in Evolink.AI response")
        self._emit(f"{prefix} ⏳ Task created: {task_id}")

        started = time.monotonic()
        last_progress = None
        while True:
            if self._timeout and time.monotonic() - started > self._timeout:
                raise RuntimeError(f"Task {task_id} timed out after {self._timeout}s")
            task_response = session.get(f"{self._base_url}/v1/tasks/{task_id}", timeout=60)
            if not task_response.ok:
                raise RuntimeError(self._error_text(task_response, f"Task poll HTTP {task_response.status_code}"))
            task = task_response.json()
            status = task.get("status", "")
            progress = task.get("progress")
            if progress != last_progress:
                last_progress = progress
                self._emit(f"{prefix} ⏳ Task {status} ({progress}%)")
            if status == "completed":
                self._emit(f"{prefix} ✅ Task completed")
                urls = task.get("results") or []
                return self._save_results(urls, media_type, stem, idx, prefix, current, total, rel_dir, payload)
            if status == "failed":
                error = task.get("error") or {}
                raise RuntimeError(f"Task failed: {error.get('message') or error.get('code') or 'unknown error'}")
            time.sleep(self._poll_interval)

    def _run_chat(
        self,
        session,
        endpoint: str,
        params: dict,
        prompt: str,
        item: InputFile,
        media_type: str,
    ) -> str:
        body = self._split_nested(params, {"prompt_prefix", "prompt_suffix"})
        body["model"] = endpoint
        body["messages"] = self._build_messages(prompt, item, media_type)
        response = session.post(f"{self._base_url}/v1/chat/completions", json=body, timeout=120)
        if not response.ok:
            raise RuntimeError(self._error_text(response, f"Evolink.AI HTTP {response.status_code}"))
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError("No content in Evolink.AI chat response") from None
        if not content:
            raise RuntimeError("Empty content in Evolink.AI chat response")
        return content

    def _build_messages(self, prompt: str, item: InputFile, media_type: str) -> list[dict]:
        system = self._profile.get("system_prompt", "")
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        if media_type == "vision" and item.reference_urls:
            content_blocks: list[dict] = [{"type": "text", "text": prompt}]
            for url in item.reference_urls:
                content_blocks.append({"type": "image_url", "image_url": {"url": url}})
            messages.append({"role": "user", "content": content_blocks})
        else:
            messages.append({"role": "user", "content": prompt})
        return messages

    def _build_task_payload(self, params: dict, prompt: str, item: InputFile, endpoint: str) -> dict:
        payload = self._split_nested(params, {"prompt_prefix", "prompt_suffix"})
        payload["model"] = endpoint
        payload["prompt"] = prompt
        ref_key = self._reference_param(endpoint)
        if item.reference_urls and ref_key == "image_start":
            payload["image_start"] = item.reference_urls[0]
            if len(item.reference_urls) > 1:
                payload["image_end"] = item.reference_urls[1]
        elif item.reference_urls and ref_key:
            payload[ref_key] = item.reference_urls
        return payload

    def _reference_param(self, endpoint: str) -> str:
        explicit = self._profile.get("reference_param")
        if explicit:
            return explicit
        if endpoint in _START_IMAGE_MODELS:
            return "image_start"
        return "image_urls"

    def _split_nested(self, params: dict, skips: set) -> dict:
        out: dict = {}
        for key, value in params.items():
            if key in skips:
                continue
            if isinstance(key, str) and "." in key:
                head, tail = key.split(".", 1)
                bucket = out.setdefault(head, {})
                if isinstance(bucket, dict):
                    bucket[tail] = value
            else:
                out[key] = value
        return out

    def _save_results(
        self,
        urls,
        media_type: str,
        stem: str,
        idx: int,
        prefix: str,
        current: int,
        total: int,
        rel_dir: str,
        api_payload: dict | None = None,
    ) -> list[Path]:
        if not urls:
            return []
        dest_dir = self._output_dir / rel_dir if rel_dir else self._output_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        saved = []
        for i, url in enumerate(urls):
            if not isinstance(url, str):
                continue
            self._emit(f"{prefix} ⬇️  Downloading...")
            with urllib.request.urlopen(url, timeout=300) as stream:
                data = stream.read()
            ext = self._sniff_extension(data, media_type)
            suffix = "" if len(urls) == 1 else f"_{i}"
            dest = dest_dir / f"{ts}-{stem}-{idx}{suffix}{ext}"
            with open(dest, "wb") as f:
                f.write(data)
            self._emit(
                f"{prefix} 💾 Saved: {dest.name}",
                current=current,
                total=total,
                saved_path=dest,
                api_payload=api_payload,
            )
            saved.append(dest)
        return saved

    def _sniff_extension(self, data: bytes, media_type: str) -> str:
        for magic, ext in (
            (b"\xff\xd8\xff", ".jpg"),
            (b"\x89PNG\r\n\x1a\n", ".png"),
            (b"GIF87a", ".gif"),
            (b"GIF89a", ".gif"),
            (b"ID3", ".mp3"),
            (b"OggS", ".ogg"),
            (b"fLaC", ".flac"),
        ):
            if data.startswith(magic):
                return ext
        if data[:4] == b"RIFF" and data[8:12] in (b"WEBP", b"WAVE"):
            return ".webp" if data[8:12] == b"WEBP" else ".wav"
        if data[4:8] == b"ftyp":
            return ".mp4"
        output_format = (
            str(self._profile.get("parameters", {}).get("output_format", "") or "")
            .strip()
            .lstrip(".")
            .lower()
        )
        if output_format and all(c.isalnum() for c in output_format):
            return f".{output_format}"
        return _DEFAULT_EXTENSIONS.get(media_type, ".png")

    def _default_extension(self, media_type: str) -> str:
        output_format = (
            str(self._profile.get("parameters", {}).get("output_format", "") or "")
            .strip()
            .lstrip(".")
            .lower()
        )
        if output_format and all(c.isalnum() for c in output_format):
            return f".{output_format}"
        return _DEFAULT_EXTENSIONS.get(media_type, ".png")

    def _error_text(self, response, fallback: str) -> str:
        try:
            data = response.json()
            error = data.get("error") or {}
            message = error.get("message") or data.get("message") or ""
            return message or fallback
        except ValueError:
            return fallback

    def _validate_preflight(self):
        if not self._profile.get("endpoint"):
            raise EngineError("Missing or empty 'endpoint' in profile")
        try:
            importlib.import_module("requests")
        except ImportError:
            raise EngineError("requests not installed. Run: pip install requests") from None
        if not self._resolve_api_key():
            raise EngineError(f"{self.API_KEY_ENV_VAR} not set in environment or .env file")

    def _resolve_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        return os.environ.get(self.API_KEY_ENV_VAR, "")

    def _emit(
        self,
        message: str,
        level: str = "info",
        current: int = 0,
        total: int = 0,
        saved_path: Path | None = None,
        api_payload: dict | None = None,
    ):
        if self._on_progress:
            self._on_progress(
                ProgressEvent(
                    message=message,
                    level=level,
                    current=current,
                    total=total,
                    saved_path=saved_path,
                    api_payload=api_payload,
                )
            )
