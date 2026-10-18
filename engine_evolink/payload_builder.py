"""Evolink payload and message shaping.
"""

from .datatypes import InputFile

_START_IMAGE_MODELS = {
    "kling-v3-image-to-video",
    "wan3.0-prime-image-to-video",
    "wan3.0-image-to-video",
    "wan2.7-image-to-video",
    "wan2.6-image-to-video",
    "wan2.5-image-to-video",
}


class EvolinkPayloadBuilder:
    """Shape provider requests from engine profiles and inputs."""

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
