# engine-evolink

Engine wrapper for the [Evolink.AI](https://evolink.ai) API.

Part of the [studiolot](https://github.com/vivid-allusion/studiolot) Vehicle /
Engine / SDK architecture. This repo wraps the Evolink.AI REST API behind the
uniform interface that Vehicles (Frame Composer, Motion Conductor, etc.) call.
The Vehicle never talks to Evolink.AI directly — it calls this Engine.

See `docs/architecture/ENGINE_CONTRACT.md` in the studiolot repo for the full
interface contract.

## Quick start

```bash
git clone https://github.com/vivid-allusion/engine-evolink.git
cd engine-evolink
pip install -r requirements.txt
```

Or install via pip:

```bash
pip install engine-evolink
```

## Usage

```python
from engine_evolink import Engine, InputFile
from pathlib import Path

profile = {
    "platform": "evolink",
    "media_type": "image",
    "endpoint": "gemini-3.1-flash-image-preview",
    "parameters": {
        "size": "16:9",
        "quality": "2K",
    },
    "prompt_prefix": "",
    "prompt_suffix": "",
}

engine = Engine(profile=profile, output_dir="/tmp/out")

inputs = [
    InputFile(
        path=Path("bullet-001.md"),
        prompt="a cinematic shot of a city at night",
        reference_urls=["https://example.com/ref.jpg"],
    ),
]

results = engine.run(inputs)
for r in results:
    print(r.status, r.path)
```

## API key

Set `EVOLINK_API_KEY` in your environment or a `.env` file:

```bash
export EVOLINK_API_KEY=...
```

Get a key at: https://evolink.ai/dashboard/keys

## How the API works

Evolink.AI is a multi-provider gateway. Media endpoints
(`/v1/images/generations`, `/v1/videos/generations`, `/v1/audios/generations`)
are **asynchronous tasks**: the POST returns a task ID, the Engine polls
`GET /v1/tasks/{task_id}` until `completed` (result URLs, valid 24 hours —
downloaded immediately) or `failed` (task error surfaced per bullet).
Language endpoints are OpenAI-compatible `POST /v1/chat/completions`
(synchronous), used for `media_type` `text` and `vision`.

Base URL: `https://api.evolink.ai` (default). Override per profile with
`base_url` (e.g. `https://direct.evolink.ai` — recommended by the provider
for text models). Optional profile keys: `poll_interval` (seconds, default 2)
and `timeout_seconds` (task wait cap, 0 = no cap).

## Reference images

- Default reference parameter is `image_urls` (list of URLs) for both image
  and video models.
- Video models that take start/end frames (`kling-v3-image-to-video`,
  `wan*-image-to-video`) use `image_start` (first reference URL) and
  `image_end` (second reference URL).
- A profile key `reference_param` overrides the default.
- Vision models receive reference URLs as `image_url` content blocks in the
  chat messages.

Dotted parameter names (e.g. `model_params.web_search`,
`thinking.reasoning_effort`) in the instance binding are nested into the
request JSON automatically.

## Endpoint models

Endpoint TOML definitions live in `endpoints/`. Each file defines a model's
valid parameter ranges — the "bounds" that the AppWizard reads to build
select menus. Curated flagship catalog (full 200-page manual catalog is a
follow-up): 8 image, 6 video, 7 text, 2 vision models.

## Repo structure

```
engine-evolink/
├── __init__.py          ← re-exports Engine, InputFile, OutputFile, etc.
├── engine.py            ← Engine class — all Evolink.AI API calls live here
├── datatypes.py         ← InputFile, OutputFile, ProgressEvent, EngineError
├── metadata.py          ← zero-dependency constants (studiolot imports this)
├── endpoints/
│   ├── IMG-Models/      ← image model TOML definitions (8 models)
│   ├── VID-Models/      ← video model TOML definitions (6 models)
│   ├── TXT-Models/      ← text model TOML definitions (7 models)
│   └── Vision-Models/   ← vision model TOML definitions (2 models)
├── requirements.txt     ← requests>=2.28
├── .env.example         ← EVOLINK_API_KEY template
├── pyproject.toml       ← pip install engine-evolink
└── README.md
```

See `docs/architecture/ENGINE_CONTRACT.md` §6 in studiolot.
