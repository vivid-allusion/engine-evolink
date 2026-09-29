## Architecture: Engine SDK Wrapper

This repo is an **Engine** in the studiolot ecosystem. It wraps the
**Evolink.AI** API and exposes a uniform interface that Vehicles call.

### The layers

```
studiolot (TUI) → Vehicle (script) → **Evolink.AI Engine** → Provider API
```

Vehicles like Frame Composer and Motion Conductor are SDK-agnostic. They
discover this Engine, load it via `engine_loader.py`, and call
`engine.run(inputs)`. This Engine handles all **Evolink.AI**-specific logic.

### Contract

- **`Engine.__init__` is PURE.** No network I/O in the constructor. It only
  stores: profile dict, output_dir, api_key, on_progress callback.
- **`Engine.run(inputs: list[InputFile]) -> list[OutputFile]`** is the ONLY
  entry point Vehicles call. Returns ALL results — success and failure —
  as OutputFile objects. Never raises for per-Markdown-file failures.
- **Error results carry `expected_path`**: the destination filename is
  precomputed BEFORE the API call (profile `output_format`/media default
  extension) so Vehicles can write error placeholders at the exact name.
- **`EngineError`** is for unrecoverable pre-flight failures only: missing
  API key, invalid profile.
- **`datatypes.py`** defines InputFile, OutputFile, ProgressEvent,
  EngineError — the gold engine-replicate interface, verbatim.
- **`metadata.py`** is zero-dependency (stdlib only). studiolot imports this
  (NOT engine.py) to read PROVIDER_NAME for the TUI label.
- **`endpoints/`** TOMLs are the model catalog. One file per model, defining
  valid parameter ranges. Curated flagship catalog — generated from the
  provider's API manual (engine-evolink-ai-docs repo).

### Provider details

| Field | Value |
|-------|-------|
| **Platform** | `evolink` |
| **API key env var** | `EVOLINK_API_KEY` |
| **Key pattern** | any non-empty string (docs specify no prefix) |
| **Homepage** | https://evolink.ai |
| **Base URL** | `https://api.evolink.ai` (text models: `https://direct.evolink.ai` recommended) |
| **API style** | Async task API for media (`POST /v1/{images,videos,audios}/generations` → poll `GET /v1/tasks/{id}`); OpenAI-compatible `POST /v1/chat/completions` for text/vision |

### Engine discovery

Vehicles find this Engine via `engine_loader.py`:
- Repo directory: `engine-evolink`
- Python package: `engine_evolink`
- `engine_loader.py` handles the hyphen→underscore mapping
- Discovery order: local clone in `00_APPLICATIONS/ENGINES/` first, pip-installed package as fallback

### Source File Map

| File | Purpose |
|------|---------|
| `engine_evolink/engine.py` | Engine implementation — all API calls live here |
| `engine_evolink/datatypes.py` | InputFile, OutputFile, ProgressEvent, EngineError |
| `engine_evolink/metadata.py` | Zero-dependency identity constants (studiolot reads this) |
| `engine_evolink/__init__.py` | Re-exports Engine + all datatypes |
| `engine_evolink/endpoints/` | TOML model catalog (IMG/VID/TXT/Vision) |
| `tests/test_engine.py` | Unit tests (interface + task flow, fully mocked) |
| `tests/test_endpoints.py` | TOML catalog integrity tests |
| `pyproject.toml` | Package metadata, pip-installable |
| `requirements.txt` | requests |

### Reference

Full contract: `~/Nextcloud/00-DEVELOPMENT/MISC_DEV_TOOLS/studiolot/docs/architecture/ENGINE_CONTRACT.md`
Docs snapshot: `~/Nextcloud/00-PRODUCTION/GAI_ENGINES/engine-evolink-ai-docs/`

### Session History

- 2026-09-03 — Created engine-evolink: gold-parity interface, task-based REST
  client, curated flagship TOML catalog (23 models) generated from the API
  manual, full mocked test suite.
