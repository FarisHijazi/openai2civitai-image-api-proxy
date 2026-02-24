# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a Python FastAPI proxy that translates OpenAI DALL-E image generation API requests into CivitAI API requests. Single-service architecture, no database.

### Running the server

```bash
CIVITAI_API_TOKEN="$CIVITAI_API_TOKEN" uvicorn openai2civitai.server:app --host 0.0.0.0 --port 8000
```

The server **requires** `CIVITAI_API_TOKEN` env var — it asserts on startup and will crash without it. Health check: `GET /health`.

### Lint / Test / Build

- **Lint**: `black --check --line-length 120 --target-version py310 --skip-string-normalization openai2civitai/` and `isort --check-only openai2civitai/`. Note: the existing codebase has pre-existing formatting issues.
- **Test**: `python3 -m pytest openai2civitai/civitai_py/tests/ -v` — unit tests for the vendored CivitAI SDK. `test_query_jobs` may timeout due to real CivitAI API calls.
- **Integration test**: `python3 test_proxy.py` — requires a running server and a valid `CIVITAI_API_TOKEN` (makes real API calls).
- **Build**: `pip install -e ".[dev]"` — editable install with dev extras (pytest, requests-mock).

### Open-WebUI integration (story + image workflow)

The intended workflow: chat with a real LLM (e.g. GPT-4o-mini) to write stories, then use Open-WebUI's image generation to visualize scenes via the CivitAI proxy.

```bash
# Start proxy first (port 8000)
CIVITAI_API_TOKEN="$CIVITAI_API_TOKEN" uvicorn openai2civitai.server:app --host 0.0.0.0 --port 8000 &

# Then start Open-WebUI (port 8080), configured for image gen via proxy
ENABLE_IMAGE_GENERATION=true \
IMAGE_GENERATION_ENGINE=openai \
IMAGES_OPENAI_API_KEY=not-needed \
IMAGES_OPENAI_API_BASE_URL=http://localhost:8000/v1 \
WEBUI_AUTH=false \
open-webui serve --port 8080
```

After startup, add two connections in Admin Panel > Settings > Connections:
1. **OpenAI** (`https://api.openai.com/v1` + real API key) — for chat models
2. **CivitAI proxy** (`http://localhost:8000/v1` + `not-needed`) — for image models

Image generation is triggered via the gear icon > "Image" toggle in the chat input area. When Image mode is on, submitted prompts generate images instead of text.

### Gotchas

- `~/.local/bin` must be on `PATH` for `uvicorn`, `pytest`, `black`, etc. to be found after pip user-install.
- The `test_query_jobs.py` test frequently times out against the real CivitAI API; this is a pre-existing issue, not an environment problem.
- Image generation tests (`test_proxy.py`) make real API calls and consume CivitAI credits.
- The vendored `civitai_py` SDK uses Pydantic V1 patterns; deprecation warnings are expected with Pydantic V2.
