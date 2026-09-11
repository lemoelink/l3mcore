# Changelog

## [1.0.0] - 2026-09-11

### Changed
- Complete simplification of the codebase
- Version bump to 1.0.0

### Removed
- Plugin system (`plugin_manager.py`, `plugins/` submodule)
- Tool calling system (`tools/` submodule, tool calling loop)
- `sync_modules.py` (commercial module synchronizer)
- `.gitmodules` (no more submodules)
- Diagnostic endpoints (`/v1/route`, `/v1/discover`)
- Plugin hooks in API server (before_routing, override_route, after_generation, etc.)
- Tool-related dependencies (`pgpy`, `duckduckgo-search`, `sqlalchemy`)
- Tool-related configuration sections (`tool_calling`, `web_search`, `code_exec`, `memory_store`, `conversation_exporter`, `paperless`, `updater`)
- "tools" expert from `experts.json`
- Plugin/tool submodule initialization from `setup.sh`
- `sync_modules.py` execution and submodule sync from `start.sh`
- Paperless Search model download from `setup.sh`

### Kept
- Full OpenAI and Ollama API compatibility
- Semantic hybrid router (embedding + keyword + fuzzy)
- Multi-backend expert dispatch (Ollama, API, local ONNX/GGUF)
- Silent self-correction (auto-fallback on expert failure)
- Hot-reload of configuration files
- Sliding-window rate limiter
- Security headers and input sanitization
- CORS support
- Health and expert health endpoints
