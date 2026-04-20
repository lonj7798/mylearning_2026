---
chapter: ch-03
course: boson-agent
phase: read
excerpt_for: packages/basement/basement/config/loader.py + basement/__main__.py
created_at: "2026-04-17"
---

# Source: `basement/config/loader.py` + `basement/__main__.py` — API Key Resolution and .env Discovery

**One-line description:** The two-stage key resolution pipeline: `__main__.py` walks up the filesystem to load a `.env` file before any imports; `loader.py` then checks provider-specific environment variable names after config parsing.

---

## Code Excerpt A — .env Walk-up (`__main__.py` lines 17–28)

```python
# packages/basement/basement/__main__.py, lines 17-28

# Auto-load .env from agent dir or parent directories
try:
    from dotenv import load_dotenv
    agent_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    # Search for .env: agent dir → parent → grandparent → ...
    for parent in [agent_dir.resolve()] + list(agent_dir.resolve().parents):
        env_file = parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            break
except ImportError:
    pass  # dotenv not installed, rely on env vars
```

---

## Code Excerpt B — Environment Variable Resolution (`loader.py` lines 95–110)

```python
# packages/basement/basement/config/loader.py, lines 95-110

def resolve_api_key(config: LLMConfig) -> str | None:
    """Resolve API key from environment variables.

    Checks provider-specific env vars:
    - anthropic: ANTHROPIC_API_KEY
    - openai: OPENAI_API_KEY
    """
    env_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }
    env_var = env_map.get(config.provider)
    if env_var:
        return os.environ.get(env_var)
    return None
```

---

## Code Excerpt C — Where Resolution Is Called (`loader.py` lines 70–74)

```python
# packages/basement/basement/config/loader.py, lines 70-74

    # Resolve API key from env var if not in config
    if config.llm.api_key is None:
        config.llm.api_key = resolve_api_key(config.llm)
```

---

## Explanation

API key resolution follows a two-stage pipeline:

**Stage 1 — .env file discovery (at process startup, before `import` of any framework module).** The `try/except ImportError` block at the top of `__main__.py` runs before any framework import. It uses `python-dotenv`'s `load_dotenv()` to parse a `.env` file and inject its contents into `os.environ`. The walk-up logic starts at the agent directory, then steps to each parent via `agent_dir.resolve().parents` (which yields `/path/to/agents/demo`, `/path/to/agents`, `/path/to`, `/`, etc.) until it finds a `.env` or exhausts the tree. First match wins; it calls `break`. If `dotenv` isn't installed, the whole block is silently skipped.

**Stage 2 — environment variable lookup (during config loading).** After `yaml.safe_load` and Pydantic validation, `load_config` checks whether `config.llm.api_key is None`. If so, it calls `resolve_api_key(config.llm)`, which maps the `provider` string to the canonical environment variable name (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`) and calls `os.environ.get(env_var)`. By this point, Stage 1 may have already populated `os.environ` from the `.env` file, so Stage 2 finds the key even if it wasn't set in the shell environment directly.

The priority order is therefore:

1. `api_key` explicitly set in `config.yaml` — used as-is, Stage 2 never runs
2. `.env` file found by walk-up → key set in `os.environ` → Stage 2 reads it
3. Key already in shell environment → Stage 2 reads it
4. None of the above → `api_key` remains `None` → provider constructor receives `None` → SDK raises its own auth error

**Notice:** `resolve_api_key` returns `str | None`, not `str`. If the environment variable isn't set, `os.environ.get(env_var)` returns `None`, and that `None` is assigned back to `config.llm.api_key`. No exception is raised here — the error surfaces later when the SDK's constructor or first API call rejects the missing credential. This is a deliberate choice: fail at use-time (informative SDK error), not at config-load-time (the agent might not need the LLM for every code path during testing).

**Connection to universal pattern:** Key resolution is pre-Step 1 — it populates the `api_key` field of `LLMConfig` before `get_provider` is called. The registry's factory passes the whole `config` to the provider constructor, which passes `config.api_key` to the SDK client. The two-stage resolution is transparent to the rest of the system.
