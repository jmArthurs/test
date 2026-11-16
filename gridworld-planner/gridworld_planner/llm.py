"""LLM backends behind one config setting: ``backend = "api"`` (Claude API) or ``"local"`` (Ollama).

Both expose ``respond(system, messages) -> Reply``. ``messages`` is a chat history of
``{"role": "user"|"assistant", "content": ...}`` dicts; the backend returns the assistant
message to append, in whatever form that backend needs echoed back.
"""
from __future__ import annotations

import json
import tomllib
import urllib.request
from dataclasses import dataclass, field, fields
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.toml"


@dataclass
class ApiSettings:
    model: str = "claude-opus-5-5"
    effort: str = "medium"        # low | medium | high | xhigh | max
    max_tokens: int = 16000
    max_retries: int = 5
    fallbacks: bool = True        # server-side refusal fallbacks; the served model is logged


@dataclass
class LocalSettings:
    model: str = "llama3.1:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    num_predict: int = 1024       # max new tokens
    timeout: float = 300.0


@dataclass
class LLMConfig:
    backend: str = "api"          # "api" or "local" -- the one switch
    api: ApiSettings = field(default_factory=ApiSettings)
    local: LocalSettings = field(default_factory=LocalSettings)

    @property
    def model(self) -> str:
        return self.api.model if self.backend == "api" else self.local.model


def _fill(cls, data: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def load_config(path: str | Path | None = None, backend: str | None = None) -> LLMConfig:
    """Read the ``[llm]`` table of a TOML file (missing file -> defaults). ``backend`` overrides."""
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = tomllib.loads(path.read_text()).get("llm", {}) if path.exists() else {}
    cfg = LLMConfig(
        backend=raw.get("backend", "api"),
        api=_fill(ApiSettings, raw.get("api", {})),
        local=_fill(LocalSettings, raw.get("local", {})),
    )
    if backend:
        cfg.backend = backend
    if cfg.backend not in ("api", "local"):
        raise ValueError(f"backend must be 'api' or 'local', got {cfg.backend!r}")
    return cfg


@dataclass
class Reply:
    text: str                     # visible text (empty on refusal)
    message: dict                 # assistant message to append to the history
    model: str                    # model that actually served the reply


class ApiBackend:
    """Claude via the Anthropic SDK. Credentials come from the environment (ANTHROPIC_API_KEY etc.)."""

    def __init__(self, settings: ApiSettings, client=None):
        self.s = settings
        if client is None:
            import anthropic  # optional dependency: pip install -e ".[api]"
            client = anthropic.Anthropic(max_retries=settings.max_retries)
        self.client = client

    def respond(self, system: str, messages: list[dict], seed: int | None = None) -> Reply:
        # seed is unused: the API exposes no sampling seed, so API episodes are not bit-reproducible.
        kwargs = dict(
            model=self.s.model,
            max_tokens=self.s.max_tokens,
            system=system,
            messages=messages,
            output_config={"effort": self.s.effort},
            cache_control={"type": "ephemeral"},  # history is append-only, so the prefix caches
        )
        if self.s.fallbacks:
            resp = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            resp = self.client.messages.create(**kwargs)
        text = "" if resp.stop_reason == "refusal" else "".join(
            b.text for b in resp.content if b.type == "text")
        # Echo the full content (thinking blocks included) unchanged on the next turn.
        return Reply(text, {"role": "assistant", "content": resp.content}, resp.model)


class LocalBackend:
    """A local model served by Ollama (``ollama serve``), via its /api/chat endpoint. Stdlib only."""

    def __init__(self, settings: LocalSettings):
        self.s = settings

    def respond(self, system: str, messages: list[dict], seed: int | None = None) -> Reply:
        options = {"temperature": self.s.temperature, "num_predict": self.s.num_predict}
        if seed is not None:
            options["seed"] = seed
        body = {"model": self.s.model, "stream": False, "options": options,
                "messages": [{"role": "system", "content": system}, *messages]}
        req = urllib.request.Request(
            self.s.base_url.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.s.timeout) as r:
            data = json.loads(r.read())
        text = data.get("message", {}).get("content", "")
        return Reply(text, {"role": "assistant", "content": text}, data.get("model", self.s.model))


def make_backend(cfg: LLMConfig):
    return ApiBackend(cfg.api) if cfg.backend == "api" else LocalBackend(cfg.local)
