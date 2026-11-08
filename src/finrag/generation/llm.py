"""LLM backends behind one `complete(system, user, max_tokens)` call.

  none                       no LLM: extractive answers only (works offline)
  anthropic:<model>          needs `pip install anthropic` and ANTHROPIC_API_KEY
  openai:<model>@<base_url>  any OpenAI-compatible server (Ollama: openai:llama3.1@http://localhost:11434/v1,
                             vLLM, LM Studio, ...). API key from OPENAI_API_KEY if the server needs one.
"""
from __future__ import annotations

import os


class LLM:
    name = "none"

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class AnthropicLLM(LLM):
    def __init__(self, model: str):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("pip install anthropic to use the anthropic:<model> backend") from e
        self.client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY
        self.model = model
        self.name = f"anthropic:{model}"

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        msg = self.client.messages.create(model=self.model, max_tokens=max_tokens, temperature=0, system=system,
                                          messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class OpenAICompatLLM(LLM):
    def __init__(self, model: str, base_url: str = "https://api.openai.com/v1"):
        self.model, self.base = model, base_url.rstrip("/")
        self.name = f"openai:{model}@{self.base}"

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        import requests
        headers = {"Content-Type": "application/json"}
        if key := os.environ.get("OPENAI_API_KEY"):
            headers["Authorization"] = f"Bearer {key}"
        r = requests.post(f"{self.base}/chat/completions", headers=headers, timeout=300, json={
            "model": self.model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def make_llm(spec: str | None) -> LLM | None:
    if not spec or spec == "none":
        return None
    kind, _, arg = spec.partition(":")
    if kind == "anthropic":
        return AnthropicLLM(arg or "claude-sonnet-5-5")
    if kind == "openai":
        model, _, base = arg.partition("@")
        return OpenAICompatLLM(model, base or "https://api.openai.com/v1")
    raise ValueError(f"Unknown llm '{spec}'. Use none | anthropic:<model> | openai:<model>@<base_url>")
