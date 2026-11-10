"""LLM backends behind one `complete(system, user, max_tokens)` call.

  none                       no LLM: extractive answers only (works offline)
  ollama:<model>             local Ollama server (the default: ollama:llama3.1:8b)
  anthropic:<model>          needs `pip install anthropic` and ANTHROPIC_API_KEY
  openai:<model>@<base_url>  any OpenAI-compatible server (Ollama: openai:llama3.1@http://localhost:11434/v1,
                             vLLM, LM Studio, ...). API key from OPENAI_API_KEY if the server needs one.
"""
from __future__ import annotations

import os
from pathlib import Path


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


def ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _raw_backend(kind: str, arg: str, spec: str) -> LLM:
    if kind == "anthropic":
        return AnthropicLLM(arg or "claude-sonnet-5-5")
    if kind == "ollama":
        llm = OpenAICompatLLM(arg or "llama3.1:8b", ollama_host() + "/v1")
        llm.name = f"ollama:{llm.model}"
        return llm
    if kind == "openai":
        model, _, base = arg.partition("@")
        return OpenAICompatLLM(model, base or "https://api.openai.com/v1")
    raise ValueError(f"Unknown llm '{spec}'. Use none | ollama:<model> | anthropic:<model> | openai:<model>@<base_url>")


def make_llm(spec: str | None, program: str | Path | None = None, raw: bool = False) -> LLM | None:
    """The generator behind the app. Every model is driven through a DSPy program (optimisable with `finrag optimize`);
    `program` is the saved optimised program, loaded if the file exists. raw=True returns the plain chat client instead
    (used where DSPy adds nothing, e.g. the LLM-as-judge)."""
    if not spec or spec == "none":
        return None
    kind, _, arg = spec.partition(":")
    if kind == "dspy":                       # "dspy:ollama:x" means the same as "ollama:x"
        spec = arg
        kind, _, arg = spec.partition(":")
    backend = _raw_backend(kind, arg, spec)
    if raw:
        return backend
    from finrag.optimization.dspy_qa import DspyLLM
    return DspyLLM(spec, str(program) if program else None)


def ollama_problem(spec: str) -> str:
    """"" if the Ollama server is up and the model is pulled, else a one-line explanation."""
    import requests
    kind, _, arg = (spec[5:] if spec.startswith("dspy:") else spec).partition(":")
    if kind != "ollama":
        return ""
    model = arg or "llama3.1:8b"
    host = ollama_host()
    try:
        names = [m.get("name", "") for m in requests.get(f"{host}/api/tags", timeout=3).json().get("models", [])]
    except Exception:
        return f"Ollama is not reachable at {host}. Start Ollama and restart finrag; until then answers are extractive lookups."
    have = {n for n in names} | {n.removesuffix(":latest") for n in names}
    if model not in have:
        return f"Ollama has no model '{model}'. Run `ollama pull {model}` (or set FINRAG_LLM to a model from `ollama list`); until then answers are extractive lookups."
    return ""


def resolve_llm(spec: str | None, program: str | Path | None = None) -> tuple[LLM | None, str]:
    """make_llm that never takes the app down: returns (llm, "") or (None, reason) so the caller can fall back to extractive answers."""
    if not spec or spec == "none":
        return None, ""
    if (problem := ollama_problem(spec)):
        return None, problem
    try:
        return make_llm(spec, program), ""
    except Exception as e:
        return None, f"Could not start the language model '{spec}' ({type(e).__name__}: {e}); answers are extractive lookups."
