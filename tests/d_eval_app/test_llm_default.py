from finrag.core.config import Settings
from finrag.generation.llm import make_llm, ollama_problem, resolve_llm


class Resp:
    def __init__(self, models): self._m = models
    def json(self): return {"models": [{"name": n} for n in self._m]}


def test_ollama_is_the_default_generator():
    assert Settings().llm.startswith("ollama:")


def test_every_model_runs_through_dspy():
    m = make_llm("ollama:llama3.1:8b")
    assert type(m).__name__ == "DspyLLM" and m.name == "dspy:ollama:llama3.1:8b"
    assert type(make_llm("ollama:llama3.1:8b", raw=True)).__name__ == "OpenAICompatLLM"      # raw client only for e.g. the judge


def test_ollama_not_running_falls_back_with_a_message(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("refused")))
    llm, why = resolve_llm("ollama:llama3.1:8b")
    assert llm is None and "not reachable" in why and "extractive" in why


def test_model_not_pulled_is_reported(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(["qwen2.5:14b"]))
    assert "ollama pull llama3.1:8b" in ollama_problem("ollama:llama3.1:8b")
    llm, why = resolve_llm("ollama:llama3.1:8b")
    assert llm is None and why


def test_ollama_up_and_model_present_gives_dspy_llm(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(["llama3.1:8b"]))
    llm, why = resolve_llm("ollama:llama3.1:8b")
    assert why == "" and type(llm).__name__ == "DspyLLM"
    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(["llama3.1:latest"]))
    assert ollama_problem("ollama:llama3.1") == ""            # ":latest" counts as the untagged name


def test_saved_program_is_loaded_automatically(tmp_path):
    from finrag.optimization.dspy_qa import make_program
    prog = make_program()
    path = tmp_path / "program.json"
    prog.save(str(path))
    assert make_llm("ollama:llama3.1:8b", path).optimized
    assert not make_llm("ollama:llama3.1:8b", tmp_path / "missing.json").optimized
