import json

import pytest

pytest.importorskip("dspy")
from dspy.utils import DummyLM

from finrag.core.schema import Chunk, RetrievedChunk
from finrag.evaluation import load_gold
from finrag.generation import make_llm
from finrag.generation.generate import generate_answer
from finrag.optimization.dspy_qa import DspyLLM, make_examples, make_metric, make_program, optimize, split


def lm(answer="Total assets were $4,680 million [1].", n=400):
    return DummyLM([{"reasoning": "read the table", "answer": answer}] * n)


def _rc(text):
    c = Chunk("c1", text, "Northwind Robotics Inc.", "NWRX", "10-K", 2024, "FY", "2024-12-31", item="Item 8")
    return [RetrievedChunk(c, 1.0, 1)]


def test_program_starts_from_the_handwritten_prompt():
    from finrag.generation.generate import SYSTEM
    prog = make_program()
    assert SYSTEM.splitlines()[0] in prog.predict.predict.signature.instructions


def test_dspy_llm_plugs_into_generation():
    llm = DspyLLM("dummy", lm=lm())
    text, name = generate_answer("What were total assets?", _rc("Total assets were $4,680 million."), llm)
    assert text == "Total assets were $4,680 million [1]." and name == "dspy:dummy"


def test_metric_rewards_correct_cited_answers_only():
    import dspy
    ctx = {"q": _rc("Total assets were $4,680 million.")}
    metric = make_metric(ctx)
    ex = dspy.Example(question="q", sources="", gold={"answer": "$4,680 million", "evidence": ["4,680"]}).with_inputs("question", "sources")
    good = metric(ex, dspy.Prediction(answer="Total assets were $4,680 million [1]."))
    uncited = metric(ex, dspy.Prediction(answer="Total assets were $4,680 million."))
    wrong = metric(ex, dspy.Prediction(answer="Total assets were $9,999 million [1]."))
    assert good > uncited > wrong == 0.0
    assert metric(ex, dspy.Prediction(answer="Total assets were $4,680 million [1]."), trace=[]) is True


def test_make_llm_dspy_spec():
    m = make_llm("dspy:ollama:llama3.1:8b")
    assert m.name == "dspy:ollama:llama3.1:8b" and not m.optimized


def test_optimize_end_to_end_with_a_stub_model(pipe, sample_dir, tmp_path):
    # Mechanics only: a stub model cannot show that optimisation helps. That needs a real LM (see README).
    gold = load_gold(sample_dir / "qa_gold.jsonl")
    exs, _ = make_examples(pipe, gold)
    assert len(exs) >= 30
    train, dev = split(exs, 0.4)
    assert not {e.question for e in train} & {e.question for e in dev}
    rep = optimize(pipe, gold, lm("I could not find this in the retrieved filings."), "bootstrap", tmp_path / "p.json")
    assert rep["n_dev"] + rep["n_train"] == len(exs) and (tmp_path / "p.json").exists() and (tmp_path / "p.report.json").exists()
    assert json.loads((tmp_path / "p.report.json").read_text())["baseline"]["mean"] == 0.0
    assert DspyLLM("dummy", str(tmp_path / "p.json"), lm=lm()).optimized


def test_unreachable_model_is_an_error_not_a_zero_report(pipe, sample_dir, tmp_path):
    class Dead:
        def __call__(self, *a, **k):
            raise ConnectionError("connection refused")
    with pytest.raises(RuntimeError, match="Could not get a reply"):
        optimize(pipe, load_gold(sample_dir / "qa_gold.jsonl"), Dead(), "bootstrap", tmp_path / "p.json")
