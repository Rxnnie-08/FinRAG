"""DSPy prompt optimisation for the grounded-answer step.

The hand-written prompt in generation/generate.py is the BASELINE. Here the same instructions become a DSPy
Signature, and an optimiser (BootstrapFewShot or MIPROv2) searches for better instructions / few-shot examples
against the gold questions, scored by the same checks the benchmark uses: is the answer correct, and does every
claim cite a source that supports it.

Every LLM answer in the app goes through this program (generation.llm.make_llm wraps the model in `DspyLLM`).
Workflow:   finrag optimize --gold qa_gold.jsonl            (uses FINRAG_LLM, default ollama:llama3.1:8b)
            -> saves <home>/dspy/program.json, which `finrag serve` / `finrag ask` load automatically.

Honest scope: only the answer-generation prompt is optimised. Retrieval, rewriting and the claim checker are not
learned. The gold set is split into train and dev; report the dev numbers, never the train ones.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

from finrag.core.schema import RetrievedChunk
from finrag.evaluation.metrics import answer_correct
from finrag.generation.attribution import verify_answer
from finrag.generation.generate import SYSTEM, format_sources
from finrag.generation.llm import LLM


def _dspy():
    try:
        import dspy
    except ImportError as e:
        raise RuntimeError("pip install dspy to use prompt optimisation") from e
    return dspy


def make_lm(spec: str):
    """none of the keys are stored: they come from the environment (ANTHROPIC_API_KEY, OPENAI_API_KEY)."""
    dspy = _dspy()
    kind, _, arg = spec.partition(":")
    if kind == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        return dspy.LM(f"openai/{arg or 'llama3.1:8b'}", api_base=host + "/v1", api_key="ollama", temperature=0.0, max_tokens=700)
    if kind == "anthropic":
        return dspy.LM(f"anthropic/{arg}", temperature=0.0, max_tokens=700)
    if kind == "openai":
        model, _, base = arg.partition("@")
        return dspy.LM(f"openai/{model}", api_base=base or None, api_key=os.environ.get("OPENAI_API_KEY", "none"), temperature=0.0, max_tokens=700)
    raise ValueError(f"Unknown LM '{spec}'. Use ollama:<model> | anthropic:<model> | openai:<model>@<base_url>")


def make_program(cot: bool = True):
    """The DSPy program: question + numbered sources -> cited answer. Instructions start as the hand-written SYSTEM prompt."""
    dspy = _dspy()

    class GroundedAnswer(dspy.Signature):
        sources: str = dspy.InputField(desc="numbered source passages, each starting with [n]")
        question: str = dspy.InputField()
        answer: str = dspy.OutputField(desc="short answer with [n] citations after every factual sentence")

    sig = GroundedAnswer.with_instructions(SYSTEM)

    class Answerer(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predict = (dspy.ChainOfThought if cot else dspy.Predict)(sig)

        def forward(self, question: str, sources: str):
            return self.predict(question=question, sources=sources)

    return Answerer()


# --------------------------------------------------------------------------- data + metric
def make_examples(pipe, gold: list[dict], k: int = 5):
    """One dspy.Example per answerable gold question, with the sources our retriever really finds (rewriting on, hybrid).
    Returns (examples, retrieved_by_question). Unanswerable questions are skipped: the pipeline's guards answer those
    before any LLM is called, so there is nothing for the prompt to learn from them."""
    dspy = _dspy()
    exs, ctx = [], {}
    for g in gold:
        if g.get("kind") == "unanswerable":
            continue
        a = pipe.ask(g["question"], mode="hybrid", rewrite="rules", k=k, use_llm=False)
        if not a.retrieved:
            continue
        q = (a.rewrite.standalone if a.rewrite else g["question"])
        ctx[q] = a.retrieved
        exs.append(dspy.Example(question=q, sources=format_sources(a.retrieved), gold=g).with_inputs("question", "sources"))
    return exs, ctx


def make_metric(ctx: dict[str, list[RetrievedChunk]]):
    def metric(example, pred, trace=None):
        text = getattr(pred, "answer", "") or ""
        if not answer_correct(text, example.gold)["correct"]:
            return 0.0 if trace is None else False
        _, stats = verify_answer(text, ctx.get(example.question, []))
        score = 0.7 + 0.15 * (stats.get("citation_coverage") or 0.0) + 0.15 * (stats.get("support_rate") or 0.0)
        score -= 0.1 * min(1, stats.get("invalid_citations", 0))
        return score if trace is None else score >= 0.99      # bootstrapped demos must be fully cited and supported
    return metric


def split(examples: list, dev_frac: float = 0.4, seed: int = 0) -> tuple[list, list]:
    idx = list(range(len(examples)))
    random.Random(seed).shuffle(idx)
    n_dev = max(1, round(len(idx) * dev_frac))
    dev = [examples[i] for i in idx[:n_dev]]
    train = [examples[i] for i in idx[n_dev:]]
    return train, dev


def score(program, dev: list, metric, lm) -> list[float]:
    dspy = _dspy()
    out = []
    with dspy.context(lm=lm):
        for ex in dev:
            try:
                out.append(float(metric(ex, program(question=ex.question, sources=ex.sources))))
            except Exception:                  # a failed call counts as a miss, not as a crash of the whole run
                out.append(0.0)
    return out


def optimize(pipe, gold: list[dict], lm, optimizer: str = "bootstrap", out: str | Path = "dspy/program.json",
             dev_frac: float = 0.4, seed: int = 0, max_demos: int = 4, k: int = 5, cot: bool = True) -> dict:
    """Optimise on the train split, report baseline vs optimised on the held-out dev split, save the compiled program."""
    from finrag.evaluation.runner import bootstrap_ci, paired_diff
    dspy = _dspy()
    try:      # fail loudly: otherwise every call errors, every score is 0, and the report would say "0.0 vs 0.0"
        lm("Reply with the single word OK.")
    except Exception as e:
        raise RuntimeError(f"Could not get a reply from the language model ({type(e).__name__}: {e}). "
                           "Is Ollama running and the model pulled, or the API key set?") from e
    examples, ctx = make_examples(pipe, gold, k)
    if len(examples) < 6:
        raise ValueError(f"Only {len(examples)} usable gold questions; DSPy needs a few dozen to optimise anything meaningful.")
    train, dev = split(examples, dev_frac, seed)
    metric = make_metric(ctx)
    base = make_program(cot)
    base_scores = score(base, dev, metric, lm)
    with dspy.context(lm=lm):
        if optimizer == "bootstrap":
            tele = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=max_demos, max_labeled_demos=max_demos)
            compiled = tele.compile(make_program(cot), trainset=train)
        elif optimizer == "mipro":
            tele = dspy.MIPROv2(metric=metric, prompt_model=lm, task_model=lm, auto="light", seed=seed)
            compiled = tele.compile(make_program(cot), trainset=train, valset=train, requires_permission_to_run=False)
        else:
            raise ValueError("optimizer must be bootstrap | mipro")
    opt_scores = score(compiled, dev, metric, lm)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(str(out))
    m0, l0, h0 = bootstrap_ci(base_scores)
    m1, l1, h1 = bootstrap_ci(opt_scores)
    report = {"optimizer": optimizer, "n_train": len(train), "n_dev": len(dev), "seed": seed,
              "baseline": {"mean": m0, "ci": [l0, h0]}, "optimized": {"mean": m1, "ci": [l1, h1]},
              "difference": paired_diff(opt_scores, base_scores), "program": str(out),
              "dev_questions": [e.gold.get("id") for e in dev]}
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


# --------------------------------------------------------------------------- serving
class DspyLLM(LLM):
    """Serves a (compiled) DSPy program through the same interface as the other LLM backends."""

    def __init__(self, lm_spec: str, program_path: str | None = None, lm=None):
        self.lm = lm or make_lm(lm_spec)
        self.program = make_program()
        self.optimized = False
        if program_path and Path(program_path).exists():
            self.program.load(program_path)
            self.optimized = True
        self.name = f"dspy:{lm_spec}" + ("+optimized" if self.optimized else "")

    def answer(self, question: str, sources: str, notes: str = "") -> str:
        dspy = _dspy()
        with dspy.context(lm=self.lm):
            return (self.program(question=(notes + question).strip(), sources=sources).answer or "").strip()

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:     # generic fallback, e.g. for the LLM judge
        dspy = _dspy()
        with dspy.context(lm=self.lm):
            return (dspy.Predict("prompt -> response")(prompt=f"{system}\n\n{user}").response or "").strip()
