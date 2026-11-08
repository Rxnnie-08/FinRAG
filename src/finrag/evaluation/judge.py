"""LLM-as-judge (faithfulness + correctness) with an offline heuristic fallback.

An LLM judge is an instrument, not ground truth. Before trusting its numbers, hand-label 30-50 answers and
report agreement (`agreement`) between the judge and your labels; a weak judge model gives unreliable scores.
"""
from __future__ import annotations

import json
import re

from finrag.core.schema import Answer

JUDGE_SYSTEM = """You are a strict evaluator of answers about SEC filings. You get: a question, a gold answer (may be empty),
numbered source passages, and a candidate answer with [n] citations. Judge ONLY with the passages.
Return JSON: {"faithful": true|false, "unsupported_claims": [..], "correct": true|false|null, "reason": "..."}
faithful = every factual statement in the candidate is supported by the passage it cites (numbers must match exactly,
allowing for unit rounding stated in the passage). correct = candidate agrees with the gold answer (null if no gold)."""


class HeuristicJudge:
    """No LLM: faithfulness from the lexical/numeric attribution check."""
    name = "heuristic"

    def judge(self, ans: Answer, gold: dict | None = None) -> dict:
        return {"faithful": None if ans.support_rate is None else ans.support_rate == 1.0, "support_rate": ans.support_rate,
                "judge": self.name}


class LLMJudge:
    def __init__(self, llm):
        self.llm, self.name = llm, f"llm:{llm.name}"

    def judge(self, ans: Answer, gold: dict | None = None) -> dict:
        if ans.abstained:
            return {"faithful": True, "correct": None if not gold else (gold.get("kind") == "unanswerable"), "judge": self.name}
        src = "\n\n".join(f"[{i}] {r.chunk.label}\n{r.chunk.text[:3000]}" for i, r in enumerate(ans.retrieved, start=1))
        user = (f"Question: {ans.question}\nGold answer: {(gold or {}).get('answer', '')}\n\nPassages:\n{src}\n\nCandidate answer:\n{ans.text}")
        try:
            raw = self.llm.complete(JUDGE_SYSTEM, user, max_tokens=400)
            m = re.search(r"\{.*\}", raw, re.S)
            d = json.loads(m.group(0)) if m else {}
        except Exception as e:
            return {"faithful": None, "correct": None, "judge": self.name, "error": f"{type(e).__name__}: {e}"}
        d["judge"] = self.name
        return d


def agreement(judge_labels: list[bool | None], human_labels: list[bool]) -> dict:
    """Percent agreement and Cohen's kappa between a judge and hand labels (pairs where the judge answered)."""
    pairs = [(j, h) for j, h in zip(judge_labels, human_labels) if j is not None]
    if not pairs:
        return {"n": 0, "agreement": None, "kappa": None}
    n = len(pairs)
    agree = sum(j == h for j, h in pairs) / n
    pj, ph = sum(j for j, _ in pairs) / n, sum(h for _, h in pairs) / n
    pe = pj * ph + (1 - pj) * (1 - ph)
    kappa = 1.0 if pe == 1 else (agree - pe) / (1 - pe)
    return {"n": n, "agreement": round(agree, 3), "kappa": round(kappa, 3)}
