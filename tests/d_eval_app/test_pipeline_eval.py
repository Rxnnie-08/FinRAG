import json

import pytest

from finrag.evaluation.financebench import convert, parse_doc_name
from finrag.evaluation.judge import HeuristicJudge, agreement
from finrag.evaluation.metrics import answer_correct, load_gold, retrieval_metrics
from finrag.evaluation.runner import bootstrap_ci, paired_diff, run_benchmark, to_markdown


def test_end_to_end_answers(pipe):
    a = pipe.ask("What were Northwind Robotics' total assets at the end of fiscal 2024?")
    assert "4,680" in a.text and not a.abstained and a.support_rate == 1.0


def test_citations_point_to_real_chunks(pipe):
    a = pipe.ask("What were Northwind Robotics' total assets at the end of fiscal 2024?")
    ids = {r.chunk.chunk_id for r in a.retrieved}
    assert a.citations and all(c.chunk_id in ids for c in a.citations)


def test_fiscal_quarter_end_to_end(pipe):
    a = pipe.ask("What was Harbor Lane Foods' revenue in Q2 of fiscal 2025?")
    assert a.rewrite.filters["fiscal_period"] == ["Q2"] and a.retrieved[0].chunk.ticker == "HLFC"


def test_unknown_company_abstains(pipe):
    a = pipe.ask("What was Apple's total assets in fiscal 2024?")
    assert a.abstained or a.text.startswith("I could not")


def test_bad_mode_raises(pipe):
    with pytest.raises(ValueError):
        pipe.ask("x", mode="nope")


def test_answer_correct_numeric_and_text():
    assert answer_correct("Total assets were $4,680 million", {"answer": "$4,680 million", "evidence": ["4,680"]})["correct"]
    assert not answer_correct("It was $1 million", {"answer": "$4,680 million", "evidence": ["4,680"]})["correct"]


def test_gold_loads_and_has_unanswerables(sample_dir):
    gold = load_gold(sample_dir / "qa_gold.jsonl")
    assert len(gold) == 38 and sum(1 for g in gold if g.get("unanswerable") or g.get("kind") == "unanswerable") >= 4


def test_bootstrap_and_paired():
    m, lo, hi = bootstrap_ci([1, 1, 0, 1, 0, 1, 1, 1])
    assert lo <= m <= hi
    d = paired_diff([1] * 20, [0] * 20)
    assert d["diff"] > 0.9


def test_benchmark_runs_and_rewrite_helps(pipe, sample_dir):
    gold = load_gold(sample_dir / "qa_gold.jsonl")
    res = run_benchmark(pipe, gold, k=5, judge=HeuristicJudge())
    assert "hybrid+rewrite" in to_markdown(res)
    sm = res["summary"]
    assert sm["hybrid+rewrite"]["answer_correct"]["mean"] > sm["hybrid"]["answer_correct"]["mean"]
    assert sm["hybrid+rewrite"]["abstain_ok"]["mean"] == 1.0


def test_judge_agreement_kappa():
    out = agreement([True, False, True, True], [True, False, False, True])
    assert 0 <= out["agreement"] <= 1


def test_financebench_conversion():
    m = parse_doc_name("APPLE_2022_10K")
    assert (m.ticker, m.filing_type, m.fiscal_year) == ("APPLE", "10-K", 2022)
    rows = [{"financebench_id": "fb1", "doc_name": "APPLE_2022_10K", "question": "q", "answer": "a", "question_type": "metrics-generated",
             "evidence": [{"evidence_page_num": 41}], "justification": ""}]
    filings, gold = convert(rows, "/pdfs")
    assert gold[0]["evidence_pages"] == [42] and filings[0].path.endswith("APPLE_2022_10K.pdf")
