# Part D: evaluation, pipeline, CLI, server, UI

Owns: `pipeline.py` (`RagPipeline.ask/compare`), `cli.py`, `evaluation/` (gold format, metrics, judges, bootstrap, FinanceBench
loader), `server/` (Flask API + single-page UI).

Test: `pytest tests/d_eval_app` (needs A, B, C merged). Run: `finrag demo && finrag serve`.
Open items: run FinanceBench end to end with `pip install -e ".[pdf]"`; hand-label ~30 answers to measure judge agreement
(`evaluation.judge.agreement`).
