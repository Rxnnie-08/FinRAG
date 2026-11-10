"""Shared fixtures. The synthetic corpus lives in sample_data/ (Part A); the pipeline fixture needs parts A-D together."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_data"


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    if not (SAMPLE / "manifest.json").exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(SAMPLE / "generate.py")], check=True, env={"PYTHONPATH": str(ROOT / "src")})
    return SAMPLE


@pytest.fixture(scope="session")
def filings(sample_dir):
    from finrag.ingestion import load_manifest
    return load_manifest(sample_dir / "manifest.json", sample_dir)


@pytest.fixture(scope="session")
def chunks(filings):
    from finrag.ingestion import ingest_manifest
    return ingest_manifest(filings)


@pytest.fixture(scope="session")
def retriever(chunks):
    from finrag.retrieval import build_index
    return build_index(chunks, "lsa")


@pytest.fixture(scope="session")
def pipe(retriever, filings):
    from finrag.pipeline import RagPipeline
    return RagPipeline(retriever, filings, None, 6, today=date(2025, 6, 1))
