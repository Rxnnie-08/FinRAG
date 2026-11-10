from urllib.parse import quote

import pytest

from finrag.core.config import Settings
from finrag.server.app import create_app


@pytest.fixture()
def client(pipe, tmp_path):
    app = create_app(pipe, Settings(home=tmp_path))
    return app.test_client()


def test_status_regression(client):          # was a 500: FilingMeta has no .get
    r = client.get("/api/status")
    assert r.status_code == 200
    j = r.get_json()
    assert j["synthetic"] is True and j["filings"]


def test_ask_ok(client):
    r = client.post("/api/ask", json={"question": "What were Northwind Robotics' total assets at the end of fiscal 2024?"})
    assert r.status_code == 200 and "4,680" in r.get_json()["answer"]


@pytest.mark.parametrize("body", [{}, {"question": "  "}, {"question": "x" * 1001}, {"question": "ok", "mode": "bad"}, {"question": "ok", "k": 99}])
def test_ask_validation(client, body):
    assert client.post("/api/ask", json=body).status_code == 400


def test_compare_and_chunk(client, pipe):
    r = client.post("/api/compare", json={"question": "total assets Northwind fiscal 2024"})
    assert r.status_code == 200
    cid = pipe.retriever.chunks[0].chunk_id
    assert client.get("/api/chunk/" + quote(cid, safe="")).status_code == 200
    assert client.get("/api/chunk/nope").status_code == 404


def test_static_and_no_innerhtml(client):
    assert client.get("/").status_code == 200
    js = client.get("/static/app.js").get_data(as_text=True)
    import re
    code = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    assert "innerHTML" not in code
    assert client.get("/static/../app.py").status_code in (404, 400)


def test_benchmark_endpoint_writes_results(client, tmp_path, sample_dir):
    # regression: on Windows the default file encoding is cp1252, which cannot write "−" in results.md (HTTP 500)
    import shutil
    shutil.copy(sample_dir / "qa_gold.jsonl", tmp_path / "qa_gold.jsonl")
    r = client.post("/api/eval/run")
    assert r.status_code == 200 and (tmp_path / "eval" / "results.md").exists()


def test_all_text_io_declares_utf8():
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    bad = []
    for f in list((root / "src").rglob("*.py")) + [root / "sample_data" / "generate.py"]:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"\.(read_text|write_text)\(", line) and "encoding" not in line and "json.dumps({" not in line:
                bad.append(f"{f.name}:{i}")
            if re.search(r"\bopen\(", line) and "encoding" not in line and not re.search(r'"[rw]b"', line):
                bad.append(f"{f.name}:{i}")
    assert not bad, bad


@pytest.fixture()
def fresh(filings, chunks, tmp_path):
    from finrag.pipeline import RagPipeline
    from finrag.retrieval import build_index
    p = RagPipeline(build_index(chunks, "lsa"), filings, None, 6)
    p.settings = Settings(home=tmp_path)
    return p, create_app(p, p.settings).test_client()


def _upload(client, path, **form):
    import io
    return client.post("/api/upload", data={"file": (io.BytesIO(path.read_bytes()), path.name), **form}, content_type="multipart/form-data")


def test_upload_new_filing_then_ask(fresh, sample_dir, tmp_path):
    pipe, client = fresh
    html = (sample_dir / "filings" / "NWRX_10-K_2023-12-31.htm").read_text()
    src = tmp_path / "ACME_10-K_2022-12-31.htm"
    src.write_text(html.replace("Northwind Robotics Inc.", "Acme Widgets Inc.").replace("NWRX", "ACME")
                   .replace("December 31, 2023", "December 31, 2022"))
    r = _upload(client, src)
    assert r.status_code == 200, r.get_json()
    j = r.get_json()
    assert j["meta"]["ticker"] == "ACME" and j["chunks"] > 5
    assert pipe.info()["n_filings"] == 8
    st = client.get("/api/status").get_json()
    assert st["synthetic"] is False and not st["has_gold"] and "Acme" in st["examples"][0]
    a = client.post("/api/ask", json={"question": "What were Acme Widgets' total assets?"}).get_json()
    assert a["rewrite"]["filters"]["ticker"] == ["ACME"]
    assert _upload(client, src).status_code == 409                       # same filing twice is refused
    assert (tmp_path / "index" / "filings.json").exists()                # persisted


def test_upload_rejects_bad_files(fresh, tmp_path):
    _, client = fresh
    bad = tmp_path / "notes.txt"; bad.write_text("hi")
    assert _upload(client, bad).status_code == 400
    junk = tmp_path / "junk.htm"; junk.write_text("<html>nothing</html>")
    r = _upload(client, junk)
    assert r.status_code == 422 and "form type" in r.get_json()["error"].lower()
    assert not list((tmp_path / "filings" / "uploads").glob("*junk*"))   # failed uploads are not kept


def test_pipeline_build_upload_reload_with_qdrant_on_disk(filings, sample_dir, tmp_path):
    from finrag.ingestion import detect_meta
    from finrag.pipeline import RagPipeline
    s = Settings(home=tmp_path, llm="none")
    p = RagPipeline.build(filings, s)
    assert p.info()["vector_store"] == "qdrant" and (tmp_path / "index" / "qdrant").exists()
    assert "4,680" in p.ask("What were Northwind Robotics' total assets at the end of fiscal 2024?").text
    meta, _ = detect_meta(sample_dir / "filings" / "NWRX_8-K_2025-04-24.htm", {"ticker": "NWX8", "company": "Northwind Two"})
    meta.accession = "UPLOAD-x"
    before = p.info()["n_chunks"]
    added = p.add_filing(meta)
    assert p.info()["n_chunks"] == before + added["chunks"]
    assert not (tmp_path / "index" / "qdrant_next").exists()           # build-then-swap leaves no scratch folder
    assert "4,680" in p.ask("What were Northwind Robotics' total assets at the end of fiscal 2024?").text   # still answers after the swap
    p.retriever.dense.close()
    p2 = RagPipeline.load(s)
    assert p2.info()["n_chunks"] == before + added["chunks"]
    p2.retriever.dense.close()
