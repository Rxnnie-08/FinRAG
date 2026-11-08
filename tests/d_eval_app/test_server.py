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
