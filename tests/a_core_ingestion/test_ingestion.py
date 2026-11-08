import pytest
from bs4 import BeautifulSoup

from finrag.ingestion.edgar import EdgarClient
from finrag.ingestion.tables import parse_table
from finrag.ingestion.xbrl import extract_facts


def _table(html):
    return BeautifulSoup(html, "lxml").find("table")


def test_table_colspan_header_and_symbol_merge():
    t = parse_table(_table("""<table>
      <tr><td></td><td colspan="2">December 31,</td></tr>
      <tr><td></td><td>2024</td><td>2023</td></tr>
      <tr><td>Cash</td><td>$</td><td>583</td><td>$</td><td>500</td></tr>
      <tr><td>Loss</td><td>(</td><td>34</td><td>)</td><td>5</td></tr></table>"""))
    assert any("2024" in h for h in t.header)
    flat = [[c for c in r.cells] for r in t.rows]
    assert any("583" in " ".join(r) and "$" in " ".join(r) for r in flat)


def test_chunks_cover_all_filings(filings, chunks):
    assert {c.doc_id for c in chunks} == {f.doc_id for f in filings}
    assert all(c.text.strip() for c in chunks)


def test_table_chunks_keep_header_and_units(chunks):
    tables = [c for c in chunks if c.element_type in ("table", "table_rows")]
    assert tables
    for c in tables:
        assert c.table_data and c.table_data["header"], c.chunk_id
    assert any(c.units for c in tables)


def test_split_tables_repeat_header(chunks):
    from collections import defaultdict
    by_table = defaultdict(list)
    for c in chunks:
        if c.element_type == "table_rows":
            by_table[(c.doc_id, c.table_id)].append(c)
    for parts in by_table.values():
        heads = {tuple(p.table_data["header"]) for p in parts}
        assert len(heads) == 1


def test_footnotes_attached_somewhere(chunks):
    assert any(c.footnotes for c in chunks)


def test_xbrl_facts_extracted(sample_dir, chunks):
    html = (sample_dir / "filings" / "NWRX_10-K_2024-12-31.htm").read_text()
    assets = {f.instant: f.value for f in extract_facts(html) if f.concept == "Assets"}
    assert assets["2024-12-31"] == 4_680_000_000 and assets["2023-12-31"] == 4_333_000_000
    assert any(c.element_type == "xbrl_facts" and "$4,680 million" in c.text for c in chunks)


def test_edgar_requires_contact_user_agent():
    with pytest.raises(ValueError):
        EdgarClient("no-contact-info")


def test_edgar_resolve_with_mock_session():
    class R:
        def __init__(s, d): s.d = d
        def json(s): return s.d
        def raise_for_status(s): pass

    class S:
        headers = {}
        def get(s, url, timeout=0):
            assert "company_tickers" in url
            return R({"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}})

    c = EdgarClient("Jane jane@uni.edu", session=S(), min_interval=0)
    assert c.resolve("aapl") == ("0000320193", "Apple Inc.")
    with pytest.raises(KeyError):
        c.resolve("ZZZZ")
