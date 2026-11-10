import pytest

from finrag.ingestion.detect import detect_meta

IXBRL = """<html><body><ix:header><ix:hidden>
<ix:nonNumeric name="dei:DocumentType" contextRef="c1">10-K</ix:nonNumeric>
<ix:nonNumeric name="dei:EntityRegistrantName" contextRef="c1">Apple Inc.</ix:nonNumeric>
<ix:nonNumeric name="dei:TradingSymbol" contextRef="c1">AAPL</ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentPeriodEndDate" contextRef="c1">September 28, 2024</ix:nonNumeric>
<ix:nonNumeric name="dei:CurrentFiscalYearEndDate" contextRef="c1">--09-28</ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentFiscalYearFocus" contextRef="c1">2024</ix:nonNumeric>
<ix:nonNumeric name="dei:DocumentFiscalPeriodFocus" contextRef="c1">FY</ix:nonNumeric>
</ix:hidden></ix:header><p>FORM 10-K</p></body></html>"""

PLAIN = """<html><body><p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p><p>FORM 10-Q</p>
<p>For the quarterly period ended December 28, 2024</p><p>Acme Widgets Corp.</p>
<p>(Exact name of registrant as specified in its charter)</p></body></html>"""


def test_detect_from_ixbrl_cover_tags(tmp_path):
    f = tmp_path / "x.htm"; f.write_text(IXBRL)
    m, notes = detect_meta(f)
    assert (m.company, m.ticker, m.filing_type, m.period_end, m.fiscal_year_end, m.fiscal_year, m.fiscal_period) == \
        ("Apple Inc.", "AAPL", "10-K", "2024-09-28", "0928", 2024, "FY") and notes == []


def test_detect_from_cover_text_and_filename(tmp_path):
    f = tmp_path / "ACME_10-Q.htm"; f.write_text(PLAIN)
    m, notes = detect_meta(f)
    assert (m.company, m.ticker, m.filing_type, m.period_end) == ("Acme Widgets Corp.", "ACME", "10-Q", "2024-12-28")
    assert any("Fiscal year end" in n for n in notes)          # 10-Q with unknown FYE is flagged, not hidden


def test_overrides_win_and_missing_info_is_an_error(tmp_path):
    f = tmp_path / "doc.htm"; f.write_text("<html><body>nothing useful</body></html>")
    with pytest.raises(ValueError):
        detect_meta(f)
    m, _ = detect_meta(f, {"company": "Foo Inc", "ticker": "foo", "filing_type": "10-K", "period_end": "2023-12-31"})
    assert (m.ticker, m.fiscal_year_end) == ("FOO", "1231")


def test_synthetic_filing_detects_correctly(sample_dir):
    m, _ = detect_meta(sample_dir / "filings" / "NWRX_10-K_2024-12-31.htm")
    assert m.ticker == "NWRX" and m.filing_type == "10-K" and m.period_end == "2024-12-31"
