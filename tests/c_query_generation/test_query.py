from datetime import date

from finrag.query import CompanyDirectory, QueryRewriter
from finrag.query.glossary import expand_terms


def rw(filings, today=date(2025, 6, 1)):
    return QueryRewriter(CompanyDirectory.from_filings(filings), None, today, "rules")


def test_company_and_period_filters(filings):
    q = rw(filings).rewrite("What were Northwind's total assets in fiscal 2024?")
    assert q.filters["ticker"] == ["NWRX"] and q.filters["fiscal_year"] == [2024]


def test_june_fye_quarter(filings):
    q = rw(filings).rewrite("Harbor Lane revenue in Q2 of fiscal 2025")
    assert q.filters["ticker"] == ["HLFC"] and q.filters["fiscal_year"] == [2025] and q.filters["fiscal_period"] == ["Q2"]


def test_year_in_maturity_is_not_a_filter(filings):
    q = rw(filings).rewrite("How much debt matures in 2027 at Northwind?")
    assert 2027 not in (q.filters.get("fiscal_year") or [])


def test_followup_carries_company(filings):
    q = rw(filings).rewrite("And what about net income?", ["What was Northwind's revenue in fiscal 2024?"])
    assert q.filters.get("ticker") == ["NWRX"]


def test_unknown_entity_detected(filings):
    q = rw(filings).rewrite("What was Apple's total assets in fiscal 2024?")
    assert "Apple" in q.unknown_entities


def test_abbreviation_expansion():
    spelled, synonyms, _ = expand_terms("capex")
    assert "capital expenditures" in spelled and "capital expenditures" in synonyms


def test_rewrite_off_changes_nothing(filings):
    q = QueryRewriter(CompanyDirectory.from_filings(filings), None, None, "off").rewrite("total assets Northwind 2024")
    assert q.filters == {} and q.standalone == "total assets Northwind 2024"
