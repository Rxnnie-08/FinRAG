from datetime import date

from finrag.core.fiscal import detect_filing_type, fiscal_period_for_date, parse_period, quarter_end
from finrag.core.numbers import any_number_match, numbers_match, parse_numbers
from finrag.core.schema import Chunk


def test_parse_numbers_scales_and_negatives():
    n = parse_numbers("$1.25 billion")[0]
    assert n.abs_value == 1.25e9
    assert parse_numbers("(1,250)")[0].neg
    assert parse_numbers("loss of -34")[0].neg


def test_footnote_marker_vs_negative():
    assert parse_numbers("Notes due 2027 (1)")[-1].value == 2027      # "(1)" ignored
    assert parse_numbers("(34)")[0].value == -34                       # leading "(34)" is a negative


def test_scale_tolerant_match():
    assert any_number_match("$1.25 billion", "1,250")
    assert not any_number_match("1,251", "1,250")
    a, b = parse_numbers("1.25")[0], parse_numbers("1,250")[0]
    assert numbers_match(a, b) and not numbers_match(a, b, scale_tolerant=False)


def test_fiscal_period_june_fye():
    assert fiscal_period_for_date(date(2024, 12, 31), 6) == (2025, 2)
    assert quarter_end(6, 2025, 2) == date(2024, 12, 31)


def test_parse_period_quarter_and_bare_year():
    p = parse_period("revenue in Q2 2025", "0630")
    assert (p.fiscal_year, p.fiscal_period) == (2025, "Q2")
    assert parse_period("debt maturing in 2027", "1231").bare_year
    assert detect_filing_type("see the 10-Q") == "10-Q"


def test_chunk_index_text_contains_header():
    c = Chunk("x", "body", "Acme Inc", "ACME", "10-K", 2024, "FY", "2024-12-31", item="Item 8")
    assert c.index_text.startswith("Acme Inc | ACME | 10-K | fiscal 2024 | Item 8")
    assert Chunk.from_dict(c.to_dict()) == c
