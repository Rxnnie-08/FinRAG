"""HTML table -> structured table that keeps column headers attached to every value.

SEC tables are layout-heavy: "$" and ")" sit in their own cells, header text spans several columns
(colspan), and units ("in millions") live outside the grid. This module rebuilds the logical table:
  1. expand colspan/rowspan into a rectangular grid,
  2. find the header rows and combine multi-row headers per column,
  3. glue currency/percent/parenthesis cells back onto their numbers,
  4. drop layout-only columns,
  5. track section rows ("Current assets:") as groups.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import Tag

_WS = re.compile(r"[\s ​]+")


def clean(s: str) -> str:
    return _WS.sub(" ", s or "").strip()


_UNITS_RE = re.compile(
    r"\(?\s*(?:amounts?\s+|dollars?\s+|\$\s*|usd\s+)?(?:in|expressed in)\s+(thousands|millions|billions)"
    r"(?:\s+of\s+(?:u\.?s\.?\s+)?dollars)?(?:\s*,?\s*except[^)]{0,80})?\s*\)?",
    re.I,
)
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")
_NUMLIKE = re.compile(r"^-?\d+(?:\.\d+)?$")
_DASHES = {"—", "–", "-", "—%", "– –"}
_SYM_PRE = {"$", "€", "£"}
_SYM_POST = {")", "%", ")%", "%)", "x"}


def detect_units(*texts: str) -> str:
    for t in texts:
        m = _UNITS_RE.search(t or "")
        if m:
            return clean(m.group(0).strip("() "))
    return ""


def _is_data_cell(t: str) -> bool:
    if not t:
        return False
    if t in _DASHES:
        return True
    core = re.sub(r"[\s$€£,()%]", "", t)
    if _YEAR_RE.match(core) and not re.search(r"[$,()%]", t):
        return False  # bare year = header
    return bool(_NUMLIKE.match(core.lstrip("-−–")))


@dataclass
class TableRow:
    label: str
    cells: list[str]
    group: str = ""
    kind: str = "data"      # data | section


@dataclass
class ParsedTable:
    kind: str                         # data | layout | toc
    header: list[str] = field(default_factory=list)   # header[0] is the label column header
    rows: list[TableRow] = field(default_factory=list)
    title: str = ""
    units: str = ""
    layout_text: list[str] = field(default_factory=list)

    @property
    def n_data_rows(self) -> int:
        return sum(1 for r in self.rows if r.kind == "data")


def _grid(table: Tag) -> tuple[list[list[tuple[str, bool]]], int]:
    """Rectangular grid of (text, is_colspan_copy). Also returns the count of leading <thead>/<th> rows."""
    rows: list[list[tuple[str, bool]]] = []
    pending: dict[tuple[int, int], str] = {}
    th_rows = 0
    trs = [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]
    counting_th = True
    for r, tr in enumerate(trs):
        row: list[tuple[str, bool]] = []
        c = 0

        def fill_pending() -> None:
            nonlocal c
            while (r, c) in pending:
                row.append((pending.pop((r, c)), False))
                c += 1

        cells = [td for td in tr.find_all(["td", "th"]) if td.find_parent("tr") is tr]
        for cell in cells:
            fill_pending()
            text = clean(cell.get_text(" "))
            try:
                cs = max(1, min(int(cell.get("colspan", 1) or 1), 60))
                rs = max(1, min(int(cell.get("rowspan", 1) or 1), 60))
            except ValueError:
                cs = rs = 1
            row.append((text, False))
            row.extend([(text, True)] * (cs - 1))
            for i in range(1, rs):
                for k in range(cs):
                    pending[(r + i, c + k)] = text
            c += cs
        fill_pending()
        rows.append(row)
        is_th = bool(cells) and all(x.name == "th" for x in cells)
        if counting_th and is_th:
            th_rows += 1
        else:
            counting_th = False
    width = max((len(r) for r in rows), default=0)
    for r in rows:
        r.extend([("", True)] * (width - len(r)))
    return rows, th_rows


def _merge_symbols(cells: list[str]) -> list[str]:
    out = list(cells)
    n = len(out)
    for j in range(1, n):
        if out[j] in _SYM_PRE:
            k = j + 1
            while k < n and out[k] == "":
                k += 1
            if k < n and out[k] not in _SYM_PRE:
                out[k] = out[j] + out[k]
                out[j] = ""
    for j in range(n - 1, 0, -1):
        if out[j] in _SYM_POST:
            k = j - 1
            while k >= 1 and out[k] == "":
                k -= 1
            if k >= 1:
                out[k] += out[j]
                out[j] = ""
    return out


def parse_table(table: Tag) -> ParsedTable:
    grid, th_rows = _grid(table)
    if not grid or max(len(r) for r in grid) == 0:
        return ParsedTable(kind="layout")
    width = len(grid[0])

    def row_has_data(r: list[tuple[str, bool]]) -> bool:
        return any(_is_data_cell(t) for (t, filler) in r[1:] if not filler)

    first_data = next((i for i, r in enumerate(grid) if row_has_data(r)), None)
    if first_data is None:
        text = [clean(" ".join(dict.fromkeys(t for t, _ in r if t))) for r in grid]
        return ParsedTable(kind="layout", layout_text=[t for t in text if t])

    # Title rows: col0 text repeated across the whole row (colspan from the left).
    title = ""
    header_idx: list[int] = []
    h_end = 0
    for i in range(first_data):
        nonempty = [t for t, _ in grid[i] if t]
        if any(grid[i][j][0] for j in range(1, width)):
            h_end = i + 1
    for i in range(h_end):
        row = grid[i]
        nonempty = [t for t, _ in row if t]
        if not nonempty:
            continue
        if row[0][0] and len(set(nonempty)) == 1 and len(nonempty) >= 2:
            title = title or row[0][0]
            continue
        header_idx.append(i)
    # rows between h_end and first_data are body rows (section labels)
    if th_rows and th_rows > h_end:
        h_end = min(th_rows, first_data)

    body = []
    for i in range(h_end, len(grid)):
        cells = [("" if filler else t) for (t, filler) in grid[i]]
        body.append(_merge_symbols(cells))

    keep = [0] + [j for j in range(1, width) if any(b[j] for b in body)]
    if len(keep) == 1:
        return ParsedTable(kind="layout", layout_text=[clean(" ".join(t for t, _ in r if t)) for r in grid])

    header: list[str] = []
    for j in keep:
        parts: list[str] = []
        for i in header_idx:
            t = grid[i][j][0]
            if t and (not parts or parts[-1] != t):
                parts.append(t)
        header.append(" / ".join(parts))
    if not header[0]:
        header[0] = "Item"
    for idx in range(1, len(header)):
        if not header[idx]:
            header[idx] = f"Column {idx}"

    units = detect_units(title, *(grid[i][0][0] for i in header_idx), *(" ".join(t for t, _ in grid[i]) for i in header_idx[:3]))

    rows: list[TableRow] = []
    group = ""
    for b in body:
        label = b[0]
        vals = [b[j] for j in keep[1:]]
        if not label and not any(vals):
            continue
        if label and not any(vals):
            group = label.rstrip(":")
            rows.append(TableRow(label=label, cells=vals, group="", kind="section"))
            continue
        rows.append(TableRow(label=label, cells=vals, group=group, kind="data"))
        if re.match(r"^total\b", label, re.I):
            group = ""

    pt = ParsedTable(kind="data", header=header, rows=rows, title=title, units=units)
    labels = [r.label for r in rows if r.kind == "data"]
    toc_like = sum(1 for l in labels if re.match(r"^(item\s+\d|part\s+[iv]+)\b", l, re.I))
    if len(labels) >= 3 and toc_like / len(labels) >= 0.6:
        pt.kind = "toc"
    return pt


def table_to_markdown(header: list[str], rows: list[TableRow], with_groups: bool = True) -> str:
    def esc(s: str) -> str:
        return s.replace("|", "/")

    lines = ["| " + " | ".join(esc(h) for h in header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        if r.kind == "section":
            lines.append("| " + esc(r.label) + " |" + " |" * (len(header) - 1))
            continue
        label = r.label
        if with_groups and r.group:
            label = f"{r.group} — {label}"
        lines.append("| " + " | ".join(esc(x) for x in [label] + r.cells) + " |")
    return "\n".join(lines)
