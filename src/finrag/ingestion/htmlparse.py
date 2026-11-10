"""EDGAR HTML -> ordered blocks (text paragraphs and tables) tagged with filing structure.

Tracks Part / Item / Note as it walks, attaches units, captions and footnotes to tables, and counts
page-break markers so citations can carry an approximate page number.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator

from bs4 import BeautifulSoup, NavigableString, Tag

from .tables import ParsedTable, clean, detect_units, parse_table

_BLOCK = {"p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "article",
          "center", "blockquote", "hr", "body", "html", "tr", "td", "th", "tbody", "thead", "pre"}
_BLOCK_LIST = sorted(_BLOCK - {"tr", "td", "th", "tbody", "thead"})
_SKIP_TAGS = {"script", "style", "head", "ix:header", "title", "meta", "link"}

_ITEM_RE = re.compile(r"^item\s+(\d{1,2}(?:\.\d{2})?[a-c]?)(?:\s*[\.\:\-–—]\s*|\s+)(.*)$", re.I | re.S)
_PART_RE = re.compile(r"^part\s+(i{1,3}|iv)\b[\s\.\:\-–—]*(.*)$", re.I)
_NOTE_RE = re.compile(r"^note\s+(\d{1,2})\s*[\.\:\-–—]+\s*(.+)$", re.I)
_NUMNOTE_RE = re.compile(r"^(\d{1,2})\.\s+([A-Z][A-Za-z ,&'’()/\-]{2,80})$")
_FOOT_RE = re.compile(r"^(\(\s*(?:\d{1,2}|[a-z])\s*\)|\*{1,3}|†|‡|\d{1,2}\))\s*\S")
_JUNK_RE = re.compile(r"^(table of contents|page \d+|\d{1,3}|-?\s*\d{1,3}\s*-?|index)$", re.I)


@dataclass
class Block:
    idx: int
    kind: str                       # text | heading | table
    text: str = ""
    table: ParsedTable | None = None
    part: str = ""
    item: str = ""
    section: str = ""
    note: str = ""
    page: int = 1
    caption: str = ""
    units: str = ""
    footnotes: list[str] = field(default_factory=list)


def _hidden(el: Tag) -> bool:
    style = (el.get("style") or "").lower().replace(" ", "")
    return "display:none" in style


def _page_break(el: Tag) -> tuple[bool, bool]:
    style = (el.get("style") or "").lower().replace(" ", "")
    before = "page-break-before:always" in style or "break-before:page" in style
    after = "page-break-after:always" in style or "break-after:page" in style
    return before, after


def _walk(container: Tag) -> Iterator[tuple]:
    buf: list[str] = []

    def flush() -> Iterator[tuple]:
        if buf:
            text = clean(" ".join(buf))
            buf.clear()
            if text:
                yield ("text", text)

    for child in container.children:
        if isinstance(child, NavigableString):
            if type(child).__name__ in ("Comment", "Doctype", "CData", "ProcessingInstruction"):
                continue
            buf.append(str(child))
            continue
        if not isinstance(child, Tag):
            continue
        name = (child.name or "").lower()
        if name in _SKIP_TAGS or _hidden(child):
            continue
        if name == "br":
            yield from flush()
            continue
        if name == "table":
            yield from flush()
            yield ("table", child)
            continue
        before, after = _page_break(child)
        if name in _BLOCK:
            yield from flush()
            if before:
                yield ("pagebreak",)
            yield from _walk(child)
            yield from flush()
            if after:
                yield ("pagebreak",)
            continue
        # inline element: may still wrap blocks (e.g. ix:nonNumeric text blocks)
        if child.find(_BLOCK_LIST):
            yield from flush()
            if before:
                yield ("pagebreak",)
            yield from _walk(child)
            yield from flush()
            if after:
                yield ("pagebreak",)
        else:
            buf.append(child.get_text(" "))
    yield from flush()


def _heading(text: str, state: dict) -> bool:
    """Update structural state if `text` is a Part/Item/Note heading. Returns True if it was one."""
    if len(text) > 220:
        return False
    if m := _PART_RE.match(text):
        state["part"] = f"Part {m.group(1).upper()}"
        return True
    if m := _ITEM_RE.match(text):
        title = clean(m.group(2))
        if title and not title[0].isupper():
            return False
        state["item"] = f"Item {m.group(1).upper()}"
        state["section"] = title.rstrip(".") if title else state.get("section", "")
        state["note"] = ""
        return True
    if m := _NOTE_RE.match(text):
        state["note"] = f"Note {m.group(1)} — {clean(m.group(2)).rstrip('.')}"
        return True
    if state.get("item") == "Item 8" and (m := _NUMNOTE_RE.match(text)):
        state["note"] = f"Note {m.group(1)} — {m.group(2).strip()}"
        return True
    return False


def parse_html(html: str) -> list[Block]:
    soup = BeautifulSoup(html, "lxml")
    root = soup.body or soup
    state = {"part": "", "item": "", "section": "", "note": ""}
    blocks: list[Block] = []
    page = 1
    recent_short: list[str] = []        # last few short text blocks (caption / units candidates)
    pending_units = ""
    last_table: Block | None = None
    since_table = 99

    def mk(kind: str, **kw) -> Block:
        b = Block(idx=len(blocks), kind=kind, part=state["part"], item=state["item"], section=state["section"],
                  note=state["note"], page=page, **kw)
        blocks.append(b)
        return b

    for ev in _walk(root):
        if ev[0] == "pagebreak":
            page += 1
            continue
        if ev[0] == "text":
            text = ev[1]
            if _JUNK_RE.match(text):
                continue
            if _heading(text, state):
                mk("heading", text=text)
                last_table = None
                recent_short.clear()
                continue
            if last_table is not None and since_table < 8 and len(text) < 700 and _FOOT_RE.match(text):
                last_table.footnotes.append(text)
                since_table += 1
                continue
            last_table = None
            u = detect_units(text) if len(text) < 140 else ""
            if u:
                pending_units = u
            recent_short.append(text if len(text) <= 160 else "")
            recent_short[:] = recent_short[-3:]
            mk("text", text=text)
            since_table += 1
            continue
        if ev[0] == "table":
            pt = parse_table(ev[1])
            if pt.kind == "toc":
                continue
            if pt.kind == "layout":
                for line in pt.layout_text:
                    if line and not _JUNK_RE.match(line):
                        if not _heading(line, state):
                            mk("text", text=line)
                continue
            caption = pt.title
            if not caption:
                for cand in reversed(recent_short):
                    if cand and not cand.rstrip().endswith(".") and not detect_units(cand):
                        caption = cand
                        break
            b = mk("table", table=pt, caption=caption, units=pt.units or pending_units)
            pending_units = ""
            recent_short.clear()
            last_table = b
            since_table = 0
    return blocks
