"use strict";
/* FinRAG UI. All dynamic text goes through textContent (never innerHTML), so filing text cannot inject markup. */
const $ = (id) => document.getElementById(id);
const state = { history: [], last: null };

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === false || v == null) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return el;
}
const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); return el; };
const checked = (name) => document.querySelector(`input[name=${name}]:checked`).value;
const svgNS = "http://www.w3.org/2000/svg";
function tickSvg(kind, draw) {
  const s = document.createElementNS(svgNS, "svg");
  s.setAttribute("viewBox", "0 0 24 24");
  s.setAttribute("class", "tickmark" + (kind === "bad" ? " bad" : "") + (kind === "none" ? " none" : "") + (draw ? " draw" : ""));
  s.setAttribute("role", "img");
  s.setAttribute("aria-label", kind === "ok" ? "Matches its source" : kind === "bad" ? "Not matched to its source" : "");
  const p = document.createElementNS(svgNS, "path");
  p.setAttribute("d", kind === "bad" ? "M6 6l12 12M18 6L6 18" : "M4 13l5 5L20 6");
  s.append(p);
  return s;
}
async function api(path, body) {
  const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : undefined);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `Request failed (${r.status})`);
  return data;
}
const pct = (x) => (x == null ? "n/a" : Math.round(x * 100) + "%");

/* ---------------------------------------------------------------- corpus + starters */
const WHY = ["total assets, XBRL and table agree", "a year inside the question, not a filing", "June fiscal year: Q2 ends in December",
  "text answer from Item 9A", "abbreviation: capex", "company not in the corpus: should refuse"];
async function init() {
  const st = await api("/api/status");
  const box = clear($("corpus"));
  const byCo = {};
  for (const f of st.filings) (byCo[f.ticker] ||= { name: f.company, items: [] }).items.push(f);
  for (const [t, c] of Object.entries(byCo)) {
    box.append(h("div", { class: "co" }, h("b", null, `${c.name} (${t})`),
      h("span", null, c.items.map((f) => `${f.filing_type} ${f.fiscal_year ? (f.fiscal_period && f.fiscal_period !== "FY" ? f.fiscal_period + " " : "") + "FY" + f.fiscal_year : f.period_end}`).join(", "))));
  }
  $("demo-note").hidden = !st.synthetic;
  $("engine").textContent = `${st.n_chunks} passages. Embeddings: ${st.embedder}. Answers: ${st.llm}.`;
  $("bench-btn").hidden = !st.has_gold;
  const ul = $("starter-list");
  (st.examples || []).forEach((q, i) => ul.append(h("li", null, h("button", { type: "button", onclick: () => { $("question").value = q; ask(); } },
    h("span", null, q), WHY[i] ? h("span", { class: "why" }, WHY[i]) : null))));
}

/* ---------------------------------------------------------------- ask */
async function ask(ev) {
  if (ev) ev.preventDefault();
  const q = $("question").value.trim();
  if (!q) { showError("Enter a question first."); return; }
  showError("");
  $("ask-btn").disabled = true; $("busy").hidden = false;
  try {
    const ans = await api("/api/ask", { question: q, mode: checked("mode"), rewrite: checked("rewrite"), k: +$("k").value, history: state.history.slice(-3) });
    state.last = ans; state.history.push(q);
    $("clear-btn").hidden = false; $("starters").hidden = true;
    render(ans);
    for (const id of ["compare", "bench"]) clear($(id));
  } catch (e) { showError(e.message); }
  finally { $("ask-btn").disabled = false; $("busy").hidden = true; }
}
function showError(msg) { const e = $("error"); e.textContent = msg; e.hidden = !msg; }

/* ---------------------------------------------------------------- render answer */
function withCites(text) {
  const frag = document.createDocumentFragment();
  let last = 0;
  for (const m of text.matchAll(/\[(\d{1,2})\]/g)) {
    frag.append(text.slice(last, m.index));
    frag.append(h("button", { type: "button", class: "cite", "aria-label": `Show source ${m[1]}`, onclick: () => jumpTo(+m[1]) }, m[1]));
    last = m.index + m[0].length;
  }
  frag.append(text.slice(last));
  return frag;
}
function jumpTo(n) {
  select("evidence");
  const el = document.getElementById("src-" + n);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
}
function render(a) {
  $("result").hidden = false;
  const box = clear($("answer"));
  for (const line of a.answer.split("\n").filter((l) => l.trim())) {
    const claim = a.claims.find((c) => c.sentence.trim() === line.trim());
    const kind = a.abstained || !claim || claim.supported == null ? "none" : claim.supported ? "ok" : "bad";
    box.append(h("div", { class: "line" + (a.abstained ? " abstain" : "") }, tickSvg(kind, true), h("div", null, withCites(line))));
  }
  const w = clear($("warnings"));
  a.warnings.forEach((x) => w.append(h("li", null, x)));
  const parts = [];
  if (a.support_rate != null) parts.push(`${pct(a.support_rate)} of statements matched their cited source`);
  if (a.citation_coverage != null) parts.push(`${pct(a.citation_coverage)} carry a citation`);
  parts.push(`${a.mode} search, ${a.generator} answer, ${Math.round(a.timings_ms.total)} ms`);
  $("meter").textContent = parts.join(". ") + ".";
  renderReading(a.rewrite);
  renderEvidence(a);
  renderClaims(a);
  select("evidence");
}
const FILTER_LABELS = { ticker: "Company", filing_type: "Filing type", fiscal_year: "Fiscal year", fiscal_period: "Period", latest: "Latest filing" };
function renderReading(rw) {
  const box = clear($("reading"));
  if (!rw) return;
  const dl = h("dl");
  dl.append(h("dt", null, "Searched as"), h("dd", null, rw.standalone));
  const f = Object.entries(rw.filters || {});
  dl.append(h("dt", null, "Filters applied"));
  dl.append(f.length ? h("dd", null, h("div", { class: "chips" }, f.map(([k, v]) => h("span", null, `${FILTER_LABELS[k] || k}: ${Array.isArray(v) ? v.join(", ") : v}`)))) : h("dd", null, "None. All filings are searched."));
  if (rw.expansions?.length) dl.append(h("dt", null, "Terms expanded"), h("dd", null, h("ul", null, rw.expansions.map((x) => h("li", null, x)))));
  if (rw.assumptions?.length) dl.append(h("dt", null, "Assumptions I made"), h("dd", null, h("ul", null, rw.assumptions.map((x) => h("li", null, x)))));
  if (rw.clarification) dl.append(h("dt", null, "Needs clarification"), h("dd", null, rw.clarification));
  if (rw.mode === "off") dl.append(h("dd", null, "Question rewriting is off, so the question was searched as typed."));
  box.append(dl);
}

/* ---------------------------------------------------------------- evidence */
function ledger(c, hitSpan) {
  const td = c.table_data;
  const wrap = h("div", { class: "ledger" });
  const cap = [c.caption, c.units].filter(Boolean).join(", ");
  const table = h("table", null, cap ? h("caption", null, cap) : null,
    h("thead", null, h("tr", null, td.header.map((x) => h("th", { scope: "col" }, x)))));
  const body = h("tbody");
  const norm = (s) => s.replace(/\s+/g, " ").trim();
  td.rows.forEach((row, i) => {
    const label = norm(row[0]);
    const hit = hitSpan && label && norm(hitSpan).includes(label) && row.slice(1).some((v) => v && norm(hitSpan).includes(norm(v)));
    body.append(h("tr", { class: hit ? "hit" : (td.groups?.[i] ? "" : "") }, row.map((v) => h("td", null, v))));
  });
  table.append(body); wrap.append(table);
  if (c.footnotes?.length) wrap.append(h("p", { class: "foot" }, c.footnotes.join(" ")));
  return wrap;
}
function passage(c, span) {
  const p = h("p", { class: "passage" });
  const t = c.text;
  const i = span ? t.indexOf(span) : -1;
  if (i >= 0) { p.append(t.slice(0, i), h("mark", null, span), t.slice(i + span.length)); } else p.append(t);
  return p;
}
function renderEvidence(a) {
  const box = clear($("evidence"));
  const cited = Object.fromEntries(a.citations.map((c) => [c.n, c]));
  a.retrieved.forEach((r, idx) => {
    const n = idx + 1, cit = cited[n], m = r.meta;
    const where = [m.item, m.note || "", m.table_id ? "Table " + m.table_id : "", m.page ? "page " + m.page : ""].filter(Boolean).join(", ");
    const ranks = Object.entries(r.ranks).map(([k, v]) => `${k === "sparse" ? "keyword" : k} rank ${v}`).join(", ");
    const body = r.table_data ? ledger(r, cit?.span) : passage(r, cit?.span);
    box.append(h("section", { class: "src" + (cit ? " cited" : ""), id: "src-" + n },
      h("h3", null, cit ? h("span", { class: "n" }, n) : null, r.label, r.outside_filter ? h("span", { class: "tag" }, "outside the requested filing") : null),
      h("p", { class: "where" }, [where, ranks].filter(Boolean).join(". ") + (m.accession ? `. Accession ${m.accession}.` : "")),
      body));
  });
  if (!a.retrieved.length) box.append(h("p", { class: "lede" }, "No passages were retrieved."));
}

/* ---------------------------------------------------------------- claims */
function renderClaims(a) {
  const box = clear($("claims"));
  const facts = a.claims.filter((c) => c.supported != null);
  if (!facts.length) { box.append(h("p", { class: "lede" }, a.abstained ? "The answer is a refusal, so there are no statements to check." : "No checkable statements.")); return; }
  box.append(h("p", { class: "lede" }, "Each statement is checked against the passage it cites: every number must appear there, and most of its wording must too. This is a lexical check, not a guarantee."));
  for (const c of facts) {
    const bits = [];
    if (c.numeric_ok != null) bits.push(c.numeric_ok ? "Numbers found in the source" : "A number is missing from the source");
    bits.push(`wording overlap ${pct(c.overlap)}`);
    box.append(h("div", { class: "claim" }, tickSvg(c.supported ? "ok" : "bad", false), h("div", { class: "s" }, withCites(c.sentence)),
      h("div", { class: "d" + (c.supported ? "" : " bad") }, (c.supported ? "Matches its source. " : "Not matched. ") + bits.join(", ") + "."),
      c.issues.length ? h("div", { class: "d bad" }, c.issues.join(" ")) : null,
      c.span ? h("blockquote", null, c.span) : null));
  }
}

/* ---------------------------------------------------------------- compare */
async function compare() {
  const q = state.last?.question; if (!q) return;
  const box = clear($("compare")); box.append(h("p", { class: "lede" }, "Running…"));
  try {
    const r = await api("/api/compare", { question: q, rewrite: checked("rewrite"), k: +$("k").value, history: state.history.slice(-4, -1) });
    clear(box);
    const sets = Object.values(r.results).map((x) => new Set(x.chunks.map((c) => c.chunk_id)));
    const inAll = (id) => sets.every((s) => s.has(id));
    for (const [mode, res] of Object.entries(r.results)) {
      box.append(h("div", null, h("h3", null, { dense: "Dense", sparse: "Keyword", hybrid: "Hybrid" }[mode]), h("div", { class: "ms" }, `${res.ms} ms`),
        h("ol", null, res.chunks.map((c) => h("li", { class: inAll(c.chunk_id) ? "all" : "" }, c.label, h("small", null, c.element_type.startsWith("table") ? "table" : "text"))))));
    }
    box.append(h("p", { class: "legend" }, "Shaded rows were returned by all three methods."));
  } catch (e) { clear(box).append(h("p", { class: "error" }, e.message)); }
}

/* ---------------------------------------------------------------- benchmark */
const f1 = (s, p = true) => (!s || s.n === 0 || s.mean !== s.mean ? "n/a" : p ? `${Math.round(s.mean * 100)}% (${Math.round(s.lo * 100)}-${Math.round(s.hi * 100)})` : Math.round(s.mean));
async function bench(run) {
  const box = clear($("bench")); box.append(h("p", { class: "lede" }, run ? "Running the benchmark…" : "Loading…"));
  try {
    const r = run ? await api("/api/eval/run", {}) : await api("/api/eval/latest");
    clear(box);
    if (!r.available) { box.append(h("p", { class: "lede" }, "No results yet. Run the benchmark to create them.")); return; }
    const cols = [["recall@3", "Recall@3"], ["mrr", "MRR"], ["answer_correct", "Answer correct"], ["faithful", "Faithful"], ["abstain_ok", "Refuses when it should"], ["latency_ms", "Latency ms"]];
    const t = h("table", null, h("thead", null, h("tr", null, h("th", { scope: "col" }, "Setup"), cols.map(([, l]) => h("th", { scope: "col" }, l)))));
    const tb = h("tbody");
    for (const [name, m] of Object.entries(r.summary))
      tb.append(h("tr", null, h("td", { class: "cfg" }, name), cols.map(([k]) => h("td", null, k === "mrr" ? m[k].mean.toFixed(2) : f1(m[k], k !== "latency_ms")))));
    t.append(tb); box.append(t);
    const list = h("ul", { class: "bench-list" });
    for (const [k, v] of Object.entries(r.comparisons)) list.append(h("li", null, `${k}: ${v.diff >= 0 ? "+" : ""}${(v.diff * 100).toFixed(1)} points (${(v.lo * 100).toFixed(1)} to ${(v.hi * 100).toFixed(1)}), ${v.significant ? "clear difference" : "could be noise"}.`));
    box.append(h("p", { class: "bench-note" }, `${r.n_questions} questions, top ${r.k} passages, judge: ${r.judge}. Differences between setups:`), list);
  } catch (e) { clear(box).append(h("p", { class: "error" }, e.message)); }
}

/* ---------------------------------------------------------------- tabs + wiring */
const TABS = ["evidence", "claims", "compare", "bench"];
function select(name) {
  for (const t of TABS) {
    const on = t === name;
    $("t-" + t).setAttribute("aria-selected", on); $("t-" + t).tabIndex = on ? 0 : -1;
    $("p-" + t).hidden = !on;
  }
  if (name === "bench" && !$("bench").firstChild) bench(false);
}
TABS.forEach((t, i) => $("t-" + t).addEventListener("click", () => select(t)));
$("t-evidence").parentElement.addEventListener("keydown", (e) => {
  const i = TABS.findIndex((t) => $("t-" + t).getAttribute("aria-selected") === "true");
  if (e.key === "ArrowRight" || e.key === "ArrowLeft") { const n = (i + (e.key === "ArrowRight" ? 1 : TABS.length - 1)) % TABS.length; select(TABS[n]); $("t-" + TABS[n]).focus(); }
});
$("ask-form").addEventListener("submit", ask);
$("question").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); } });
$("compare-btn").addEventListener("click", compare);
$("bench-btn").addEventListener("click", () => bench(true));
$("clear-btn").addEventListener("click", () => { state.history = []; state.last = null; $("result").hidden = true; $("starters").hidden = false; $("clear-btn").hidden = true; $("question").value = ""; $("question").focus(); });
init().catch((e) => { $("corpus").textContent = e.message; });
