"""Generate SYNTHETIC SEC-style filings and a gold Q&A set, for development, tests and demos.

Everything here is fictional: two made-up companies (Northwind Robotics, Harbor Lane Foods) with made-up
numbers. The HTML mimics real EDGAR layout (colspan headers, separate "$" and ")" cells, inline XBRL tags,
footnotes, page breaks) so the parser can be developed and tested offline. Results measured on this data
are a smoke test, NOT evidence for a report: evaluate on real EDGAR filings / FinanceBench for that.

Run:  python sample_data/generate.py            (writes into sample_data/)
"""
from __future__ import annotations

import json
import random
from datetime import date
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent
FILINGS = OUT / "filings"

COMPANIES = {
    "NWRX": dict(name="Northwind Robotics Inc.", cik="0009000001", fye="1231", rev0=4200, shares=310),
    "HLFC": dict(name="Harbor Lane Foods Corp.", cik="0009000002", fye="0630", rev0=2600, shares=190),
}
BANNER = "SYNTHETIC FILING FOR SOFTWARE TESTING. Fictional company and fictional numbers. Not a real SEC document."
TLA_DUE, RCF_DUE, SN_DUE = 2027, 2028, 2030


# ------------------------------------------------------------------ financial model (millions) ---------------
def revenue(t: str, fy: int) -> int:
    rnd = random.Random(f"rev-{t}-{fy}")
    return round(COMPANIES[t]["rev0"] * (1.11 ** (fy - 2022)) * rnd.uniform(0.98, 1.02))


def debt(t: str, fy: int) -> dict:
    R = revenue(t, fy)
    base = round(R * 0.18 / 25) * 25
    tla_out = round(base * (1 - 0.10 * (fy - 2022)))
    sn = round(R * 0.20 / 50) * 50
    rcf = round(R * 0.07 / 25) * 25
    sched, remaining = {}, tla_out
    for y in range(fy + 1, fy + 6):
        pay = 0
        if y < TLA_DUE:
            pay = min(round(base * 0.10), remaining)
            remaining -= pay
        elif y == TLA_DUE:
            pay, remaining = remaining, 0
        if y == RCF_DUE:
            pay += rcf
        sched[str(y)] = pay
    thereafter = sn
    principal = tla_out + sn + rcf
    costs = round(principal * 0.006)
    carrying = principal - costs
    cur = sched[str(fy + 1)]
    return dict(tla=tla_out, sn=sn, rcf=rcf, principal=principal, costs=costs, carrying=carrying, cur=cur,
                lt=carrying - cur, sched=sched, thereafter=thereafter, rcf_commit=round(rcf * 4 / 50) * 50)


def income(t: str, fy: int, R: int | None = None) -> dict:
    rnd = random.Random(f"inc-{t}-{fy}-{R}")
    R = R or revenue(t, fy)
    cogs = round(R * rnd.uniform(0.60, 0.64))
    gross = R - cogs
    opex = round(R * rnd.uniform(0.20, 0.22))
    op = gross - opex
    d = debt(t, fy)
    interest = round(d["principal"] * 0.045)
    pre = op - interest
    tax = round(pre * 0.22)
    net = pre - tax
    sh = COMPANIES[t]["shares"]
    return dict(rev=R, cogs=cogs, gross=gross, opex=opex, op=op, interest=interest, pre=pre, tax=tax, net=net,
                eps=round(net / sh, 2), eps_d=round(net / (sh * 1.02), 2))


def balance(t: str, fy: int, key: str = "", R: int | None = None, dbt: dict | None = None) -> dict:
    rnd = random.Random(f"bs-{t}-{fy}-{key}")
    R = R or revenue(t, fy)
    d = dbt or debt(t, fy)
    cash, ar, inv, oca = round(R * rnd.uniform(.10, .12)), round(R * rnd.uniform(.11, .13)), round(R * rnd.uniform(.08, .10)), round(R * .02)
    tca = cash + ar + inv + oca
    ppe, rou, gw = round(R * rnd.uniform(.33, .37)), round(R * .06), round(R * .15)
    ta = tca + ppe + rou + gw
    ap, acc, cur_lease = round(R * .08), round(R * .06), round(rou * .15)
    tcl = ap + acc + d["cur"] + cur_lease
    lt_lease, oth = round(rou * .85), round(R * .03)
    tl = tcl + d["lt"] + lt_lease + oth
    eq = ta - tl
    stock = round(eq * .4)
    return dict(cash=cash, ar=ar, inv=inv, oca=oca, tca=tca, ppe=ppe, rou=rou, gw=gw, ta=ta, ap=ap, acc=acc,
                cur_debt=d["cur"], cur_lease=cur_lease, tcl=tcl, lt_debt=d["lt"], lt_lease=lt_lease, oth=oth, tl=tl,
                stock=stock, re=eq - stock, eq=eq, tle=tl + eq)


def leases(t: str, fy: int) -> dict:
    rnd = random.Random(f"lease-{t}-{fy}")
    R = revenue(t, fy)
    rou = round(R * .06)
    op_liab = round(rou * 1.0)
    fin_liab = round(R * 0.012)
    sched_op = {str(y): round(op_liab * w) for y, w in zip(range(fy + 1, fy + 6), (.22, .19, .16, .13, .10))}
    after_op = round(op_liab * 0.30)
    sched_fin = {str(y): round(fin_liab * w) for y, w in zip(range(fy + 1, fy + 6), (.25, .22, .20, .15, .10))}
    after_fin = round(fin_liab * 0.18)
    tot_op, tot_fin = sum(sched_op.values()) + after_op, sum(sched_fin.values()) + after_fin
    imp_op, imp_fin = round(tot_op * rnd.uniform(.08, .10)), round(tot_fin * rnd.uniform(.06, .08))
    return dict(sched_op=sched_op, after_op=after_op, sched_fin=sched_fin, after_fin=after_fin, tot_op=tot_op,
                tot_fin=tot_fin, imp_op=imp_op, imp_fin=imp_fin, pv_op=tot_op - imp_op, pv_fin=tot_fin - imp_fin,
                op_cost=round(R * .021), fin_amort=round(R * .006), fin_int=round(R * .0015), variable=round(R * .004))


def largest_customer_pct(t: str, fy: int) -> float:
    return round(random.Random(f"cust-{t}-{fy}").uniform(14, 26), 1)


# ------------------------------------------------------------------------------- html helpers ---------------
def n(v: int | float, dec: int = 0) -> str:
    return f"{v:,.{dec}f}"


def td(text: str = "", colspan: int = 1, align: str = "left", bold: bool = False, indent: int = 0) -> str:
    st = f"text-align:{align};" + ("font-weight:700;" if bold else "") + (f"padding-left:{indent}px;" if indent else "")
    cs = f' colspan="{colspan}"' if colspan > 1 else ""
    return f'<td{cs} style="{st}">{text}</td>'


def ix(concept: str, ctx: str, v, unit: str = "usd", scale: int = 6, dec: int = 0) -> str:
    if unit == "usdPerShare":
        return f'<ix:nonFraction name="us-gaap:{concept}" contextRef="{ctx}" unitRef="{unit}" decimals="2" scale="0">{v:.2f}</ix:nonFraction>'
    return f'<ix:nonFraction name="us-gaap:{concept}" contextRef="{ctx}" unitRef="{unit}" decimals="-{scale}" scale="{scale}" format="ixt:num-dot-decimal">{n(v, dec)}</ix:nonFraction>'


def vcell(v, dollar: bool = False, neg: bool = False, tag: tuple | None = None) -> str:
    """One value = three cells like real EDGAR: [$] [number or '(number'] [')']."""
    shown = ix(*tag[:2], v, *tag[2:]) if tag else (n(v) if isinstance(v, int) else v)
    if tag and neg:
        shown = f"({shown}"
    elif neg:
        shown = f"({shown}"
    return td("$" if dollar else "") + td(shown, align="right") + td(")" if neg else "")


def row(label: str, vals: list, dollar=False, bold=False, indent=0, tags: list | None = None, neg=False) -> str:
    cells = td(label, bold=bold, indent=indent)
    for i, v in enumerate(vals):
        cells += vcell(v, dollar, neg, tags[i] if tags else None)
    return f"<tr>{cells}</tr>"


def header_rows(label_hdr: str, spans: list[tuple[str, int]] | None, cols: list[str], title: str | None = None) -> str:
    out = ""
    ncols = 1 + 3 * len(cols)
    if title:
        out += f'<tr><td colspan="{ncols}" style="font-weight:700;">{title}</td></tr>'
    if spans:
        out += "<tr>" + td("") + "".join(td(s, colspan=3 * w, align="center", bold=True) for s, w in spans) + "</tr>"
    out += "<tr>" + td(label_hdr) + "".join(td(c, colspan=3, align="center", bold=True) for c in cols) + "</tr>"
    return out


def table(inner: str) -> str:
    return f'<table style="width:100%;border-collapse:collapse;">{inner}</table>'


def p(text: str) -> str:
    return f"<div><span>{text}</span></div>"


def h_item(num: str, title: str) -> str:
    return f'<div><span style="font-weight:700;">Item {num}. {title}</span></div>'


def h_note(num: int, title: str) -> str:
    return f'<div><span style="font-weight:700;">Note {num} — {title}</span></div>'


PB = '<hr style="page-break-after:always"/>'


def contexts(period_end: date, prior_end: date, start: date, prior_start: date, extra: list[tuple] | None = None) -> str:
    def inst(cid, d):
        return f'<xbrli:context id="{cid}"><xbrli:entity><xbrli:identifier scheme="http://www.sec.gov/CIK">0009000000</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:instant>{d}</xbrli:instant></xbrli:period></xbrli:context>'

    def dur(cid, a, b):
        return f'<xbrli:context id="{cid}"><xbrli:entity><xbrli:identifier scheme="http://www.sec.gov/CIK">0009000000</xbrli:identifier></xbrli:entity><xbrli:period><xbrli:startDate>{a}</xbrli:startDate><xbrli:endDate>{b}</xbrli:endDate></xbrli:period></xbrli:context>'

    seg = (f'<xbrli:context id="seg_tla"><xbrli:entity><xbrli:identifier scheme="http://www.sec.gov/CIK">0009000000</xbrli:identifier>'
           f'<xbrli:segment><xbrldi:explicitMember dimension="us-gaap:DebtInstrumentAxis">nwrx:TermLoanAMember</xbrldi:explicitMember></xbrli:segment></xbrli:entity>'
           f'<xbrli:period><xbrli:instant>{period_end}</xbrli:instant></xbrli:period></xbrli:context>')
    body = inst("i_cur", period_end) + inst("i_pri", prior_end) + dur("d_cur", start, period_end) + dur("d_pri", prior_start, prior_end) + seg
    for cid, a, b in (extra or []):
        body += dur(cid, a, b)
    return f'<ix:header style="display:none"><ix:resources>{body}</ix:resources></ix:header>'


def doc(title: str, ctx_xml: str, body: str) -> str:
    return (f'<html xmlns:ix="http://www.xbrl.org/2013/inlineXBRL" xmlns:xbrli="http://www.xbrl.org/2003/instance">'
            f'<head><title>{escape(title)}</title></head><body>{ctx_xml}'
            f'<div style="color:#900">{BANNER}</div>{body}</body></html>')


def toc(items: list[tuple[str, int]]) -> str:
    rows = "".join(f"<tr>{td(f'Item {k}')}{td(str(pg), align='right')}</tr>" for k, pg in items)
    return table(f"<tr>{td('Table of Contents', 2, bold=True)}</tr>{rows}")


# ------------------------------------------------------------------------------- statements ---------------
def bs_table(t: str, cur: dict, pri: dict, cur_lbl: str, pri_lbl: str, tag_ctx=("i_cur", "i_pri")) -> str:
    c1, c2 = tag_ctx

    def r(label, key, concept=None, first=False, bold=False, indent=12):
        tags = [(concept, c1), (concept, c2)] if concept else None
        return row(label, [cur[key], pri[key]], dollar=first or bold, bold=bold, indent=indent, tags=tags)

    inner = header_rows("(in millions)", None, [cur_lbl, pri_lbl], title="Consolidated Balance Sheets")
    inner += f'<tr>{td("ASSETS", bold=True)}{td("", 6)}</tr><tr>{td("Current assets:", indent=6)}{td("", 6)}</tr>'
    inner += r("Cash and cash equivalents", "cash", "CashAndCashEquivalentsAtCarryingValue", first=True)
    inner += r("Accounts receivable, net", "ar", "AccountsReceivableNetCurrent")
    inner += r("Inventories, net", "inv", "InventoryNet")
    inner += r("Other current assets", "oca")
    inner += r("Total current assets", "tca", "AssetsCurrent", bold=True, indent=6)
    inner += r("Property, plant and equipment, net", "ppe", first=True)
    inner += r("Operating lease right-of-use assets", "rou", "OperatingLeaseRightOfUseAsset")
    inner += r("Goodwill", "gw")
    inner += r("Total assets", "ta", "Assets", bold=True, indent=0)
    inner += f'<tr>{td("LIABILITIES AND STOCKHOLDERS’ EQUITY", bold=True)}{td("", 6)}</tr><tr>{td("Current liabilities:", indent=6)}{td("", 6)}</tr>'
    inner += r("Accounts payable", "ap", first=True)
    inner += r("Accrued liabilities", "acc")
    inner += r("Current portion of long-term debt", "cur_debt")
    inner += r("Current portion of operating lease liabilities", "cur_lease")
    inner += r("Total current liabilities", "tcl", "LiabilitiesCurrent", bold=True, indent=6)
    inner += r("Long-term debt, net of current portion", "lt_debt", "LongTermDebtNoncurrent", first=True)
    inner += r("Operating lease liabilities, non-current", "lt_lease")
    inner += r("Other long-term liabilities", "oth")
    inner += r("Total liabilities", "tl", "Liabilities", bold=True, indent=0)
    inner += r("Common stock and additional paid-in capital", "stock", first=True)
    inner += r("Retained earnings", "re")
    inner += r("Total stockholders’ equity", "eq", "StockholdersEquity", bold=True, indent=0)
    inner += r("Total liabilities and stockholders’ equity", "tle", "LiabilitiesAndStockholdersEquity", bold=True, indent=0)
    return table(inner)


def is_table(cur: dict, pri: dict, spans, cols, ctx=("d_cur", "d_pri")) -> str:
    c1, c2 = ctx

    def r(label, key, concept=None, first=False, bold=False, neg=False):
        tags = [(concept, c1), (concept, c2)] if concept else None
        return row(label, [cur[key], pri[key]], dollar=first, bold=bold, indent=0 if bold else 12, tags=tags, neg=neg)

    inner = header_rows("(in millions, except per share data)", spans, cols, title="Consolidated Statements of Income")
    inner += r("Revenue", "rev", "Revenues", first=True)
    inner += r("Cost of revenue", "cogs", "CostOfRevenue")
    inner += r("Gross profit", "gross", "GrossProfit", bold=True)
    inner += r("Operating expenses", "opex")
    inner += r("Operating income", "op", "OperatingIncomeLoss", bold=True)
    inner += r("Interest expense, net", "interest", neg=True)
    inner += r("Income before income taxes", "pre")
    inner += r("Provision for income taxes", "tax")
    inner += r("Net income", "net", "NetIncomeLoss", first=True, bold=True)
    tags_b = [("EarningsPerShareBasic", c1, "usdPerShare"), ("EarningsPerShareBasic", c2, "usdPerShare")]
    tags_d = [("EarningsPerShareDiluted", c1, "usdPerShare"), ("EarningsPerShareDiluted", c2, "usdPerShare")]
    inner += row("Basic earnings per share", [cur["eps"], pri["eps"]], dollar=True, indent=12, tags=[(a, b, u) for a, b, u in tags_b])
    inner += row("Diluted earnings per share", [cur["eps_d"], pri["eps_d"]], dollar=True, indent=12, tags=[(a, b, u) for a, b, u in tags_d])
    return table(inner)


def debt_note(t: str, d_cur: dict, d_pri: dict, fy: int, cur_lbl: str, pri_lbl: str, tag_ctx=("i_cur", "i_pri")) -> str:
    rows = header_rows("(in millions)", None, [cur_lbl, pri_lbl], title="Components of long-term debt")
    rows += row(f"Term Loan A due {TLA_DUE} (1)", [d_cur["tla"], d_pri["tla"]], dollar=True, indent=12)
    rows += row(f"4.250% Senior Notes due {SN_DUE} (2)", [d_cur["sn"], d_pri["sn"]], indent=12)
    rows += row(f"Revolving credit facility due {RCF_DUE} (3)", [d_cur["rcf"], d_pri["rcf"]], indent=12)
    rows += row("Total principal amount of debt", [d_cur["principal"], d_pri["principal"]], dollar=True, bold=True)
    rows += row("Less: unamortized debt issuance costs", [d_cur["costs"], d_pri["costs"]], indent=12, neg=True)
    rows += row("Total debt", [d_cur["carrying"], d_pri["carrying"]], bold=True)
    rows += row("Less: current portion of long-term debt", [d_cur["cur"], d_pri["cur"]], indent=12, neg=True)
    rows += row("Long-term debt, net of current portion", [d_cur["lt"], d_pri["lt"]], dollar=True, bold=True)
    return table(rows)


def sched_table(sched: dict, thereafter: int, title: str, hdr: str) -> str:
    inner = f'<tr><td colspan="2" style="font-weight:700;">{title}</td></tr>'
    inner += f'<tr>{td("(in millions)")}{td(hdr, align="center", bold=True)}</tr>'
    for i, (y, v) in enumerate(sched.items()):
        inner += f'<tr>{td(y)}{td(("$ " if i == 0 else "") + n(v), align="right")}</tr>'
    inner += f'<tr>{td("Thereafter")}{td(n(thereafter), align="right")}</tr>'
    inner += f'<tr>{td("Total", bold=True)}{td("$ " + n(sum(sched.values()) + thereafter), align="right", bold=True)}</tr>'
    return table(inner)


def lease_sched_table(L: dict, fy: int) -> str:
    inner = f'<tr><td colspan="7" style="font-weight:700;">Maturities of lease liabilities</td></tr>'
    inner += '<tr>' + td("(in millions)") + td("Operating leases", colspan=3, align="center", bold=True) + td("Finance leases", colspan=3, align="center", bold=True) + "</tr>"
    for i, y in enumerate(L["sched_op"]):
        d = i == 0
        inner += "<tr>" + td(y) + td("$" if d else "") + td(n(L["sched_op"][y]), align="right") + td("") + td("$" if d else "") + td(n(L["sched_fin"][y]), align="right") + td("") + "</tr>"
    inner += "<tr>" + td("Thereafter") + td("") + td(n(L["after_op"]), align="right") + td("") + td("") + td(n(L["after_fin"]), align="right") + td("") + "</tr>"
    inner += "<tr>" + td("Total lease payments", bold=True) + td("$") + td(n(L["tot_op"]), align="right") + td("") + td("$") + td(n(L["tot_fin"]), align="right") + td("") + "</tr>"
    inner += "<tr>" + td("Less: imputed interest") + td("") + td("(" + n(L["imp_op"]), align="right") + td(")") + td("") + td("(" + n(L["imp_fin"]), align="right") + td(")") + "</tr>"
    inner += "<tr>" + td("Present value of lease liabilities", bold=True) + td("$") + td(n(L["pv_op"]), align="right") + td("") + td("$") + td(n(L["pv_fin"]), align="right") + td("") + "</tr>"
    return table(inner)


# ------------------------------------------------------------------------------- filings ---------------
def tenk(t: str, fy: int) -> tuple[str, dict]:
    co = COMPANIES[t]
    fye_m = int(co["fye"][:2])
    end = date(fy, fye_m, 30 if fye_m == 6 else 31)
    prior_end = date(fy - 1, fye_m, 30 if fye_m == 6 else 31)
    mon = "June 30" if fye_m == 6 else "December 31"
    I, Ip = income(t, fy), income(t, fy - 1)
    D, Dp = debt(t, fy), debt(t, fy - 1)
    B, Bp = balance(t, fy), balance(t, fy - 1)
    L = leases(t, fy)
    R = I["rev"]
    g = (R / Ip["rev"] - 1) * 100
    pct = largest_customer_pct(t, fy)
    ocf, capex = round(R * .14), round(R * .055)
    mw = (t == "HLFC" and fy == 2024)
    cur, pri = f"{mon}, {fy}", f"{mon}, {fy - 1}"
    ctx = contexts(end, prior_end, date(fy - 1, prior_end.month, prior_end.day + 1) if False else date(prior_end.year, prior_end.month, prior_end.day), date(fy - 2, fye_m, 30 if fye_m == 6 else 31))
    # (period starts are nominal for the synthetic data)
    parts = [p(f"UNITED STATES SECURITIES AND EXCHANGE COMMISSION — FORM 10-K for the fiscal year ended {mon}, {fy}"),
             p(f"{co['name']}"),
             toc([("1", 3), ("1A", 5), ("7", 12), ("8", 20), ("9A", 40)]), PB,
             '<div><span style="font-weight:700;">PART I</span></div>', h_item("1", "Business")]
    if t == "NWRX":
        parts += [p(f"{co['name']} (the “Company”) designs and manufactures warehouse automation systems, including autonomous mobile robots and fleet-management software used by logistics and e-commerce customers."),
                  p(f"As of {mon}, {fy}, we employed approximately {round(R * 1.9):,} people worldwide and operated {3 + fy - 2022} assembly facilities.")]
    else:
        parts += [p(f"{co['name']} (the “Company”) distributes packaged and refrigerated food products to grocery retailers and sells private-label products under long-term supply agreements."),
                  p(f"As of {mon}, {fy}, we employed approximately {round(R * 2.4):,} people and operated {5 + fy - 2022} distribution centers.")]
    parts += [PB, h_item("1A", "Risk Factors")]
    if t == "NWRX":
        parts += [p(f"We depend on single-source suppliers for key components. Several critical components, including lidar sensors and drive motors, are available only from single-source suppliers. A disruption at any of these suppliers could delay shipments and harm our results. Supply chain disruptions in fiscal {fy} lengthened our average component lead times by {2 + fy - 2022} weeks.")]
    else:
        parts += [p(f"Our products are perishable and spoilage could reduce our margins. Perishable inventory write-offs were ${round(R * .006):,} million in fiscal {fy}, and a failure of our cold-chain logistics could result in larger losses.")]
    cust = "our largest customer" if t == "NWRX" else "our largest customer, a national grocery chain,"
    parts += [p(f"Our revenue is concentrated among a limited number of customers. In fiscal {fy}, {cust} accounted for {pct}% of total revenue, and the loss of that customer would materially reduce our revenue."),
              p(f"We have substantial indebtedness. As of {mon}, {fy}, we had ${D['principal']:,} million of principal indebtedness outstanding, which could limit our flexibility to fund operations and invest in growth."),
              p("Cybersecurity incidents or failures of our information systems could disrupt operations, expose confidential information and damage our reputation."),
              PB, h_item("7", "Management’s Discussion and Analysis of Financial Condition and Results of Operations"),
              p(f"Revenue. Revenue for fiscal {fy} was ${R:,} million, an increase of {g:.1f}% compared with ${Ip['rev']:,} million in fiscal {fy - 1}, driven by higher unit volumes and pricing."),
              p(f"Gross margin. Gross profit was ${I['gross']:,} million, or {I['gross'] / R * 100:.1f}% of revenue, compared with {Ip['gross'] / Ip['rev'] * 100:.1f}% in fiscal {fy - 1}."),
              p(f"Liquidity and capital resources. As of {mon}, {fy}, we had cash and cash equivalents of ${B['cash']:,} million and our revolving credit facility provided total commitments of ${D['rcf_commit']:,} million, of which ${D['rcf']:,} million was drawn. The facility matures in {RCF_DUE}."),
              p(f'Net cash provided by operating activities was ${ocf:,} million. Capital expenditures were <ix:nonFraction name="us-gaap:PaymentsToAcquirePropertyPlantAndEquipment" contextRef="d_cur" unitRef="usd" decimals="-6" scale="6" format="ixt:num-dot-decimal">{capex:,}</ix:nonFraction> million in fiscal {fy}, primarily for manufacturing and distribution capacity.'),
              PB, h_item("8", "Financial Statements and Supplementary Data"),
              bs_table(t, B, Bp, cur, pri), PB,
              is_table(I, Ip, [(f"Year Ended {mon}," if False else f"Year Ended {mon.split(' ')[0]} {mon.split(' ')[1]}", 2)], [str(fy), str(fy - 1)]), PB,
              h_note(1, "Summary of Significant Accounting Policies"),
              p("The consolidated financial statements are prepared in accordance with U.S. generally accepted accounting principles. Revenue is recognized when control of products transfers to the customer."),
              h_note(8, "Debt"),
              p(f"The following table summarizes our long-term debt as of {mon}, {fy} and {fy - 1} (in millions)."),
              debt_note(t, D, Dp, fy, cur, pri),
              p(f"(1) Term Loan A bears interest at SOFR plus 1.75% and is repayable in quarterly installments with the remaining balance due in {TLA_DUE}."),
              p(f"(2) The 4.250% Senior Notes pay interest semi-annually and are unsecured."),
              p(f"(3) Borrowings under the revolving credit facility bear interest at SOFR plus 1.50%."),
              p(f"Scheduled maturities of long-term debt as of {mon}, {fy} are as follows."),
              sched_table(D["sched"], D["thereafter"], "Maturities of long-term debt", "Principal payments"),
              h_note(9, "Leases"),
              p(f"We lease manufacturing, office and distribution facilities under operating leases and certain equipment under finance leases (also referred to as capital leases). Operating lease cost was ${L['op_cost']:,} million in fiscal {fy}. Finance lease cost consisted of amortization of right-of-use assets of ${L['fin_amort']:,} million and interest on lease liabilities of ${L['fin_int']:,} million. Variable lease cost was ${L['variable']:,} million."),
              p(f"Finance lease liabilities of ${L['pv_fin']:,} million are included in other long-term liabilities. Maturities of lease liabilities as of {mon}, {fy} were as follows."),
              lease_sched_table(L, fy), PB,
              h_item("9A", "Controls and Procedures")]
    if mw:
        parts += [p("In connection with the audit of fiscal 2024, management identified a material weakness in internal control over financial reporting related to inventory costing at our Riverbend distribution center. Management is remediating the material weakness by redesigning inventory costing controls.")]
    else:
        parts += [p(f"Management concluded that our internal control over financial reporting was effective as of {mon}, {fy}. We did not identify any material weaknesses during fiscal {fy}.")]
    meta = dict(I=I, Ip=Ip, D=D, Dp=Dp, B=B, Bp=Bp, L=L, pct=pct, g=g, capex=capex, mon=mon, mw=mw, end=end)
    return doc(f"{co['name']} 10-K {fy}", ctx, "".join(parts)), meta


def tenq(t: str, fy: int, q: int) -> tuple[str, dict]:
    co = COMPANIES[t]
    fye_m = int(co["fye"][:2])
    # period end of quarter
    from finrag.core.fiscal import quarter_end
    end = quarter_end(fye_m, fy, q)
    prior_fye = date(fy - 1 if q else fy, fye_m, 30 if fye_m == 6 else 31)
    pm = prior_fye
    R_ann = round(revenue(t, fy) * 1.0)
    rq = round(R_ann / 4 * (1 + 0.02 * q))
    rq_prior = round(revenue(t, fy - 1) / 4 * (1 + 0.02 * q))
    Iq, Ipq = income(t, fy, rq), income(t, fy - 1, rq_prior)
    # six-month (Q2) figures for the 10-Q of a non-calendar company
    I6 = income(t, fy, rq + round(R_ann / 4 * 1.02)) if q == 2 else None
    I6p = income(t, fy - 1, rq_prior + round(revenue(t, fy - 1) / 4 * 1.02)) if q == 2 else None
    Dq, Dpr = debt(t, fy), debt(t, fy - 1)
    Bq = balance(t, fy, key=f"q{q}", R=R_ann, dbt=Dq)
    Bp = balance(t, fy - 1, R=revenue(t, fy - 1), dbt=Dpr)
    cur_lbl, pri_lbl = end.strftime("%B %d, %Y").replace(" 0", " "), pm.strftime("%B %d, %Y").replace(" 0", " ")
    ctx = contexts(end, pm, date(end.year, max(1, end.month - 2), 1), date(pm.year, max(1, pm.month - 2), 1))
    qname = {1: "first", 2: "second", 3: "third"}[q]
    unit_words = f"Three Months Ended {end.strftime('%B')} {end.day},"
    parts = [p(f"FORM 10-Q for the quarterly period ended {cur_lbl} ({qname} quarter of fiscal {fy})"), p(co["name"]),
             '<div><span style="font-weight:700;">PART I — FINANCIAL INFORMATION</span></div>',
             h_item("1", "Financial Statements"),
             bs_table(t, Bq, Bp, cur_lbl, pri_lbl)]
    if I6:
        inner = header_rows("(in millions, except per share data)",
                            [(f"Three Months Ended {end.strftime('%B')} {end.day},", 1), (f"Six Months Ended {end.strftime('%B')} {end.day},", 1)],
                            [str(end.year), str(end.year - 1)], title="Condensed Consolidated Statements of Income")
        # build with both 3m and 6m columns (four value columns)
        def r4(label, key, first=False, bold=False, concept=None):
            tags = [(concept, "d_cur"), (concept, "d_pri"), None, None] if concept else None
            cells = td(label, bold=bold)
            for i, v in enumerate([Iq[key], Ipq[key], I6[key], I6p[key]]):
                tag = tags[i] if tags and tags[i] else None
                cells += vcell(v, first, False, tag)
            return f"<tr>{cells}</tr>"
        inner = ('<tr>' + td("Condensed Consolidated Statements of Income", 13, bold=True) + '</tr>'
                 '<tr>' + td("") + td(f"Three Months Ended {end.strftime('%B')} {end.day},", 6, align="center", bold=True) + td(f"Six Months Ended {end.strftime('%B')} {end.day},", 6, align="center", bold=True) + '</tr>'
                 '<tr>' + td("(in millions, except per share data)") + "".join(td(y, 3, align="center", bold=True) for y in [str(end.year), str(end.year - 1), str(end.year), str(end.year - 1)]) + '</tr>')
        inner += r4("Revenue", "rev", True, concept="Revenues") + r4("Cost of revenue", "cogs") + r4("Gross profit", "gross", bold=True) + r4("Operating expenses", "opex") + r4("Operating income", "op", bold=True) + r4("Net income", "net", True, True)
        parts += [table(inner)]
    else:
        parts += [is_table(Iq, Ipq, [(unit_words, 2)], [str(end.year), str(end.year - 1)])]
    parts += [h_note(5, "Debt"),
              p(f"Long-term debt consisted of the following as of {cur_lbl} and {pri_lbl} (in millions)."),
              debt_note(t, Dq, Dpr, fy, cur_lbl, pri_lbl),
              p(f"(1) Term Loan A bears interest at SOFR plus 1.75% and matures in {TLA_DUE}."),
              h_item("2", "Management’s Discussion and Analysis of Financial Condition and Results of Operations"),
              p(f"Revenue. Revenue for the three months ended {cur_lbl} was ${Iq['rev']:,} million, compared with ${Ipq['rev']:,} million for the same period last year, an increase of {(Iq['rev'] / Ipq['rev'] - 1) * 100:.1f}%.")]
    if I6:
        parts += [p(f"Revenue for the six months ended {cur_lbl} was ${I6['rev']:,} million.")]
    parts += ['<div><span style="font-weight:700;">PART II — OTHER INFORMATION</span></div>', h_item("1A", "Risk Factors"),
              p("There have been no material changes to the risk factors disclosed in our most recent Annual Report on Form 10-K.")]
    meta = dict(Iq=Iq, Ipq=Ipq, I6=I6, Dq=Dq, Bq=Bq, end=end, cur_lbl=cur_lbl)
    return doc(f"{co['name']} 10-Q Q{q} FY{fy}", ctx, "".join(parts)), meta


def eightk(t: str, fy: int, q: int) -> tuple[str, dict]:
    co = COMPANIES[t]
    rq = round(revenue(t, fy) / 4 * (1 + 0.02 * q))
    rp = round(revenue(t, fy - 1) / 4 * (1 + 0.02 * q))
    g = (rq / rp - 1) * 100
    gm = income(t, fy, rq)["gross"] / rq * 100
    body = [p(f"FORM 8-K — CURRENT REPORT. {co['name']}"),
            h_item("2.02", "Results of Operations and Financial Condition"),
            p(f"On April 24, 2025, {co['name']} issued a press release announcing results for the quarter ended March 31, 2025."),
            p(f"First quarter revenue was ${rq:,} million, up {g:.1f}% from the first quarter of 2024. Gross margin was {gm:.1f}%."),
            p("The press release is furnished as Exhibit 99.1 and is not deemed filed for purposes of Section 18 of the Securities Exchange Act."),
            h_item("9.01", "Financial Statements and Exhibits"), p("Exhibit 99.1 — Press release dated April 24, 2025.")]
    return doc(f"{co['name']} 8-K", "", "".join(body)), dict(rq=rq, g=g, gm=gm)


# ------------------------------------------------------------------------------- build + gold ---------------
def main() -> None:
    FILINGS.mkdir(parents=True, exist_ok=True)
    manifest, gold = [], []
    metas: dict[tuple, dict] = {}

    def add(t, form, fy, fp, period_end, acc, html, meta, filed):
        co = COMPANIES[t]
        name = f"{t}_{form}_{period_end}.htm"
        (FILINGS / name).write_text(html, encoding="utf-8")
        manifest.append(dict(company=co["name"], ticker=t, filing_type=form, period_end=str(period_end), accession=acc,
                             cik=co["cik"], fiscal_year=fy, fiscal_period=fp, fiscal_year_end=co["fye"], filing_date=filed,
                             path=f"filings/{name}", url=""))
        metas[(t, form, fy, fp)] = meta

    for t, fys in (("NWRX", (2023, 2024)), ("HLFC", (2023, 2024))):
        for fy in fys:
            html, m = tenk(t, fy)
            add(t, "10-K", fy, "FY", m["end"], f"SYN-{t}-10K-{fy}", html, m, f"{fy + 1}-02-20" if t == "NWRX" else f"{fy}-08-25")
    html, m = tenq("NWRX", 2025, 1)
    add("NWRX", "10-Q", 2025, "Q1", m["end"], "SYN-NWRX-10Q-2025Q1", html, m, "2025-04-24")
    html, m = tenq("HLFC", 2025, 2)
    add("HLFC", "10-Q", 2025, "Q2", m["end"], "SYN-HLFC-10Q-2025Q2", html, m, "2025-02-06")
    html, m = eightk("NWRX", 2025, 1)
    add("NWRX", "8-K", None, "", date(2025, 4, 24), "SYN-NWRX-8K-20250424", html, m, "2025-04-24")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ---------------- gold questions (answers computed from the same numbers) ----------------
    def q(qid, question, answer, evidence, filing, kind, note=""):
        gold.append(dict(id=qid, question=question, answer=answer, evidence=evidence, gold_filing=filing, kind=kind, note=note))

    K = lambda t, fy: metas[(t, "10-K", fy, "FY")]
    nw24, nw23, hl24, hl23 = K("NWRX", 2024), K("NWRX", 2023), K("HLFC", 2024), K("HLFC", 2023)
    F = lambda t, form, fy, fp="": dict(ticker=t, filing_type=form, fiscal_year=fy)

    q("q001", "What were Northwind Robotics' total assets at the end of fiscal 2024?", f"${nw24['B']['ta']:,} million", [f"{nw24['B']['ta']:,}"], F("NWRX", "10-K", 2024), "table")
    q("q002", "How much cash and cash equivalents did NWRX hold as of December 31, 2024?", f"${nw24['B']['cash']:,} million", [f"{nw24['B']['cash']:,}"], F("NWRX", "10-K", 2024), "table")
    q("q003", "What were Northwind Robotics' total liabilities in fiscal 2023?", f"${nw23['B']['tl']:,} million", [f"{nw23['B']['tl']:,}"], F("NWRX", "10-K", 2023), "table")
    q("q004", "What was Harbor Lane Foods' total stockholders' equity at June 30, 2024?", f"${hl24['B']['eq']:,} million", [f"{hl24['B']['eq']:,}"], F("HLFC", "10-K", 2024), "table")
    q("q005", "What was Harbor Lane Foods' long-term debt, net of current portion, in fiscal 2024?", f"${hl24['B']['lt_debt']:,} million", [f"{hl24['B']['lt_debt']:,}"], F("HLFC", "10-K", 2024), "table")
    q("q006", "How much debt principal does Northwind Robotics have maturing in 2027?", f"${nw24['D']['sched']['2027']:,} million", ["Maturities of long-term debt", f"{nw24['D']['sched']['2027']:,}"], F("NWRX", "10-K", 2024), "table")
    q("q007", "What is the principal due thereafter in Northwind's fiscal 2024 debt maturity schedule?", f"${nw24['D']['thereafter']:,} million", ["Thereafter", f"{nw24['D']['thereafter']:,}"], F("NWRX", "10-K", 2024), "table")
    q("q008", "What principal payments does Harbor Lane Foods owe in 2026 per its fiscal 2024 debt note?", f"${hl24['D']['sched']['2026']:,} million", ["Maturities of long-term debt", f"{hl24['D']['sched']['2026']:,}"], F("HLFC", "10-K", 2024), "table")
    q("q009", "What is the total principal amount of debt outstanding for NWRX at the end of fiscal 2024?", f"${nw24['D']['principal']:,} million", ["Total principal amount of debt", f"{nw24['D']['principal']:,}"], F("NWRX", "10-K", 2024), "table")
    q("q010", "What interest rate do Northwind's Senior Notes carry?", "4.250%", ["4.250% Senior Notes"], F("NWRX", "10-K", 2024), "table")
    q("q011", "What are NWRX's total operating lease payments in the fiscal 2024 lease maturity table?", f"${nw24['L']['tot_op']:,} million", ["Total lease payments", f"{nw24['L']['tot_op']:,}"], F("NWRX", "10-K", 2024), "table")
    q("q012", "How much imputed interest is deducted from NWRX's operating lease payments in fiscal 2024?", f"${nw24['L']['imp_op']:,} million", ["imputed interest", f"{nw24['L']['imp_op']:,}"], F("NWRX", "10-K", 2024), "table")
    q("q013", "What was Harbor Lane Foods' operating lease cost in fiscal 2024?", f"${hl24['L']['op_cost']:,} million", [f"Operating lease cost was ${hl24['L']['op_cost']:,} million"], F("HLFC", "10-K", 2024), "text")
    q("q014", "What are Northwind's finance (capital) lease liabilities in fiscal 2024?", f"${nw24['L']['pv_fin']:,} million", [f"Finance lease liabilities of ${nw24['L']['pv_fin']:,} million"], F("NWRX", "10-K", 2024), "text")
    q("q015", "What is the present value of Harbor Lane's operating lease liabilities for fiscal 2024?", f"${hl24['L']['pv_op']:,} million", ["Present value of lease liabilities", f"{hl24['L']['pv_op']:,}"], F("HLFC", "10-K", 2024), "table")
    q("q016", "What was Northwind Robotics' revenue in fiscal 2024?", f"${nw24['I']['rev']:,} million", [f"{nw24['I']['rev']:,}"], F("NWRX", "10-K", 2024), "table")
    q("q017", "What was Northwind's net income in fiscal 2023?", f"${nw23['I']['net']:,} million", [f"{nw23['I']['net']:,}"], F("NWRX", "10-K", 2023), "table")
    q("q018", "What was Harbor Lane Foods' diluted EPS in fiscal 2024?", f"${hl24['I']['eps_d']:.2f}", [f"{hl24['I']['eps_d']:.2f}"], F("HLFC", "10-K", 2024), "table")
    q("q019", "What was Harbor Lane Foods' operating income in fiscal 2024?", f"${hl24['I']['op']:,} million", [f"{hl24['I']['op']:,}"], F("HLFC", "10-K", 2024), "table")
    hq = metas[("HLFC", "10-Q", 2025, "Q2")]
    q("q020", "What was Harbor Lane Foods' revenue in Q2 of fiscal 2025?", f"${hq['Iq']['rev']:,} million", [f"{hq['Iq']['rev']:,}"], F("HLFC", "10-Q", 2025), "period", "fiscal Q2 FY2025 = quarter ended Dec 31, 2024 (June FYE)")
    q("q021", "What was Harbor Lane Foods' revenue for the six months ended December 31, 2024?", f"${hq['I6']['rev']:,} million", [f"{hq['I6']['rev']:,}"], F("HLFC", "10-Q", 2025), "period")
    nq = metas[("NWRX", "10-Q", 2025, "Q1")]
    q("q022", "What was Northwind's Q1 2025 revenue?", f"${nq['Iq']['rev']:,} million", [f"{nq['Iq']['rev']:,}"], F("NWRX", "10-Q", 2025), "period")
    q("q023", "What was NWRX's total principal debt at March 31, 2025?", f"${nq['Dq']['principal']:,} million", ["Total principal amount of debt", f"{nq['Dq']['principal']:,}"], F("NWRX", "10-Q", 2025), "table")
    ek = metas[("NWRX", "8-K", None, "")]
    q("q024", "What first quarter revenue did Northwind announce in its press release?", f"${ek['rq']:,} million", [f"{ek['rq']:,}"], F("NWRX", "8-K", None), "text")
    q("q025", "By what percentage did Northwind's revenue grow in fiscal 2024?", f"{nw24['g']:.1f}%", [f"increase of {nw24['g']:.1f}%"], F("NWRX", "10-K", 2024), "text")
    q("q026", "What percent of Northwind's revenue came from its largest customer in fiscal 2024?", f"{nw24['pct']}%", [f"{nw24['pct']}%"], F("NWRX", "10-K", 2024), "text")
    q("q027", "What supply chain risks did Northwind disclose?", "Single-source suppliers for lidar sensors and drive motors", ["single-source suppliers"], F("NWRX", "10-K", 2024), "text")
    q("q028", "What material weakness did Harbor Lane Foods identify in fiscal 2024?", "Inventory costing at the Riverbend distribution center", ["material weakness", "Riverbend"], F("HLFC", "10-K", 2024), "text")
    q("q029", "Did Harbor Lane Foods report any material weaknesses in fiscal 2023?", "No, it did not identify any material weaknesses", ["did not identify any material weaknesses"], F("HLFC", "10-K", 2023), "text")
    q("q030", "How large is Northwind's revolving credit facility?", f"${nw24['D']['rcf_commit']:,} million", [f"total commitments of ${nw24['D']['rcf_commit']:,} million"], F("NWRX", "10-K", 2024), "text")
    q("q031", "What are the perishable inventory risks for Harbor Lane Foods?", "Spoilage and cold-chain failures could reduce margins", ["perishable"], F("HLFC", "10-K", 2024), "text")
    q("q032", "What interest rate applies to Northwind's Term Loan A?", "SOFR plus 1.75%", ["SOFR plus 1.75%"], F("NWRX", "10-K", 2024), "footnote")
    q("q033", "What was NWRX capex in fiscal 2024?", f"${nw24['capex']:,} million", [f"{nw24['capex']:,}"], F("NWRX", "10-K", 2024), "xbrl", "abbreviation 'capex' -> capital expenditures")
    q("q034", "How much did Northwind spend on capital expenditures in fiscal 2023?", f"${nw23['capex']:,} million", [f"{nw23['capex']:,}"], F("NWRX", "10-K", 2023), "xbrl")

    q("q035", "What was Northwind Robotics' revenue in fiscal 2019?", None, [], F("NWRX", "10-K", 2019), "unanswerable", "no FY2019 filing in corpus")
    q("q036", "Who is the chief executive officer of Harbor Lane Foods?", None, [], F("HLFC", "10-K", 2024), "unanswerable", "not stated in the synthetic filings")
    q("q037", "What was Apple's total assets in fiscal 2024?", None, [], F("AAPL", "10-K", 2024), "unanswerable", "company not in corpus")
    q("q038", "What dividend per share did Northwind declare in fiscal 2024?", None, [], F("NWRX", "10-K", 2024), "unanswerable", "dividends not disclosed")

    with open(OUT / "qa_gold.jsonl", "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g) + "\n")
    print(f"wrote {len(manifest)} filings, {len(gold)} gold questions to {OUT}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(OUT.parent / "src"))
    main()
