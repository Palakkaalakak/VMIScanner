"""Build the 3-tier moat-reasoning training dataset.

TIER A — gold      : Adam Khoo's own verdicts (adam_seed_labels.json) fused with the
                     scanner's quantitative evidence card. ~38 examples, highest weight.
TIER B — silver    : rubric-only prompts for every other scanned ticker. A teacher
                     model (local large model or API) answers them using ONLY the
                     rubric system prompt — it never sees consensus moat ratings.
                     Human spot-review ~10% before training.
TIER C — contrastive: corrupted-evidence variants of gold tickers. Same company name,
                     degraded fundamentals → the correct answer DOWNGRADES the verdict
                     citing the evidence. This is what makes it a reasoning model
                     rather than a ticker-memorizer.

Output (chat-messages JSONL, axolotl/unsloth/llama-factory ready):
  ai_moat/dataset/gold.jsonl
  ai_moat/dataset/contrastive.jsonl
  ai_moat/dataset/silver_prompts.jsonl   (prompts only — teacher fills answers)

Usage:  python3 -m ai_moat.build_dataset
"""
from __future__ import annotations

import json
import os
from copy import deepcopy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCAN = os.path.join(ROOT, "public", "data", "scan_results.json")
LABELS = os.path.join(HERE, "adam_seed_labels.json")
SYSPROMPT = os.path.join(HERE, "rubric_system_prompt.md")
OUTDIR = os.path.join(HERE, "dataset")

VERDICT_MAP = {
    "wide": ("WIDE", 9),
    "wide_weak": ("WIDE", 7),
    "narrow": ("NARROW", 5),
    "none": ("NO MOAT", 2),
    "none_or_narrow": ("NO MOAT", 3),
    "lost_moat": ("NARROW", 4),
}

ACTION_MAP = {
    "WIDE": "invest-grade",
    "NARROW": "buyable-with-caution",
    "NO MOAT": "avoid at all costs",
}


def load_scan() -> dict:
    """Return {ticker: row} from scan_results.json (results is a list of rows)."""
    with open(SCAN) as f:
        d = json.load(f)
    rows = d["results"] if isinstance(d, dict) else d
    out = {}
    for r in rows:
        if isinstance(r, dict) and r.get("ticker"):
            out[r["ticker"]] = r
    return out


def _fmt(v, suffix=""):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.1f}{suffix}"
    return f"{v}{suffix}"


def evidence_block(row: dict, ticker: str) -> str:
    """Render the scanner's evidence card as plain text for the prompt."""
    m = row.get("metrics") or {}
    h = row.get("moat_hints") or {}
    lines = [
        f"TICKER: {ticker}",
        f"COMPANY: {row.get('company', 'n/a')}",
        f"SECTOR / INDUSTRY: {row.get('sector', 'n/a')} / {row.get('industry', 'n/a')}",
        "",
        "QUANTITATIVE EVIDENCE CARD (10-year, scanner-derived):",
        f"- moat evidence score (0-100 composite): {_fmt(m.get('moat_evidence_score'))}",
        f"- structurally no-moat industry flag: {m.get('moat_no_moat_industry', 'n/a')}",
        f"- gross-margin trend (3y avg vs decade start): {_fmt(m.get('moat_gm_trend_pp'), 'pp')}",
        f"- operating-margin trend (newest-3y vs oldest-3y): {_fmt(m.get('moat_om_trend_pp'), 'pp')}",
        f"- TTM ROIC: {_fmt(m.get('ttm_roic'), '%')}",
        f"- TTM FCF yield: {_fmt(m.get('ttm_fcf_yield'), '%')}",
        f"- PEG (Adam method, PE / avg growth): {_fmt(m.get('peg_adam'))}",
        f"- PSG (P/S / projected rev growth): {_fmt(m.get('psg'))}",
        "",
        "EVIDENCE NOTES:",
    ]
    for k in ("pricing_power", "roic_persistence", "margin_durability",
              "growth_consistency", "self_financing", "industry_warning"):
        if h.get(k):
            lines.append(f"- {k}: {h[k]}")
    return "\n".join(lines)


def gold_answer(lab: dict) -> str:
    """Render Adam's verdict in the mandated output format."""
    grade, default10 = VERDICT_MAP[lab["verdict"]]
    score10 = lab.get("score10") or default10
    ss = lab.get("source_scores") or {}
    names = [
        ("brand_pricing", "Brand monopoly / pricing power"),
        ("switching", "Switching costs"),
        ("network", "Network effect"),
        ("barriers", "Barriers to entry"),
        ("scale", "Economies of scale"),
    ]
    src_lines, passing = [], 0
    for i, (key, label) in enumerate(names, 1):
        v = ss.get(key)
        if v is None:
            src_lines.append(f"  {i}. {label}: not separately scored by Adam — see reasoning")
        else:
            src_lines.append(f"  {i}. {label}: {v}/10")
            if v >= 6:
                passing += 1
    scored_any = any(ss.get(k) is not None for k, _ in names)
    if scored_any:
        rule_line = (f"SOURCES PASSING (>=6/10): {passing}/5 -> "
                     f"{'meets' if passing >= 3 else 'fails'} the >=3-of-5 rule")
    else:
        rule_line = ("SOURCES PASSING (>=6/10): per Adam's stated verdict "
                     f"({'meets' if grade == 'WIDE' else 'does not clearly meet'} the >=3-of-5 rule)")
    keyman = "flagged" if lab["ticker"] == "TSLA" else "none identified"
    decay = "decaying — this is a canonical lost-moat case" if lab["verdict"] == "lost_moat" else "stable per Adam's assessment at the time"
    industry = "flagged: structurally no-moat industry" if lab["ticker"] in ("TSLA", "CVX", "XOM", "SHEL", "SLB", "AA", "DOW") else "clean"
    return "\n".join([
        f"MOAT VERDICT: {grade} — {score10}/10",
        "SOURCES (each /10):",
        *src_lines,
        rule_line,
        f"PRICING-POWER TEST: {'fail — ' + lab['adam_reasoning'][:120] if lab['ticker']=='TSLA' else ('pass' if grade=='WIDE' else 'weak / fail')}",
        f"INDUSTRY SCREEN: {industry}",
        f"KEY-MAN RISK: {keyman}",
        f"DECAY CHECK: {decay}",
        f"REASONING: {lab['adam_reasoning']}",
        f"ACTION (Adam's framework): {ACTION_MAP[grade]}",
    ])


CORRUPT_NOTE = (
    "\n\n[NOTE: the evidence card above shows serious fundamental deterioration — "
    "eroding gross margins, collapsing operating margins, sub-par ROIC. Evaluate the "
    "moat on THIS evidence, not on the company's reputation.]"
)

CONTRASTIVE_ANSWER_TMPL = """MOAT VERDICT: NARROW — 4/10
SOURCES (each /10):
  1. Brand monopoly / pricing power: 4/10 — the brand may persist, but a gross-margin erosion of ~12pp against the decade start means the pricing-power test is failing in the numbers
  2. Switching costs: 5/10 — whatever lock-in existed is not showing up as durable margins
  3. Network effect: 4/10 — a real network effect defends margins; these margins are not being defended
  4. Barriers to entry: 4/10 — competitors are visibly getting through: operating margin down ~9pp newest-3y vs oldest-3y
  5. Economies of scale: 4/10 — scale without margin durability is just size
SOURCES PASSING (>=6/10): 0/5 -> fails the >=3-of-5 rule
PRICING-POWER TEST: fail — sustained margin erosion is the financial fingerprint of a company that cannot raise prices without losing share
INDUSTRY SCREEN: clean
KEY-MAN RISK: none identified
DECAY CHECK: decaying — this is the Intel/GE pattern: a formerly strong franchise whose moat is being competed away; ROIC of ~6% is below the 15% persistence bar
REASONING: The reputation says wide moat, but the evidence card says otherwise, and the evidence wins. A genuine wide moat leaves tracks — high stable gross margins, ROIC persistently above 15%, durable operating margins. This card shows the opposite on every count. Moats are not permanent (Intel, GE), and grading on past fame instead of present evidence is exactly the mistake the decay check exists to prevent. At best this is a narrow moat in decline that must be re-verified each year.
ACTION (Adam's framework): buyable-with-caution at most; treat as avoid until the margin trend stabilizes"""


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    with open(SYSPROMPT) as f:
        system = f.read()
    with open(LABELS) as f:
        labels = json.load(f)["labels"]
    scan = load_scan()

    ask = ("Evaluate this company's economic moat using the rubric. "
           "Use the evidence card; follow the output format exactly.\n\n")

    # ---- TIER A: gold ----
    gold, missing = [], []
    for lab in labels:
        t = lab["ticker"]
        row = scan.get(t)
        if row is None:
            # still trainable: evidence card replaced by a stub
            ev = (f"TICKER: {t}\n(No scanner evidence card available — "
                  f"reason qualitatively from the rubric.)")
            missing.append(t)
        else:
            ev = evidence_block(row, t)
        gold.append({"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": ask + ev},
            {"role": "assistant", "content": gold_answer(lab)},
        ], "tier": "gold", "ticker": t})

    # ---- TIER C: contrastive (corrupted gold WIDE tickers) ----
    contrastive = []
    for lab in labels:
        if VERDICT_MAP[lab["verdict"]][0] != "WIDE":
            continue
        t = lab["ticker"]
        row = scan.get(t)
        if row is None:
            continue
        bad = deepcopy(row)
        m = bad.setdefault("metrics", {})
        m["moat_gm_trend_pp"] = -12.0
        m["moat_om_trend_pp"] = -9.0
        m["ttm_roic"] = 6.0
        m["moat_evidence_score"] = 25.0
        h = bad.setdefault("moat_hints", {})
        h["pricing_power"] = ("gross margin eroded ~12pp vs decade start — the "
                              "financial fingerprint of LOST pricing power")
        h["roic_persistence"] = "ROIC ~6%, far below the 15% persistence bar"
        h["margin_durability"] = "operating margin down ~9pp newest-3y vs oldest-3y"
        ev = evidence_block(bad, t) + CORRUPT_NOTE
        contrastive.append({"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": ask + ev},
            {"role": "assistant", "content": CONTRASTIVE_ANSWER_TMPL},
        ], "tier": "contrastive", "ticker": t})

    # ---- TIER B: silver prompts (no answers — teacher model fills them) ----
    gold_ticks = {lab["ticker"] for lab in labels}
    silver = []
    for t, row in sorted(scan.items()):
        if t in gold_ticks:
            continue
        ev = evidence_block(row, t)
        silver.append({"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": ask + ev},
        ], "tier": "silver_prompt", "ticker": t})

    def dump(name, items):
        p = os.path.join(OUTDIR, name)
        with open(p, "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        return p

    dump("gold.jsonl", gold)
    dump("contrastive.jsonl", contrastive)
    dump("silver_prompts.jsonl", silver)

    print(f"gold: {len(gold)}  (no evidence card for: {missing or 'none'})")
    print(f"contrastive: {len(contrastive)}")
    print(f"silver prompts: {len(silver)}")
    print(f"written to {OUTDIR}/")


if __name__ == "__main__":
    main()
