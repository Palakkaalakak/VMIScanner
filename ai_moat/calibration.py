"""Deterministic moat-score calibration — shared by the dashboard and training.

Why this exists (user report 2026-08-19): the rubric's prose tier-discipline
("SCORE CALIBRATION WITHIN WIDE") was added on 2026-08-18 and the model kept
printing 9/10 for both AAPL and ABBV anyway — including for ABBV answers that
THEMSELVES said "DECAY CHECK: decaying". Prose instructions alone don't bind
an 8B model. So the arithmetic is now enforced in three places:

  1. HERE, in code, applied by the dashboard to every parsed answer —
     the model can print whatever it wants, the SAVED score obeys the rules.
  2. In the rubric text (so the model's printed number usually agrees).
  3. In a quick top-up retrain (teach_calibration.py) whose training answers
     are themselves transformed by this same function — the model literally
     learns the arithmetic it will be graded by.

THE RULES (scoring discipline derived from the rubric / Adam's framework —
no invented financial numbers):

  A. ROUND DOWN: the overall score may never exceed the FLOOR of the
     average of the five per-source scores. (9,8,7,9,8 → avg 8.2 → max 8.)
  B. REDUNDANCY CAP (the 8-vs-10 difference IS redundancy of sources):
       5 sources ≥8  → cap 10
       4 sources ≥8  → cap 9
       3 sources ≥8  → cap 9 only if all three are ≥9 ("3 exceptional"),
                        else cap 8
       ≤2 sources ≥8 → cap 8  (single/dual-source moat — the pharma-patent
                        case: Adam — "it doesn't last forever")
  C. DECAY PENALTY −1: if the answer's own DECAY CHECK says decaying/eroding,
     OR the scanner measured operating margin ≤ −3pp vs decade start (the
     same Intel-style threshold checks.py uses for its ERODING warning),
     subtract 1. Consistency of growing profits/FCF is what separates a
     9-10 from an 8 — Mastercard is the golden example (ROIC ≥15% in 10/10
     years, 85% revenue/income up-years, +7pp margin trend, FCF+ 10/10).
  D. BENCHMARK ANCHOR (user requirement 2026-08-19): a score is never given
     in isolation — 9/10 means "belongs on the same shelf as AAPL/MA/MSFT".
     The BENCHMARKS block below is injected into every prompt so the model
     compares the company against the Heavenly-Queen bar instead of grading
     each stock against nothing.
"""
import re
from typing import List, Optional, Tuple

# Reference bar for 9-10/10, from adam_seed_labels.json heavenly_queens +
# the scanner's own measured evidence (scan_results.json 2026-08). These
# lines state MEASURED facts, not invented ones.
BENCHMARK_BLOCK = """\
BENCHMARKS — the 9-10/10 shelf (compare against these, not against nothing):
- MA (Mastercard, the golden example): ROIC ≥15% in 10/10 years, revenue/income
  up-years 85%, operating margin 57.6% and +7.0pp vs decade start, FCF positive
  10/10 years. Network effect + switching costs + scale, each independently strong.
- AAPL: brand + ecosystem lock-in + App Store network + scale — kill any single
  source and the moat survives. ROIC ≥15% in 10/10 years, margins stable.
- MSFT: switching costs (enterprise lock-in) + network + scale + brand, operating
  margin +13.7pp vs decade start.
A 9/10 claim means: "this moat is as redundant and as consistently PROVEN in the
numbers as those three." If the company leans on one expiring source, or its
margins are eroding while the benchmarks' expand, it is NOT on that shelf — 8 max."""

# "  1. Brand monopoly / pricing power: 8/10 — ..." (numbered source lines)
SRC_SCORE_RE = re.compile(r"^\s*[1-5][.)][^:\n]{0,120}:\s*(\d{1,2})\s*/\s*10",
                          re.M)
# Match only when the VERDICT WORD itself is decaying/eroding — a substring
# match would false-positive on "stable — no significant decay".
DECAY_RE = re.compile(r"DECAY CHECK:\s*\W*(decay(?:ing|ed)?|erod(?:ing|ed))\b",
                      re.I)


def enforce_calibration(answer: str, score: Optional[int],
                        scan_row: Optional[dict] = None
                        ) -> Tuple[Optional[int], List[str]]:
    """Return (enforced_score, notes). notes explain every adjustment;
    empty notes == the model's own number already obeyed the rules."""
    if score is None:
        return None, []
    notes: List[str] = []
    s = int(score)

    srcs = [int(x) for x in SRC_SCORE_RE.findall(answer or "")][:5]
    if len(srcs) >= 3:
        floor_avg = int(sum(srcs) / len(srcs))  # floor (scores are ≥0)
        if s > floor_avg:
            notes.append(f"rounded DOWN to the source average: {s} → "
                         f"{floor_avg} (sources {srcs}, avg "
                         f"{sum(srcs)/len(srcs):.1f})")
            s = floor_avg
        strong = [x for x in srcs if x >= 8]
        if len(strong) >= 5:
            cap = 10
        elif len(strong) == 4:
            cap = 9
        elif len(strong) == 3:
            cap = 9 if min(strong) >= 9 else 8
        else:
            cap = 8
        if s > cap:
            notes.append(f"redundancy cap: only {len(strong)}/5 sources "
                         f"≥8/10 → capped {s} → {cap} (10/10 needs the moat "
                         "to survive losing any single source)")
            s = cap

    decayed = bool(DECAY_RE.search(answer or ""))
    why = "the answer's own DECAY CHECK says decaying"
    if not decayed and scan_row:
        om_pp = (scan_row.get("metrics") or {}).get("moat_om_trend_pp")
        if om_pp is not None and om_pp <= -3:
            decayed = True
            why = (f"scanner evidence: operating margin {om_pp:+.1f}pp vs "
                   "decade start — eroding (Intel-style decay)")
    if decayed and s > 1:
        notes.append(f"decay penalty −1: {s} → {s - 1} ({why}; a decaying "
                     "moat cannot hold a 9-10 — consistency of growing "
                     "profits is the 9-10 qualifier, MA is the golden "
                     "example)")
        s -= 1
    return s, notes


def rewrite_verdict_score(answer: str, new_score: int) -> str:
    """Rewrite the '— N/10' in the MOAT VERDICT line (training transform)."""
    return re.sub(r"(MOAT VERDICT:[^\n]*?—\s*)\d+(\s*/\s*10)",
                  lambda m: f"{m.group(1)}{new_score}{m.group(2)}",
                  answer, count=1)
