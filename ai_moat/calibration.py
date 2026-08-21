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
     EXCEPTION (user report 2026-08-20 — AAPL was being underestimated):
     when ALL FIVE sources are ≥8 AND there is no decay anywhere (answer
     verdict + scanner margin trend both clean), the average is rounded to
     the NEAREST integer instead of floored (9,8,9,8,9 → avg 8.6 → 9).
     A fully-redundant, non-decaying moat is exactly what the 9-10 shelf
     means — the floor discipline exists to stop saturation, not to pull
     true Heavenly-Queens off their own shelf.
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
margins are eroding while the benchmarks' expand, it is NOT on that shelf — 8 max.
IMPORTANT: these benchmarks are a MEASURING STICK for the score, not competitors.
Never call MA/AAPL/MSFT a "competitor" of the company you are grading unless they
actually compete in its industry (a pharma company does not compete with Apple).
Name actual industry rivals when you compare; use the benchmarks only to judge
whether the moat's REDUNDANCY and CONSISTENCY earn a 9-10."""

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
    empty notes == the model's own number already obeyed the rules.

    v3 (user report 2026-08-21): the model computed the CORRECT number in
    its own SCORE ARITHMETIC line ("final 7") and then printed a DIFFERENT
    one in the verdict ("— 6/10") — a copy failure. The old clamp-only
    logic could only push the printed number DOWN, so it punished the copy
    error twice (6 → 5 via the decay penalty) instead of restoring 7. Now
    the score is RECOMPUTED forward from the source scores the model
    actually wrote (same compute_arithmetic used by training), in either
    direction — when sources parse, the printed verdict number is ignored
    entirely."""
    # Decay is decided first because it feeds the arithmetic.
    decayed = bool(DECAY_RE.search(answer or ""))
    why = "the answer's own DECAY CHECK says decaying"
    if not decayed and scan_row:
        om_pp = (scan_row.get("metrics") or {}).get("moat_om_trend_pp")
        if om_pp is not None and om_pp <= -3:
            decayed = True
            why = (f"scanner evidence: operating margin {om_pp:+.1f}pp vs "
                   "decade start — eroding (Intel-style decay)")

    srcs = [int(x) for x in SRC_SCORE_RE.findall(answer or "")][:5]
    if len(srcs) >= 3:
        s, work = compute_arithmetic(srcs, decayed)
        notes: List[str] = []
        if decayed:
            work += f" [decay evidence: {why}]"
        if score is None:
            notes.append(f"score computed from the source scores: {work}")
        elif int(score) != s:
            direction = "UP" if s > int(score) else "DOWN"
            notes.append(f"verdict number corrected {direction}: the model "
                         f"printed {score}, but its own source scores imply "
                         f"{s} ({work})")
        return s, notes

    # No parsable source scores — fall back to clamping the printed number.
    if score is None:
        return None, []
    s = int(score)
    notes = []
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


def compute_arithmetic(srcs: List[int], decayed: bool) -> Tuple[int, str]:
    """PURE scoring function: the deterministic score the 5 source scores
    and the decay flag imply, plus the shown work. This is the number the
    model SHOULD print — same rules as enforce_calibration, but computed
    forward from the sources instead of clamping a model guess."""
    avg = sum(srcs) / len(srcs)
    strong = [x for x in srcs if x >= 8]
    all5 = len(srcs) == 5 and min(srcs) >= 8
    if all5 and not decayed:
        base = int(avg + 0.5)
        how = "all 5 sources ≥8 and no decay → round to NEAREST"
    else:
        base = int(avg)
        how = "round DOWN (floor)"
    if len(strong) >= 5:
        cap = 10
    elif len(strong) == 4:
        cap = 9
    elif len(strong) == 3:
        cap = 9 if min(strong) >= 9 else 8
    else:
        cap = 8
    s = min(base, cap)
    parts = [f"sources {srcs} → average {avg:.1f} → {how} → {base}",
             f"redundancy: {len(strong)}/5 sources ≥8 → cap {cap}"]
    if decayed:
        s = max(1, s - 1)
        parts.append(f"decay: eroding → −1 → {s}")
    else:
        parts.append("decay: none")
    parts.append(f"final {s}/10")
    return s, "; ".join(parts)


_VERDICT_LINE_RE = re.compile(r"^MOAT VERDICT:\s*([A-Z][A-Z /-]*?)\s*[—–-]"
                              r"\s*(\d+)\s*/\s*10.*$", re.M)
_ACTION_LINE_RE = re.compile(r"^ACTION\b", re.M)


def restructure_answer(answer: str, scan_row: Optional[dict] = None
                       ) -> Tuple[str, Optional[int]]:
    """THE ROOT-CAUSE FIX (user report 2026-08-20): the trained output
    format put 'MOAT VERDICT: … — N/10' on LINE 1, so an autoregressive
    model had to commit to the score BEFORE generating the five source
    scores — it cannot average numbers it has not written yet. No retrain
    volume fixes that ordering.

    This transform rebuilds any answer into the arithmetic-first format:
      sources / tests / reasoning …
      SCORE ARITHMETIC: <computed work, from the sources actually written>
      MOAT VERDICT: <word> — <computed score>/10
      ACTION …
    so the score token is generated AFTER the arithmetic that determines
    it. Used by teach_calibration + build_dataset so the model learns to
    write in this order. Returns (new_answer, final_score); if the answer
    has no parsable verdict line it is returned unchanged with None."""
    m = _VERDICT_LINE_RE.search(answer or "")
    if not m:
        return answer, None
    word, printed = m.group(1).strip(), int(m.group(2))

    decayed = bool(DECAY_RE.search(answer))
    if not decayed and scan_row:
        om_pp = (scan_row.get("metrics") or {}).get("moat_om_trend_pp")
        if om_pp is not None and om_pp <= -3:
            decayed = True

    srcs = [int(x) for x in SRC_SCORE_RE.findall(answer)][:5]
    if len(srcs) >= 3:
        final, work = compute_arithmetic(srcs, decayed)
    else:
        # Sources not individually scored (some gold labels) — keep the
        # stated score, enforce only the decay rule; never invent numbers.
        final, _ = enforce_calibration(answer, printed, scan_row)
        work = ("sources not separately scored — grade taken as stated"
                + (f"; decay: eroding → −1 → {final}" if decayed else "")
                + f"; final {final}/10")

    # Drop the old verdict line (and any old arithmetic line), then insert
    # SCORE ARITHMETIC + MOAT VERDICT immediately before the ACTION line.
    body = _VERDICT_LINE_RE.sub("", answer, count=1)
    body = re.sub(r"^SCORE ARITHMETIC.*$\n?", "", body, flags=re.M)
    body = body.lstrip("\n").rstrip()
    block = (f"SCORE ARITHMETIC: {work}\n"
             f"MOAT VERDICT: {word} — {final}/10")
    am = _ACTION_LINE_RE.search(body)
    if am:
        new = body[:am.start()].rstrip() + "\n" + block + "\n" + \
            body[am.start():]
    else:
        new = body + "\n" + block
    return new, final
