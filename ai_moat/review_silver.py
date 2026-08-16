"""Automated quality-check pass over silver.jsonl — NO manual review needed.

WHY THIS EXISTS
---------------
The teacher tends to grade-inflate: too many 9/10 and 10/10 verdicts.
In Adam's framework a 9-10 is reserved for near-monopolies (think META's
10/10 network effect: ~60% of all internet users on its products). Very
few businesses deserve that.

WHAT IT DOES
------------
1. Reads dataset/silver.jsonl and parses each overall MOAT VERDICT score.
2. Every answer scoring ABOVE 8/10 (i.e. 9 or 10) is sent BACK to the
   teacher with a skeptical second-reviewer prompt: justify the 9-10
   against the evidence card, or downgrade. Same rubric, same evidence —
   nothing invented.
3. The reviewer's corrected answer replaces the original ONLY if it is
   format-valid and does not RAISE the score (a skeptical pass may
   confirm or lower, never inflate).
4. The original file is backed up to silver.pre_qc.jsonl before writing.
   Progress checkpoints after every review — safe to interrupt/re-run.

Run AFTER gen_silver.py finishes (LM Studio server still running):
  python ai_moat/review_silver.py             # QC all >8/10 answers
  python ai_moat/review_silver.py --min-score 8   # stricter: also re-check 8s
  python ai_moat/review_silver.py --workers 1     # if context errors appear

NOTE ON CONTEXT: each review prompt = rubric + evidence card + the first
answer (~2600 tokens) + up to 800 answer tokens. With Context Length 8192
and 2 workers (4096 each) this fits. If you see context errors: --workers 1.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen_silver import chat, looks_valid  # reuse the exact same plumbing

SILVER = os.path.join(HERE, "dataset", "silver.jsonl")
BACKUP = os.path.join(HERE, "dataset", "silver.pre_qc.jsonl")
PROGRESS = os.path.join(HERE, "dataset", "silver_qc_progress.jsonl")

SCORE_RE = re.compile(r"MOAT VERDICT:.*?(\d+)\s*/\s*10")

REVIEWER_ADDON = """

--- SECOND-REVIEWER MODE (STRICT) ---
You are now acting as a SKEPTICAL senior reviewer. A first reviewer
graded this company's moat 9/10 or 10/10 overall. In this framework,
9-10 is reserved for near-monopoly moats — businesses like a dominant
network-effect platform used by the majority of its addressable market.
Most genuinely great businesses are 6-8. Grade inflation is the single
most common first-reviewer error.

Your job:
1. Re-examine ONLY the evidence card below. Do not invent numbers or
   facts that are not in it.
2. For each of the 5 sources, ask: does the EVIDENCE support this score,
   or did the first reviewer assume it from the company's reputation?
3. Sustain the 9-10 overall ONLY if the evidence itself is exceptional.
   Otherwise downgrade to what the evidence supports (often 6-8).
   You may NEVER raise the score above the first reviewer's.
4. Output the FULL corrected answer in EXACTLY the same mandated format
   (MOAT VERDICT / SOURCES / SOURCES PASSING / PRICING-POWER TEST /
   INDUSTRY SCREEN / KEY-MAN RISK / DECAY CHECK / REASONING / ACTION).
   Output ONLY the corrected answer, nothing else."""

_write_lock = threading.Lock()


def parse_score(answer: str):
    m = SCORE_RE.search(answer)
    return int(m.group(1)) if m else None


def review_one(row: dict, args) -> tuple:
    """Returns (row, revised_answer_or_None, note)."""
    original = row["messages"][-1]["content"]
    old_score = parse_score(original)
    system = row["messages"][0]["content"] + REVIEWER_ADDON
    user = (row["messages"][1]["content"]
            + "\n\n--- FIRST REVIEWER'S ANSWER (verify or downgrade) ---\n"
            + original)
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": user}]
    for attempt in range(1 + args.retries):
        try:
            a = chat(args.base_url, args.model, msgs,
                     args.max_tokens, thinking=False)
        except Exception as e:
            print(f"  {row['ticker']}: request failed ({e})")
            time.sleep(5)
            continue
        if not looks_valid(a):
            print(f"  {row['ticker']}: reviewer answer failed the format "
                  f"gate (attempt {attempt+1}), retrying")
            continue
        new_score = parse_score(a)
        if new_score is None:
            continue
        if old_score is not None and new_score > old_score:
            # A skeptical pass must never inflate — keep the original.
            return row, None, f"reviewer tried to RAISE {old_score}->" \
                              f"{new_score}; kept original"
        if old_score is not None and new_score == old_score:
            return row, a, f"confirmed {old_score}/10"
        return row, a, f"downgraded {old_score}/10 -> {new_score}/10"
    return row, None, "SKIPPED (reviewer kept failing) — original kept"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--model", default="local-model")
    ap.add_argument("--min-score", type=int, default=9,
                    help="re-review answers scoring >= this overall "
                         "(default 9 = everything above 8/10)")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=800)
    args = ap.parse_args()

    if not os.path.exists(SILVER):
        sys.exit(f"missing {SILVER} — run gen_silver.py first")

    rows = [json.loads(l) for l in open(SILVER, encoding="utf-8")]

    # Resume support: skip tickers already reviewed in a previous run.
    done = {}
    if os.path.exists(PROGRESS):
        for l in open(PROGRESS, encoding="utf-8"):
            try:
                d = json.loads(l)
                done[d["ticker"]] = d
            except Exception:
                pass

    scores = [parse_score(r["messages"][-1]["content"]) for r in rows]
    dist = {}
    for s in scores:
        dist[s] = dist.get(s, 0) + 1
    print("score distribution BEFORE QC:",
          {k: dist[k] for k in sorted(dist, key=lambda x: (x is None, x))})

    targets = [r for r, s in zip(rows, scores)
               if s is not None and s >= args.min_score
               and r["ticker"] not in done]
    already = sum(1 for r, s in zip(rows, scores)
                  if s is not None and s >= args.min_score
                  and r["ticker"] in done)
    print(f"{len(targets)} answers scored >= {args.min_score}/10 and need "
          f"review ({already} already reviewed in a previous run)")

    if targets:
        if not os.path.exists(BACKUP):
            with open(BACKUP, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"backup written -> {BACKUP}")

        with open(PROGRESS, "a", encoding="utf-8") as pf, \
             ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futures = [ex.submit(review_one, r, args) for r in targets]
            for i, fut in enumerate(as_completed(futures), 1):
                row, revised, note = fut.result()
                print(f"[{i}/{len(targets)}] {row['ticker']}: {note}")
                rec = {"ticker": row["ticker"], "note": note}
                if revised is not None:
                    rec["answer"] = revised
                with _write_lock:
                    pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    pf.flush()
                done[row["ticker"]] = rec

    # Merge reviews into silver.jsonl (atomic: tmp file then replace).
    replaced = 0
    for r in rows:
        rec = done.get(r["ticker"])
        if rec and rec.get("answer"):
            r["messages"][-1]["content"] = rec["answer"]
            r["qc"] = rec["note"]
            replaced += 1
    tmp = SILVER + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, SILVER)

    after = {}
    for r in rows:
        s = parse_score(r["messages"][-1]["content"])
        after[s] = after.get(s, 0) + 1
    print(f"\n{replaced} answers updated in {SILVER}")
    print("score distribution AFTER QC: ",
          {k: after[k] for k in sorted(after, key=lambda x: (x is None, x))})
    print("originals preserved in silver.pre_qc.jsonl; "
          "re-run any time — already-reviewed tickers are skipped")
    print("NEXT: python ai_moat/train_qlora.py")


if __name__ == "__main__":
    main()
