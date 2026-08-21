"""teach_calibration.py — QUICK score-calibration top-up (tier discipline).

WHY (user report 2026-08-19): after full training the model scored AAPL and
ABBV both 9/10 — and even printed "DECAY CHECK: decaying" for ABBV while
STILL giving it a 9. Prose rubric edits alone didn't fix it. This top-up
teaches the model the exact scoring arithmetic the dashboard now enforces
in code (ai_moat/calibration.py):

  * ROUND DOWN — overall ≤ floor(average of the 5 source scores)
  * REDUNDANCY CAP — 10 needs all 5 sources ≥8; 9 needs 4 (or 3 all ≥9)
  * DECAY PENALTY — a decaying DECAY CHECK costs 1 point
  * BENCHMARK ANCHOR — every prompt carries the MA/AAPL/MSFT 9-10 shelf,
    so the model compares against the Heavenly-Queen bar, not a vacuum
    (MA is the golden example: ROIC ≥15% 10/10y, 85% up-years, +7pp
    margins, FCF+ 10/10 — CONSISTENCY of growing profits is the qualifier)

HOW: it rebuilds training answers from the datasets you ALREADY have
(gold.jsonl + silver.jsonl) by passing each answer through the very same
enforce_calibration() the dashboard uses, rewriting the verdict line to
the enforced score, appending the arithmetic work shown, and adding the
benchmark block to the system prompt. The model literally learns the
arithmetic it will be graded by. No new labels are invented — verdict
words (WIDE/NARROW/...) are untouched, only numeric tier discipline.

Like teach_tools.py, this does NOT retrain from scratch: it continues from
your newest adapter (the -tools one if present, so tool-calling survives)
and writes a NEW folder: outputs/moat-<base>-…-calib-lora. ~15-40 min.

Usage (on your PC, LM Studio model EJECTED first):
    python ai_moat/teach_calibration.py     # builds data in-process, trains
    python ai_moat/quantize_model.py        # auto-picks the newest adapter
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

OUTDIR = os.path.join(HERE, "outputs")
DS = os.path.join(HERE, "dataset")

from ai_moat.calibration import (restructure_answer,       # noqa: E402
                                 BENCHMARK_BLOCK)
from ai_moat.hf_auth import load_hf_token                  # noqa: E402

SCORE_RE_TXT = r"MOAT VERDICT:.*?(\d+)\s*/\s*10"


def find_newest_adapter():
    """Newest adapter, preferring -tools-lora so tool-calling survives the
    calibration pass. Never continues from a -calib one (no stacking)."""
    cands = []
    for d in glob.glob(os.path.join(OUTDIR, "moat-*-lora")):
        if d.endswith("-calib-lora"):
            continue
        if os.path.isfile(os.path.join(d, "adapter_model.safetensors")):
            cands.append((os.path.getmtime(d), d.endswith("-tools-lora"), d))
    if not cands:
        sys.exit("No trained adapter found in ai_moat/outputs/.\n"
                 "Run teach_tools.py (or train_qlora.py) first.")
    cands.sort()
    return cands[-1][2]


def _load_scan_rows():
    """scan_results.json by ticker — supplies the measured decay evidence
    (moat_om_trend_pp) so restructuring matches the dashboard exactly."""
    p = os.path.join(ROOT, "public", "data", "scan_results.json")
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        rows = d["results"] if isinstance(d, dict) and "results" in d else d
        return {r["ticker"]: r for r in rows
                if isinstance(r, dict) and r.get("ticker")}
    except (OSError, json.JSONDecodeError, KeyError):
        return {}


def build_rows():
    """Calibrated training rows from the existing gold + silver answers.

    ROOT-CAUSE FIX (user report 2026-08-20): rewriting only the number was
    not enough — the model kept printing wrong scores because the trained
    format put the verdict on LINE 1, forcing it to commit to a score
    BEFORE writing the source scores it should be averaging. Every answer
    is now RESTRUCTURED arithmetic-first (sources → tests → SCORE
    ARITHMETIC with the work shown → MOAT VERDICT last), so the score
    token is generated AFTER the numbers that determine it."""
    rows = []
    scan_rows = _load_scan_rows()
    stats = {"kept": 0, "rescored": 0, "skipped": 0}
    # Repeats trimmed (user report 2026-08-21: 40 steps × 138s = 92 min on a
    # 5070 Ti — way over budget). Every row already teaches the new
    # arithmetic-first ORDER, so heavy gold repetition adds little; the
    # step budget in main() is the real length control.
    for name, repeat in (("gold.jsonl", 2), ("silver.jsonl", 1)):
        p = os.path.join(DS, name)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            msgs = r.get("messages") or []
            if len(msgs) < 3 or msgs[-1]["role"] != "assistant":
                continue
            ans = msgs[-1]["content"]
            m = re.search(SCORE_RE_TXT, ans)
            if not m:
                stats["skipped"] += 1
                continue
            score = int(m.group(1))
            new_ans, final = restructure_answer(
                ans, scan_rows.get(r.get("ticker")))
            if final is None:
                stats["skipped"] += 1
                continue
            notes = final != score
            stats["rescored" if notes else "kept"] += 1
            sysmsg = msgs[0]["content"]
            if "BENCHMARKS —" not in sysmsg:
                sysmsg = sysmsg + "\n\n" + BENCHMARK_BLOCK
            rows.append({"messages": [
                {"role": "system", "content": sysmsg},
                {"role": "user", "content": msgs[1]["content"]},
                {"role": "assistant", "content": new_ans},
            ], "tier": "calib", "ticker": r.get("ticker", "?"),
                # rescored examples carry the lesson — repeat them harder
                "_repeat": repeat * (2 if notes else 1)})  # noqa: B023
    print(f"calibration rows: {len(rows)} "
          f"(rescored {stats['rescored']}, already-correct {stats['kept']}, "
          f"skipped-no-score {stats['skipped']})")
    if len(rows) < 20:
        sys.exit("Too few usable rows — is dataset/gold.jsonl present? "
                 "Run: python -m ai_moat.build_dataset")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None,
                    help="adapter dir to continue from "
                         "(default: newest, prefers -tools-lora)")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=2,
                    help="2 epochs — we're changing the OUTPUT ORDER "
                         "(arithmetic before verdict), a stronger habit "
                         "change than a number tweak")
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="slightly higher than before — fewer optimizer "
                         "steps under the step budget need a bit more "
                         "push per step")
    ap.add_argument("--max-steps", type=int, default=8,
                    help="hard step budget — 8 steps ≈ 18 min at the "
                         "~140 s/step measured on a 5070 Ti Laptop "
                         "(packed 16x2048-token batches ≈ 260k tokens "
                         "≈ the whole calibration set once). 0 = no cap, "
                         "run the full --epochs")
    ap.add_argument("--dry-run", action="store_true",
                    help="only build + print the calibrated rows, no GPU")
    args = ap.parse_args()

    rows = build_rows()
    if args.dry_run:
        for r in rows[:3]:
            print("=" * 60)
            tail = r["messages"][2]["content"].splitlines()[-4:]
            print(r["ticker"], "->")
            for ln in tail:
                print("   ", ln[:150])
        return

    adapter_dir = args.adapter or find_newest_adapter()
    out_dir = adapter_dir.replace("-lora", "-calib-lora")
    print(f"continuing FROM: {adapter_dir}")
    print(f"writing NEW adapter TO: {out_dir}  (original untouched)")

    load_hf_token()

    import torch
    if torch.cuda.is_available():
        free, _total = torch.cuda.mem_get_info()
        free_gb = free / 1024**3
        print(f"free VRAM: {free_gb:.1f}GB")
        if free_gb < 7.0:
            sys.exit("Less than 7GB free VRAM. Eject the model from "
                     "LM Studio (and close Chrome), then re-run.")

    from unsloth import FastLanguageModel
    sys.path.insert(0, HERE)
    from train_qlora import (_patch_py314_pickle_compat,
                             make_step_speed_sentinel)
    StepSpeedSentinel = make_step_speed_sentinel()
    _patch_py314_pickle_compat()
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = FastLanguageModel.from_pretrained(
        adapter_dir, max_seq_length=args.seq, load_in_4bit=True)
    FastLanguageModel.for_training(model)

    import random
    expanded = []
    for r in rows:
        expanded.extend([r] * r.pop("_repeat"))
    random.seed(42)
    random.shuffle(expanded)
    print(f"train rows after repeats: {len(expanded)}")

    def to_text(row):
        return {"text": tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False)}

    ds = Dataset.from_list(expanded).map(
        to_text, remove_columns=["messages", "tier", "ticker"])

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        args=SFTConfig(
            output_dir=out_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            num_train_epochs=args.epochs,
            # max_steps overrides epochs when set — the ≤20-min budget
            **({"max_steps": args.max_steps} if args.max_steps > 0 else {}),
            learning_rate=args.lr,
            lr_scheduler_type="cosine", warmup_ratio=0.1,
            # Checkpoint every 2 steps (~5 min) so an interrupted run
            # RESUMES instead of restarting from 0% (user report
            # 2026-08-21). save_total_limit keeps disk use bounded.
            logging_steps=1, save_strategy="steps", save_steps=2,
            save_total_limit=2,
            bf16=True, optim="paged_adamw_8bit",
            max_seq_length=args.seq, packing=True, seed=42,
        ),
        callbacks=[StepSpeedSentinel()],
    )
    def _usable_checkpoint():
        """Newest checkpoint from a run with the SAME step budget —
        resuming across a changed config would silently mis-train."""
        best = None
        for c in glob.glob(os.path.join(out_dir, "checkpoint-*")):
            st = os.path.join(c, "trainer_state.json")
            try:
                with open(st, encoding="utf-8") as f:
                    state = json.load(f)
                if state.get("max_steps") == trainer.args.max_steps and \
                        state.get("global_step", 0) < trainer.args.max_steps:
                    if best is None or os.path.getmtime(c) > \
                            os.path.getmtime(best):
                        best = c
            except (OSError, json.JSONDecodeError):
                continue
        return best

    latest = _usable_checkpoint()
    if latest:
        print(f"RESUMING from {os.path.basename(latest)} — not from 0%")
        trainer.train(resume_from_checkpoint=latest)
    else:
        trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nDONE — calibrated adapter at: {out_dir}")
    print("Next: python ai_moat/quantize_model.py   (auto-picks it)")


if __name__ == "__main__":
    main()
