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

from ai_moat.calibration import (enforce_calibration,      # noqa: E402
                                 rewrite_verdict_score, BENCHMARK_BLOCK)
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


def build_rows():
    """Calibrated training rows from the existing gold + silver answers."""
    rows = []
    stats = {"kept": 0, "rescored": 0, "skipped": 0}
    for name, repeat in (("gold.jsonl", 4), ("silver.jsonl", 1)):
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
            final, notes = enforce_calibration(ans, score)
            new_ans = (rewrite_verdict_score(ans, final)
                       if final != score else ans)
            if notes:
                # Show the WORK, not just the corrected number — this is
                # what teaches the arithmetic rather than new magic values.
                new_ans += ("\nSCORE ARITHMETIC (mandatory): "
                            + "; ".join(notes))
                stats["rescored"] += 1
            else:
                new_ans += ("\nSCORE ARITHMETIC (mandatory): source average "
                            "and redundancy checked — no adjustment needed.")
                stats["kept"] += 1
            sysmsg = msgs[0]["content"]
            if "BENCHMARKS —" not in sysmsg:
                sysmsg = sysmsg + "\n\n" + BENCHMARK_BLOCK
            rows.append({"messages": [
                {"role": "system", "content": sysmsg},
                {"role": "user", "content": msgs[1]["content"]},
                {"role": "assistant", "content": new_ans},
            ], "tier": "calib", "ticker": r.get("ticker", "?"),
                # rescored examples carry the lesson — repeat them harder
                "_repeat": repeat * (2 if notes else 1)})
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
    ap.add_argument("--epochs", type=int, default=1,
                    help="1 epoch is enough — we're correcting a numeric "
                         "habit, not teaching new knowledge")
    ap.add_argument("--lr", type=float, default=4e-5)
    ap.add_argument("--dry-run", action="store_true",
                    help="only build + print the calibrated rows, no GPU")
    args = ap.parse_args()

    rows = build_rows()
    if args.dry_run:
        for r in rows[:3]:
            print("=" * 60)
            print(r["ticker"], "->",
                  r["messages"][2]["content"].splitlines()[0])
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
            learning_rate=args.lr,
            lr_scheduler_type="cosine", warmup_ratio=0.1,
            logging_steps=1, save_strategy="epoch",
            bf16=True, optim="paged_adamw_8bit",
            max_seq_length=args.seq, packing=True, seed=42,
        ),
        callbacks=[StepSpeedSentinel()],
    )
    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\nDONE — calibrated adapter at: {out_dir}")
    print("Next: python ai_moat/quantize_model.py   (auto-picks it)")


if __name__ == "__main__":
    main()
