#!/usr/bin/env python3
"""
check_training.py — did my training survive the shutdown?
=========================================================

Run this after a crash / accidental shutdown / power cut during or after
`train_qlora.py`. It tells you EXACTLY what state your training is in,
and rescues it if needed. It never deletes anything.

Usage (from the repo root):

    python ai_moat/check_training.py

What it checks, in plain language:

  train_qlora.py does things in this order:
    1. trains (the progress bar you watched)          <- slow, hours
    2. saves the final adapter to the output folder    <- fast, seconds
    3. saves the tokenizer                             <- fast, seconds
    4. runs the held-out trust-gate eval (8 samples)   <- slow, minutes

  So if the bar FINISHED before the shutdown, step 2 almost certainly
  ran too — your trained brain is safe and only the eval was skipped.

  Also, training saves a checkpoint at the end of EVERY epoch
  (save_strategy="epoch"). With the default 3 epochs / 90 steps that
  means checkpoint-30, checkpoint-60, checkpoint-90. Even if the final
  save was interrupted, the newest checkpoint holds an adapter we can
  promote to "final" — worst case you lose less than one epoch.

Outcomes this script reports:

  COMPLETE   -> final adapter present. Nothing lost. Run the eval next.
  RECOVERED  -> final adapter was missing; promoted it from the newest
                checkpoint. Run the eval next.
  NOTHING    -> no adapter and no checkpoints. Training has to be re-run.

No numbers are invented: everything printed is read from the actual
files on your disk (file sizes, trainer_state.json step counts).
"""

import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "outputs")

# Files that make an adapter loadable by --eval-only / quantize_model.py
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")

# Nice-to-have tokenizer files (train_qlora saves these in step 3;
# checkpoints usually contain them too)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)

CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def human_size(path):
    try:
        n = os.path.getsize(path)
    except OSError:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f}TB"


def checkpoint_info(ckpt_dir):
    """Return (global_step_from_state_or_None, has_adapter_files)."""
    step = None
    state_path = os.path.join(ckpt_dir, "trainer_state.json")
    if os.path.isfile(state_path):
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                step = json.load(f).get("global_step")
        except Exception:
            step = None
    has_adapter = all(
        os.path.isfile(os.path.join(ckpt_dir, f)) for f in ADAPTER_FILES
    )
    return step, has_adapter


def scan_run(run_dir):
    """Inspect one moat-<base>-lora directory. Returns a status string."""
    name = os.path.basename(run_dir)
    print(f"\n=== {name} ===")

    # --- 1. Is the FINAL adapter present in the folder root? ---
    root_ok = all(os.path.isfile(os.path.join(run_dir, f)) for f in ADAPTER_FILES)
    if root_ok:
        for f in ADAPTER_FILES:
            p = os.path.join(run_dir, f)
            print(f"  [OK] {f}  ({human_size(p)})")
        tok_present = [f for f in TOKENIZER_FILES
                       if os.path.isfile(os.path.join(run_dir, f))]
        if tok_present:
            print(f"  [OK] tokenizer files present ({len(tok_present)} found)")
        else:
            print("  [note] no tokenizer files in root — the eval/quantize "
                  "steps can still fetch the tokenizer from the base model, "
                  "so this is not a blocker.")
        print("  STATUS: COMPLETE — the final adapter was saved before the "
              "shutdown. Nothing was lost.")
        print("  The only thing the shutdown skipped is the held-out "
              "trust-gate eval (it runs AFTER the save).")
        return "COMPLETE"

    # --- 2. No final adapter — look for epoch checkpoints. ---
    print("  [missing] final adapter not found in folder root")
    ckpts = []
    for entry in sorted(os.listdir(run_dir)):
        m = CKPT_RE.match(entry)
        if m:
            ckpt_dir = os.path.join(run_dir, entry)
            if os.path.isdir(ckpt_dir):
                step, has_adapter = checkpoint_info(ckpt_dir)
                num = step if step is not None else int(m.group(1))
                ckpts.append((num, ckpt_dir, has_adapter))

    if not ckpts:
        print("  [missing] no checkpoints either")
        print("  STATUS: NOTHING — no usable training artifacts here. "
              "Training would need to be re-run for this base.")
        return "NOTHING"

    ckpts.sort(key=lambda t: t[0])
    for num, d, has_adapter in ckpts:
        flag = "usable" if has_adapter else "INCOMPLETE (no adapter file)"
        print(f"  found {os.path.basename(d)}  (global_step={num}, {flag})")

    # newest checkpoint that actually has adapter files
    usable = [c for c in ckpts if c[2]]
    if not usable:
        print("  STATUS: NOTHING usable — checkpoints exist but none contain "
              "an adapter file. Training would need to be re-run.")
        return "NOTHING"

    num, best, _ = usable[-1]
    print(f"  -> promoting newest usable checkpoint (step {num}) to final ...")
    copied = []
    for f in ADAPTER_FILES + TOKENIZER_FILES:
        src = os.path.join(best, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(run_dir, f))
            copied.append(f)
    for f in copied:
        print(f"     copied {f}  ({human_size(os.path.join(run_dir, f))})")
    print(f"  STATUS: RECOVERED — adapter restored from checkpoint-{num}. "
          "At most part of one epoch of learning is lost (checkpoints are "
          "written at every epoch boundary).")
    return "RECOVERED"


def main():
    print("check_training.py — post-shutdown training inspector")
    print(f"looking in: {OUTDIR}")

    if not os.path.isdir(OUTDIR):
        print("\nNo outputs/ folder exists at all — training never got far "
              "enough to create it. You'll need to run "
              "`python ai_moat/train_qlora.py` again.")
        sys.exit(1)

    runs = [os.path.join(OUTDIR, d) for d in sorted(os.listdir(OUTDIR))
            if d.startswith("moat-") and d.endswith("-lora")
            and os.path.isdir(os.path.join(OUTDIR, d))]

    if not runs:
        print("\nNo moat-*-lora folders found in outputs/. Training never "
              "created its output directory — re-run "
              "`python ai_moat/train_qlora.py`.")
        sys.exit(1)

    results = {}
    for run_dir in runs:
        results[os.path.basename(run_dir)] = scan_run(run_dir)

    good = [n for n, s in results.items() if s in ("COMPLETE", "RECOVERED")]

    print("\n" + "=" * 60)
    if good:
        print("NEXT STEPS (your trained brain is safe):")
        print("  1. Run the trust-gate eval the shutdown skipped:")
        print("         python ai_moat/train_qlora.py --eval-only")
        print("     (loads the saved adapter, answers the 8 held-out")
        print("      questions so YOU can judge if it learned properly)")
        print("  2. If the answers look good, package it for LM Studio:")
        print("         python ai_moat/quantize_model.py")
        print("     (auto-detects the newest trained adapter)")
    else:
        print("NEXT STEPS: no recoverable training found — re-run:")
        print("         python ai_moat/train_qlora.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
