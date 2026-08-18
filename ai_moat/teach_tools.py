"""teach_tools.py — QUICK tool-calling top-up for the ALREADY-trained model.

This does NOT retrain from scratch. It loads the adapter you already
trained (outputs/moat-<base>-lora) and continues teaching it on the small
tool-calling dataset (~59 trajectories) so it learns to:

  * call research_stock(ticker) when no evidence card is given
  * call web_search(query) for recent-events questions
  * answer directly (NO tool call) when the card is already provided

Because the moat knowledge is already in the adapter and this dataset is
tiny, the whole pass is ~8-12 optimizer steps ≈ **20-40 minutes** on your
card (vs hours for full training).

Output goes to a NEW folder: outputs/moat-<base>-tools-lora
(your original adapter is never touched — if the top-up ever disappoints,
you still have the pure judge).

Usage:
    python ai_moat/build_tool_dataset.py     # once, seconds
    python ai_moat/teach_tools.py            # ~20-40 min
    python ai_moat/quantize_model.py         # auto-detects the newest adapter

Prereqs: same as train_qlora.py (eject LM Studio model first).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

OUTDIR = os.path.join(HERE, "outputs")
TOOLS_DATA = os.path.join(HERE, "dataset", "tools.jsonl")

from ai_moat.build_tool_dataset import TOOLS  # noqa: E402  (tool schemas)
from ai_moat.hf_auth import load_hf_token     # noqa: E402


def find_trained_adapter():
    """Newest outputs/moat-*-lora with a final adapter (not a -tools one)."""
    cands = []
    for d in glob.glob(os.path.join(OUTDIR, "moat-*-lora")):
        if d.endswith("-tools-lora"):
            continue
        if os.path.isfile(os.path.join(d, "adapter_model.safetensors")):
            cands.append((os.path.getmtime(d), d))
    if not cands:
        sys.exit("No trained adapter found in ai_moat/outputs/.\n"
                 "Run check_training.py first (recovers from checkpoints), "
                 "or train_qlora.py if you never trained.")
    cands.sort()
    return cands[-1][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None,
                    help="path to the trained adapter dir "
                         "(default: newest moat-*-lora)")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="lower than full training — we're topping up, "
                         "not overwriting")
    args = ap.parse_args()

    if not os.path.exists(TOOLS_DATA):
        sys.exit("dataset/tools.jsonl missing — run first:\n"
                 "  python -m ai_moat.build_tool_dataset")

    adapter_dir = args.adapter or find_trained_adapter()
    out_dir = adapter_dir.replace("-lora", "-tools-lora")
    print(f"continuing FROM: {adapter_dir}")
    print(f"writing NEW adapter TO: {out_dir}  (original untouched)")

    load_hf_token()

    # VRAM preflight (same spirit as train_qlora: teacher must be ejected)
    import torch
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        free_gb = free / 1024**3
        print(f"free VRAM: {free_gb:.1f}GB")
        if free_gb < 7.0:
            sys.exit("Less than 7GB free VRAM. Eject the model from "
                     "LM Studio (and close Chrome), then re-run.")

    from unsloth import FastLanguageModel
    sys.path.insert(0, HERE)
    from train_qlora import (_patch_py314_pickle_compat,   # reuse fixes
                             StepSpeedSentinel)
    _patch_py314_pickle_compat()
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    # Loading the adapter dir directly restores base + LoRA weights and
    # returns a trainable PEFT model — this is the "continue" trick.
    model, tokenizer = FastLanguageModel.from_pretrained(
        adapter_dir, max_seq_length=args.seq, load_in_4bit=True)
    FastLanguageModel.for_training(model)

    rows = [json.loads(l) for l in open(TOOLS_DATA, encoding="utf-8")]
    print(f"tool trajectories: {len(rows)}")

    def to_text(row):
        # tools=... makes Qwen3's template render the tool schemas and the
        # <tool_call> JSON blocks exactly as LM Studio will expect them.
        return {"text": tokenizer.apply_chat_template(
            row["messages"], tools=TOOLS, tokenize=False,
            add_generation_prompt=False)}

    ds = Dataset.from_list(rows).map(
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
    print("expected: ~8-15 steps total, ~20-40 min at healthy speed")
    try:
        trainer.train()
    except RuntimeError as e:
        if "memory" in str(e).lower() or "cuda" in str(e).lower():
            sys.exit(f"\nGPU memory error: {e}\nEject LM Studio's model, "
                     "close Chrome, re-run. Still failing -> --seq 1024")
        raise

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("\n" + "=" * 60)
    print(f"DONE. Tool-calling adapter saved: {out_dir}")
    print("Sanity check (should emit a <tool_call> for research_stock):")

    # quick smoke: no evidence card -> model should call the tool
    FastLanguageModel.for_inference(model)
    msgs = [{"role": "system", "content": "You are the moat analyst. "
             "Call research_stock if no evidence card is provided."},
            {"role": "user", "content": "Evaluate this company's economic "
             "moat using the rubric. No evidence card is attached — "
             "research it first.\n\nTICKER: KO /no_think"}]
    ids = tokenizer.apply_chat_template(
        msgs, tools=TOOLS, tokenize=True, add_generation_prompt=True,
        return_tensors="pt").to(model.device)
    out = model.generate(input_ids=ids, max_new_tokens=120,
                         temperature=0.2, do_sample=True)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
    print(text[:400])
    called = "tool_call" in text and "research_stock" in text
    print(f"\n>>> tool-call emitted: {'YES ✅' if called else 'NO ❌ — tell me'}")
    print("next: python ai_moat/quantize_model.py  "
          "(auto-detects this newest adapter)")


if __name__ == "__main__":
    main()
