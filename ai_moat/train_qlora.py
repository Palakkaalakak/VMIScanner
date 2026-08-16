"""QLoRA fine-tune on the 3-tier moat dataset.
Sized for an RTX 5070 Ti (12GB VRAM) + 16GB RAM. 100% free stack.

BASE MODEL CHOICE (2026-08 review):
  qwen3-14b (DEFAULT) — Qwen3-14B: newer generation, stronger reasoning,
      mature unsloth QLoRA support AND mature llama.cpp GGUF support
      (standard transformer arch — quantizes cleanly with quantize_model.py).
  14b / 7b — the previous Qwen2.5 bases, kept as proven fallbacks.
  Qwen3.8-27B was evaluated and REJECTED as a student: (a) 27B 4-bit QLoRA
      needs ~17-19GB — does not fit 12GB VRAM for training; (b) its new
      Gated DeltaNet hybrid architecture is days old — fine as a TEACHER
      via ready-made GGUFs, wrong as a fine-tune target on this hardware.

What it does:
  1. Loads the free pre-quantized 4-bit base (auto-downloads ~9GB once).
  2. Builds the training mix: gold x6 + contrastive x4 + silver x1,
     HOLDING OUT eval tickers (never trained on) for the trust gate.
  3. Trains a rank-64 LoRA adapter (~2-4h).
  4. Saves the adapter + runs the held-out eval so you see immediately
     whether the model reasons (downgrades corrupted evidence) or memorized.

Setup on your PC (once):
  pip install unsloth            # pulls torch/transformers/peft/trl/bitsandbytes

Run (from the repo root):
  python ai_moat/train_qlora.py                 # full run (Qwen3-14B)
  python ai_moat/train_qlora.py --base 14b      # Qwen2.5-14B fallback
  python ai_moat/train_qlora.py --base 7b       # smallest fallback if OOM
  python ai_moat/train_qlora.py --eval-only     # re-run the eval on a saved adapter

After training, quantize for LM Studio AUTOMATICALLY:
  python ai_moat/quantize_model.py --base qwen3-14b
  (one command: merge fp16 -> llama.cpp convert -> Q5_K_M GGUF ~9.9GB)

OOM ladder (apply in order): --seq 1536, then --rank 32, then --base 7b.
"""
from __future__ import annotations

import argparse
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "dataset")
OUTDIR = os.path.join(HERE, "outputs")

# Held-out gold tickers — the trust gate. Never trained on.
EVAL_GOLD = {"META", "TSLA", "AMZN", "INTC", "ZM"}
# Held-out contrastive tickers — must DOWNGRADE despite famous names.
EVAL_CONTRASTIVE = {"MSFT", "GOOGL", "AAPL"}

BASES = {
    "qwen3-14b": "unsloth/Qwen3-14B-unsloth-bnb-4bit",   # default: newest with mature support
    "14b": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",       # proven fallback
    "7b": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",         # smallest fallback
}


def load_jsonl(name):
    p = os.path.join(DS, name)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def build_mix(allow_no_silver: bool = False):
    gold = load_jsonl("gold.jsonl")
    contr = load_jsonl("contrastive.jsonl")
    silver = load_jsonl("silver.jsonl")
    if not gold:
        raise SystemExit("dataset/gold.jsonl missing — run: "
                         "python3 -m ai_moat.build_dataset")
    if not silver and not allow_no_silver:
        raise SystemExit(
            "\n" + "=" * 68 + "\n"
            "STOP: dataset/silver.jsonl is EMPTY — you skipped Step 2.2.\n"
            "The silver lessons are ~70% of the curriculum; teaching without\n"
            "them gives a much weaker model.\n\n"
            "DO THIS FIRST (with LM Studio running + server started):\n"
            "  python ai_moat/gen_silver.py\n"
            "(~1-1.5h, safe to interrupt/resume; see AI_SUPERGUIDE.md 2.2)\n\n"
            "Or, if you REALLY want a quick gold-only test run:\n"
            "  python ai_moat/train_qlora.py --no-silver\n"
            + "=" * 68)
    if not silver and allow_no_silver:
        print("NOTE: --no-silver set — teaching on gold+contrastive only "
              "(quick-test mode, not the full curriculum).")

    train, eval_rows = [], []
    for r in gold:
        (eval_rows if r["ticker"] in EVAL_GOLD else train).append(
            {**r, "_repeat": 6})
    for r in contr:
        (eval_rows if r["ticker"] in EVAL_CONTRASTIVE else train).append(
            {**r, "_repeat": 4})
    for r in silver:
        if r["ticker"] not in EVAL_GOLD | EVAL_CONTRASTIVE:
            train.append({**r, "_repeat": 1})

    expanded = []
    for r in train:
        expanded.extend([r] * r["_repeat"])
    random.seed(42)
    random.shuffle(expanded)
    print(f"train rows (after repeats): {len(expanded)}  "
          f"(gold {sum(1 for r in train if r['tier']=='gold')}, "
          f"contrastive {sum(1 for r in train if r['tier']=='contrastive')}, "
          f"silver {sum(1 for r in train if r['tier']=='silver')})  "
          f"| held-out eval rows: {len(eval_rows)}")
    return expanded, eval_rows


def run_eval(model, tokenizer, eval_rows, max_new=900):
    """Trust gate: print model answers on held-out rows for human review."""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    print("\n" + "=" * 70)
    print("HELD-OUT EVAL — verify by hand (TRAINING.md step 5):")
    print("  gold rows must match Adam; contrastive rows MUST downgrade.")
    print("=" * 70)
    for r in eval_rows:
        msgs = r["messages"][:2]          # system + user only
        ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_tensors="pt").to(model.device)
        out = model.generate(input_ids=ids, max_new_tokens=max_new,
                             temperature=0.2, do_sample=True)
        text = tokenizer.decode(out[0][ids.shape[1]:],
                                skip_special_tokens=True)
        print(f"\n----- {r['ticker']} [{r['tier']}] -----")
        print(text[:1200])
        expected = r["messages"][2]["content"].splitlines()[0]
        print(f">>> EXPECTED: {expected}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=list(BASES), default="qwen3-14b")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-silver", action="store_true",
                    help="allow teaching without silver lessons "
                         "(quick gold-only test run)")
    args = ap.parse_args()

    try:                                       # HF token (faster downloads)
        from ai_moat.hf_auth import load_hf_token
    except ImportError:                        # run as plain script
        from hf_auth import load_hf_token
    load_hf_token()

    from unsloth import FastLanguageModel     # import late: needs GPU
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    train_rows, eval_rows = build_mix(
        allow_no_silver=args.no_silver or args.eval_only)

    adapter_dir = os.path.join(OUTDIR, f"moat-{args.base}-lora")
    if args.eval_only:
        model, tokenizer = FastLanguageModel.from_pretrained(
            adapter_dir, max_seq_length=args.seq, load_in_4bit=True)
        run_eval(model, tokenizer, eval_rows)
        return

    model, tokenizer = FastLanguageModel.from_pretrained(
        BASES[args.base], max_seq_length=args.seq, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none",
        use_gradient_checkpointing="unsloth",   # what makes 14B fit in 12GB
        random_state=42)

    def to_text(row):
        return {"text": tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False)}

    ds = Dataset.from_list(train_rows).map(to_text,
                                           remove_columns=["messages", "tier",
                                                           "ticker", "_repeat"])

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=ds,
        dataset_text_field="text",
        args=SFTConfig(
            output_dir=adapter_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,      # effective batch 16
            num_train_epochs=args.epochs,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            logging_steps=5,
            save_strategy="epoch",
            bf16=True,
            optim="paged_adamw_8bit",
            max_seq_length=args.seq,
            packing=True,   # SPEED: our lessons are short; packing fills
                            # each 2048-token window instead of padding it
                            # -> ~2-3x fewer training steps, same learning.
            seed=42,
        ))
    trainer.train()
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nadapter saved -> {adapter_dir}")

    run_eval(model, tokenizer, eval_rows)

    print("\nNEXT: automatic quantization for LM Studio (one command):")
    print(f"  python ai_moat/quantize_model.py --base {args.base}")


if __name__ == "__main__":
    main()
