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
  3. Trains a rank-64 LoRA adapter (~1-2h with packing enabled).
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

IMPORTANT — VRAM: the student needs ~10-11GB of the 12GB card. If LM
Studio still has the TEACHER loaded, it is holding ~9GB and this script
CANNOT fit ("Some modules are dispatched on the CPU" error). Eject the
model in LM Studio (or quit LM Studio) before running. A preflight check
below catches this and tells you.
"""
from __future__ import annotations

import argparse
import json
import os
import random

# Bumped on every behavioural change — printed at startup so a stale
# checkout is obvious at a glance ("did my git pull actually land?").
SCRIPT_VERSION = "2026-08-17f (auto student sizing by measured free VRAM — training-time need, not load-time)"

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "dataset")
OUTDIR = os.path.join(HERE, "outputs")

# Held-out gold tickers — the trust gate. Never trained on.
EVAL_GOLD = {"META", "TSLA", "AMZN", "INTC", "ZM"}
# Held-out contrastive tickers — must DOWNGRADE despite famous names.
EVAL_CONTRASTIVE = {"MSFT", "GOOGL", "AAPL"}

# MEASURED weight sizes (HuggingFace API, 2026-08): the "unsloth-dynamic"
# 14B variant is 10.36GB of weights — on a 12GB Windows laptop (~1.5GB
# taken by the desktop) that leaves NOTHING for activations and the load
# fails with "Some modules are dispatched on the CPU". The standard
# bnb-4bit repo is 9.25GB and fits (tightly). Qwen3-8B (5.66GB) is the
# comfortable fallback — still a strong student for a rubric task.
BASES = {
    "qwen3-14b": "unsloth/Qwen3-14B-bnb-4bit",            # default: 9.25GB weights
    "qwen3-8b": "unsloth/Qwen3-8B-bnb-4bit",              # 5.66GB — safe fallback
    "14b": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",       # proven fallback
    "7b": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",         # smallest fallback
}

# Minimum free VRAM to TRAIN (not just load!). Measured reality from an
# RTX 5070 Ti Laptop run (2026-08-17): 14B loaded fine into 10.8GB free
# but died at the first training step ("No or negligible GPU memory
# available for fused cross entropy") — training needs weights PLUS
# LoRA gradients + activations + the loss buffer. 9.25GB of weights +
# ~2.5-3GB of training overhead means a 14B student needs ~12GB FREE,
# which a Windows laptop 12GB card (desktop eats ~1GB) never has.
TRAIN_FREE_GB = {"qwen3-14b": 12.0, "14b": 12.0, "qwen3-8b": 7.6, "7b": 6.2}


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


def _patch_py314_pickle_compat(_force: bool = False):
    """Python 3.14 changed pickle's dict-saving internals: save_dict now
    calls self._batch_setitems(items, obj) with a NEW second argument,
    while dill / datasets still define and CALL the old (self, items)
    form. The mismatch surfaces both ways ('takes 2 positional arguments
    but 3 were given' AND 'missing 1 required positional argument: obj')
    because the call chain crosses the boundary twice:
        stdlib save_dict -> datasets Pickler -> stdlib _batch_setitems.

    Defense in depth, two independent layers:
      L1: REPLACE datasets' Pickler._batch_setitems with a version that
          keeps its key-sorting behaviour but adapts to whichever
          signature the running stdlib exposes (2-arg or 3-arg).
      L2: FAILSAFE — wrap datasets' Hasher.hash so ANY exception during
          fingerprinting falls back to a random fingerprint. The
          fingerprint is only a cache key; for a one-shot training run a
          random one merely disables dataset caching, which we don't
          need. Training can then never die in fingerprinting again.

    No-op on Python <= 3.13. Harmless once dill/datasets ship fixes."""
    import sys
    if sys.version_info < (3, 14) and not _force:
        return
    patched = []

    # ---- L1: signature-adaptive _batch_setitems on datasets' Pickler ----
    try:
        import pickle as _pickle
        from datasets.utils import _dill as _ds_dill

        def _adaptive_batch_setitems(self, items, obj=None):
            if not getattr(self, "_legacy_no_dict_keys_sorting", False):
                try:
                    items = sorted(items)      # datasets: order-free hashing
                except Exception:
                    from datasets.fingerprint import Hasher
                    items = sorted(items, key=lambda x: Hasher.hash(x[0]))
            base = _pickle._Pickler._batch_setitems
            try:                               # py3.14 stdlib: (items, obj)
                return base(self, iter(items), obj)
            except TypeError:                  # older stdlib: (items)
                return base(self, iter(items))

        if getattr(_ds_dill, "Pickler", None) is not None:
            _ds_dill.Pickler._batch_setitems = _adaptive_batch_setitems
            patched.append("datasets Pickler (adaptive)")
    except Exception:
        pass

    # ---- L2: never-crash fingerprinting failsafe ----
    try:
        from datasets.fingerprint import Hasher
        _orig_hash = Hasher.hash.__func__

        def _safe_hash(cls, value):
            try:
                return _orig_hash(cls, value)
            except Exception:
                import uuid                    # random cache key: caching
                return uuid.uuid4().hex        # off, correctness intact

        Hasher.hash = classmethod(_safe_hash)
        patched.append("Hasher.hash failsafe")
    except Exception:
        pass

    if patched:
        print("py3.14 compat: " + "; ".join(patched))


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
    ap.add_argument("--base", choices=list(BASES) + ["auto"], default="auto",
                    help="auto = pick the largest student that ACTUALLY "
                         "fits your free VRAM for training (recommended)")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--no-silver", action="store_true",
                    help="allow teaching without silver lessons "
                         "(quick gold-only test run)")
    args = ap.parse_args()

    print(f"train_qlora version: {SCRIPT_VERSION}")

    # ---- auto-pick the student by MEASURED free VRAM (training needs
    # much more than loading — see TRAIN_FREE_GB comment) ----
    if args.base == "auto":
        chosen = "7b"                                   # absolute fallback
        try:
            import torch
            if torch.cuda.is_available():
                free_gb = torch.cuda.mem_get_info()[0] / 1024**3
                for cand in ("qwen3-14b", "qwen3-8b", "7b"):
                    if free_gb >= TRAIN_FREE_GB[cand]:
                        chosen = cand
                        break
                print(f"auto student selection: {free_gb:.1f}GB free VRAM "
                      f"-> {chosen} "
                      f"(needs ~{TRAIN_FREE_GB[chosen]}GB to train; "
                      f"a 14B student needs ~{TRAIN_FREE_GB['qwen3-14b']}GB "
                      f"free — loading it is not enough, training "
                      f"gradients/activations must fit too)")
            else:
                print("auto student selection: no CUDA visible -> 7b")
        except Exception:
            print("auto student selection: VRAM probe failed -> 7b")
        args.base = chosen

    print(f"student model: {BASES[args.base]}")

    try:                                       # HF token (faster downloads)
        from ai_moat.hf_auth import load_hf_token
    except ImportError:                        # run as plain script
        from hf_auth import load_hf_token
    load_hf_token()

    # ---- VRAM preflight: the #1 failure is LM Studio still holding the
    # teacher (~9GB). Training needs nearly the whole 12GB card free. ----
    try:
        import torch
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            free_gb, total_gb = free_b / 1024**3, total_b / 1024**3
            need = TRAIN_FREE_GB.get(args.base, 12.0)
            if free_gb < need:
                raise SystemExit(
                    "\n" + "=" * 68 + "\n"
                    f"STOP: only {free_gb:.1f}GB of {total_gb:.1f}GB VRAM is "
                    f"free — '{args.base}' needs ~{need}GB+.\n"
                    "Something else is holding the GPU (LM Studio model "
                    "still loaded?\ngames? browser video tabs?).\n\n"
                    "FIX: free the VRAM and re-run — or use the smaller "
                    "student that\nalways fits on this card:\n"
                    "  python ai_moat/train_qlora.py --base qwen3-8b\n"
                    + "=" * 68)
            print(f"VRAM preflight OK: {free_gb:.1f}GB free "
                  f"of {total_gb:.1f}GB")
    except SystemExit:
        raise
    except Exception:
        pass  # preflight is best-effort; unsloth gives its own error if OOM

    from unsloth import FastLanguageModel     # import late: needs GPU
    _patch_py314_pickle_compat()               # BEFORE datasets is used
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

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            BASES[args.base], max_seq_length=args.seq, load_in_4bit=True)
    except ValueError as e:
        if "dispatched on the CPU" in str(e) and args.base != "qwen3-8b":
            raise SystemExit(
                "\n" + "=" * 68 + "\n"
                "The model didn't fit in your free VRAM even after the "
                "preflight.\nYour card is right on the edge for a 14B "
                "student. Two options:\n"
                "  1. Free more VRAM (quit LM Studio fully, close every "
                "browser tab\n     playing video, disconnect a second "
                "monitor) and re-run.\n"
                "  2. Use the 8B student — guaranteed fit, still strong:\n"
                "     python ai_moat/train_qlora.py --base qwen3-8b\n"
                + "=" * 68)
        raise
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
    try:
        trainer.train()
    except RuntimeError as e:
        low = str(e).lower()
        if "memory" in low or "cuda" in low:
            raise SystemExit(
                "\n" + "=" * 68 + "\n"
                f"GPU ran out of memory DURING training of '{args.base}'.\n"
                "Drop one size down — it will finish and the quality "
                "difference on\na fixed-rubric task is small:\n"
                "  python ai_moat/train_qlora.py --base qwen3-8b\n"
                "(or --base 7b if 8B also fails)\n"
                + "=" * 68)
        raise
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nadapter saved -> {adapter_dir}")

    run_eval(model, tokenizer, eval_rows)

    print("\nNEXT: automatic quantization for LM Studio (one command):")
    print(f"  python ai_moat/quantize_model.py --base {args.base}")


if __name__ == "__main__":
    main()
