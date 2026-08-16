# Training the Moat-Reasoning Model — Super-High-Quality Recipe
### Target hardware: RTX 5070 Ti 12GB VRAM + 16GB system RAM

This is the maximum-quality path for your card, implementing your own idea:
**train a model LARGER than what you'd run day-to-day, then quantize it down.**

---

## 0. The decision (why this exact setup)

| Choice | Decision | Why |
|---|---|---|
| Base model | **Qwen2.5-14B-Instruct** (fallback: Qwen2.5-7B-Instruct if OOM) | 14B is the largest model whose **QLoRA training** fits a 12GB card. Qwen2.5 has the strongest instruction-following + reasoning per parameter in this class, Apache-licensed. |
| Method | **QLoRA** (4-bit NF4 frozen base + LoRA adapters, rank 64) | You train ~0.5% of the weights. 14B at 4-bit ≈ 8.2GB; adapters + optimizer + activations fit in the remaining ~3.5GB with the settings below. Full fine-tune of even 7B is impossible on 12GB. |
| After training | **Merge adapter → quantize to GGUF Q5_K_M** (Q4_K_M if you want more headroom) | This is the "bigger then quantize" step. A 14B at Q5 ≈ 9.9GB — fits fully in VRAM for inference at full GPU speed. Quality loss from Q5 is far smaller than the quality gain from 14B vs 7B. |
| Why not offload to RAM | PCIe ≈ 64GB/s vs GDDR7 ≈ 600-900GB/s | Offload gives capacity, not speed. For **training** it's worse: gradient traffic murders throughput. Everything below is sized to stay on-card. |

**Reasoning-model claim, honestly stated:** this is not an RL-trained "reasoning
model" like o1. It's a supervised model whose dataset *forces* reasoning:
- **Gold tier** anchors verdicts to Adam's actual judgments (no consensus anywhere in the pipeline).
- **Contrastive tier** shows the SAME famous ticker with corrupted fundamentals and a
  DOWNGRADED answer — so the model cannot pass by memorizing "MSFT = wide"; it must
  read the evidence card. That's the mechanism that makes it reason instead of recall.
- **Silver tier** teaches breadth: a teacher model answers rubric-only prompts (it
  never sees Morningstar/analyst moat ratings), you spot-check ~10%, then train.

---

## 1. Generate the silver tier (do this first, one overnight run)

The builder produced `dataset/silver_prompts.jsonl` (445 prompts, answers missing).
Fill them with a **teacher model** that sees ONLY the rubric — two options:

**Option A — fully local, fully free (slower):**
```bash
# On your machine, with ollama or llama.cpp:
ollama pull qwen2.5:32b-instruct-q3_K_M   # ~14GB, will offload to RAM — fine for
                                          # batch generation overnight (speed doesn't matter here)
python3 ai_moat/gen_silver.py --backend ollama --model qwen2.5:32b-instruct-q3_K_M
```
**Option B — API (a few dollars, better labels):** any strong API model, same script
with `--backend openai`. The system prompt already forbids consensus; the teacher
only receives the rubric + evidence card, so consensus bias cannot leak in.

Then **human review**: open `dataset/silver.jsonl`, check ~45 random rows (10%).
Delete rows where the verdict contradicts the evidence card. This half hour is the
highest-leverage quality step in the whole pipeline.

## 2. Assemble the training mix

```
gold.jsonl         × 6 repeats   (40 → 240 rows; these are the anchors)
contrastive.jsonl  × 4 repeats   (24 → 96 rows; anti-memorization)
silver.jsonl       × 1           (~420 rows after review)
                                 ≈ 750 training rows, ~40 held out for eval
```
**Hold out** 5 gold tickers (e.g. META, TSLA, AMZN, INTC, ZM) + 3 contrastive rows as
the eval set — never trained on, used to verify the model reproduces Adam's verdicts
AND downgrades corrupted evidence.

## 3. Train (unsloth — fastest QLoRA on consumer cards)

```bash
pip install unsloth
python3 ai_moat/train_qlora.py   # config below, ~2-4h on the 5070 Ti
```

Key config (already what `train_qlora.py` should contain):
```python
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",   # pre-quantized 4-bit base
    max_seq_length = 2048,                     # evidence card + answer fits easily
    load_in_4bit = True,
)
model = FastLanguageModel.get_peft_model(
    model, r = 64, lora_alpha = 64,
    target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing = "unsloth",    # the thing that makes 14B fit
)
# TrainingArguments:
#   per_device_train_batch_size = 1
#   gradient_accumulation_steps = 16          # effective batch 16
#   num_train_epochs = 3
#   learning_rate = 1e-4, cosine schedule, warmup_ratio = 0.05
#   bf16 = True, optim = "paged_adamw_8bit"
```
If you OOM at 14B despite this: drop `max_seq_length` to 1536, then r to 32,
then (last resort) switch base to Qwen2.5-7B — in that order.

## 4. Merge + quantize (the "bigger then quantize" step)

```bash
# merge LoRA into fp16 weights (needs disk, not VRAM — streams to disk)
python3 -c "from unsloth import FastLanguageModel; \
  m,t = FastLanguageModel.from_pretrained('outputs/checkpoint-final'); \
  m.save_pretrained_merged('moat-14b-merged', t, save_method='merged_16bit')"

# quantize to GGUF with llama.cpp
python3 llama.cpp/convert_hf_to_gguf.py moat-14b-merged --outfile moat-14b-f16.gguf
llama.cpp/llama-quantize moat-14b-f16.gguf moat-14b-Q5_K_M.gguf Q5_K_M   # ~9.9GB
# fallback if you want more free VRAM while it runs alongside other apps:
llama.cpp/llama-quantize moat-14b-f16.gguf moat-14b-Q4_K_M.gguf Q4_K_M   # ~8.4GB
```

## 5. Evaluate before trusting it

Run the held-out set and check, in order of importance:
1. **Contrastive rows downgrade.** If corrupted-MSFT still comes back WIDE, the model
   memorized tickers → retrain with more contrastive repeats (×6) and fewer epochs (2).
2. **Held-out gold verdicts match Adam** (META wide 9, TSLA narrow, ZM narrow 6,
   INTC lost-moat, AMZN wide-but-weak with the segment split stated).
3. **Format compliance** — every answer in the mandated block format.
4. **No invented numbers** — every figure cited must appear in the evidence card.

## 6. Wire into the scanner

Inference: `llama-server -m moat-14b-Q5_K_M.gguf -ngl 99 -c 2048` → the Streamlit
moat expander POSTs the evidence card to `localhost:8080/v1/chat/completions` with
the rubric system prompt. The evidence card the scanner already renders per ticker
IS the model's input format — zero glue code beyond the HTTP call.

---
## File map
- `adam_seed_labels.json` — 40 gold verdicts, verbatim from the doc, with line refs
- `rubric_system_prompt.md` — Adam's rubric as the system prompt (consensus forbidden)
- `build_dataset.py` — builds gold/contrastive/silver_prompts (run: `python3 -m ai_moat.build_dataset`)
- `dataset/` — the JSONL outputs (chat-messages format, axolotl/unsloth/llama-factory ready)
- `gen_silver.py` — teacher loop (LM Studio/OpenAI-compatible, resumable)
- `train_qlora.py` — full unsloth QLoRA script with built-in held-out eval

---

## PC Safety — researched, not guessed (2026-08-16)

**Question: can any of this crash/damage the PC? Short answer: it cannot damage
hardware; the only real risk is an out-of-memory (OOM) freeze, and every step
below has a specific mitigation. Sources: unsloth official requirements docs,
LM Studio docs/blog (v0.3.14 guardrails), r/LocalLLaMA OOM reports.**

### Teacher phase (Qwen2.5-32B Q3_K_M, 15.95GB in LM Studio)
- The GGUF does NOT fit 12GB VRAM; LM Studio splits it: ~10-11GB in VRAM,
  the remaining ~5-6GB + context (KV cache) in system RAM.
- **Researched requirement: ≥16GB free system RAM for the overflow; 32GB
  total strongly recommended** (LM Studio low-VRAM guidance). A reported
  failure mode on Reddit: RAM fills, whole system freezes for minutes before
  the app OOMs. Mitigations, in order:
  1. In LM Studio: Settings → Hardware → keep **"Model loading guardrails"
     ON** (default "Balanced") — LM Studio then refuses/auto-trims loads
     that would exceed memory instead of freezing the machine.
  2. Set **context length to 4096** when loading the model (our prompts are
     ~2-3k tokens; default 32k contexts balloon the KV cache into RAM).
  3. Close browsers/games while the overnight run goes (frees both RAM and
     VRAM).
  4. If your machine has 16GB RAM total (not 32): load with GPU offload
     slider reduced OR use a smaller teacher
     (Qwen2.5-32B **Q2_K** ~12.3GB, or 14B-Instruct Q6 as teacher-lite).
- Worst case if you ignore all this: a freeze + app crash, requiring a
  reboot. No data loss (gen_silver.py flushes after every answer and
  resumes), no hardware damage.

### Student phase (14B QLoRA in unsloth)
- **Official unsloth table: 14B QLoRA minimum = 8.5GB VRAM** — your 12GB
  card has ~3.5GB headroom. Our script already uses every documented
  OOM-avoidance lever: `load_in_4bit`, batch size 1 (unsloth's own #1 OOM
  fix), gradient accumulation 16, `use_gradient_checkpointing="unsloth"`,
  `paged_adamw_8bit` (spills optimizer state to RAM safely).
- If it still OOMs (long samples can spike KV memory): the built-in ladder
  is `--seq 3072` → `--rank 32` → `--base 7b` (7B QLoRA = 5GB min, trivial
  fit). CUDA OOM in PyTorch just kills the python process with an error —
  it does not hang the machine the way system-RAM exhaustion does.
- Disk: base model download ~9GB + adapter ~1GB + merged fp16 ~28GB +
  final GGUF ~10GB → **keep ~50GB free** during the merge step; the merged
  fp16 can be deleted right after quantization.

### Duration expectations (so nothing looks "stuck")
- Teacher: 443 prompts × 1-3 min ≈ one overnight run (resumable).
- Training: 3 epochs × ~500 effective rows on a 12GB card ≈ 2-5 hours.
- Quantization: ~20-40 min CPU-bound.
