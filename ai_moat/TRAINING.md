# Teaching the Moat-Reasoning Model — High-Quality, FAST Recipe
### Target hardware: RTX 5070 Ti 12GB VRAM + 16GB system RAM

**Why "teaching", not "training"** (your framing — adopted, it's more accurate):
we are NOT training a model from scratch (that takes trillions of tokens and a
datacenter). We take a model that already knows English, finance and reasoning,
and **teach it one specific discipline** — Adam's moat-analysis rubric — with a
small, deliberately-designed curriculum (~200 lessons): worked examples from the
teacher (gold), trick questions that punish memorization (contrastive), and
practice breadth (silver). The mechanism (QLoRA) literally freezes 99.5% of the
model and only adjusts small "habit" adapters — the model keeps everything it
knew and gains one skill. That is teaching in any meaningful sense of the word.

---

## 0. The decision (updated 2026-08-16 after the Qwen3.8 review)

| Choice | Decision | Why |
|---|---|---|
| Student (the model we teach) | **Qwen3-14B** (`unsloth/Qwen3-14B-unsloth-bnb-4bit`), fallbacks: Qwen2.5-14B, Qwen2.5-7B | Newest generation that BOTH unsloth QLoRA and llama.cpp GGUF support maturely. 14B is the largest QLoRA-teachable size on 12GB. |
| Teacher (labels the silver tier) | **Qwen3-14B GGUF Q4_K_M (~9.0GB)** in LM Studio — fits FULLY in 12GB VRAM → 30-50 tok/s | The old 32B Q3_K_M half-offloaded to RAM at 1-3 tok/s → your 16h ETA. A fully-in-VRAM 14B is **20-50× faster** and near-equal on a structured rubric task. |
| Quality-option teacher | Qwen3.8-27B UD-Q3_K_XL (unsloth GGUF exists, llama.cpp supports Gated DeltaNet) | ~14GB → partial offload on 12GB, ~5-10 tok/s. Use overnight only if you want maximum label quality. |
| Qwen3.8-27B as STUDENT | **REJECTED** | 27B 4-bit QLoRA needs 17-19GB — does not fit 12GB for teaching. Fine as a teacher via GGUF, wrong as a fine-tune target on this card. |
| Method | **QLoRA** rank 64 (4-bit frozen base + adapters) | Teaches ~0.5% of weights; fits 12GB with unsloth gradient checkpointing. |
| After teaching | **`python ai_moat/quantize_model.py`** — fully automatic | Merges the adapter → converts → quantizes to Q5_K_M GGUF (~9.9GB, fits fully in VRAM). One command; it even fetches llama.cpp itself. |

**Reasoning-model claim, honestly stated:** this is not an RL "reasoning model"
like o1. It's a supervised model whose curriculum *forces* reasoning:
- **Gold tier** anchors verdicts to Adam's actual judgments (no consensus anywhere).
- **Contrastive tier** shows the SAME famous ticker with corrupted fundamentals and
  a DOWNGRADED answer — the model cannot pass by memorizing "MSFT = wide"; it must
  read the evidence card.
- **Silver tier** teaches breadth — now **great-businesses only** (~137 prompts,
  not 443): the moat model's job is grading companies that already pass the
  great-business scan; labelling the ~300 rejects wasted GPU hours for nothing.

---

## 1. Generate the silver tier (NOW ~1-1.5 HOURS, not 16)

**Setup in LM Studio (one time):**
1. Search & download: **`Qwen3-14B GGUF`** (lmstudio-community or unsloth), pick **Q4_K_M** (~9.0GB).
2. Load it with: GPU offload = **MAX**, context = **4096** (no more — long context steals VRAM), "keep model in RAM" **OFF**.
3. Developer tab → **Start Server** (default `http://localhost:1234`).

**Run (from the repo root):**
```bash
python ai_moat/gen_silver.py --limit 5     # ALWAYS smoke-test 5 first
python ai_moat/gen_silver.py               # full run: great-only, ~137 prompts
```
What makes it fast now (all default, no flags needed):
- teacher fully in VRAM (the 20-50× lever),
- great-businesses-only prompt set (~137 vs 443),
- thinking mode suppressed (Qwen3 thinks by default — hundreds of hidden tokens per answer; `--thinking` re-enables if you ever want it),
- 2 parallel in-flight requests (`--workers 1` if LM Studio misbehaves),
- 800-token answer cap.

Progress checkpoints after EVERY answer — stop/re-run anytime, it resumes.

Then **human review**: open `dataset/silver.jsonl`, check ~14 random rows (10%).
Delete rows whose verdict contradicts the evidence card. Highest-leverage
half hour in the whole pipeline.

## 2. The curriculum mix (automatic in train_qlora.py)

```
gold.jsonl         × 6 repeats   (42 → ~220 rows; the anchors)
contrastive.jsonl  × 4 repeats   (26 → ~90 rows; anti-memorization)
silver.jsonl       × 1           (~120 rows after review, great-only)
                                 ≈ 430 teaching rows
HELD OUT, never taught: gold META/TSLA/AMZN/INTC/ZM + contrastive MSFT/GOOGL/AAPL
```
The held-out set is the trust gate: the model must reproduce Adam's verdicts on
gold it never saw AND downgrade corrupted versions of famous names.

## 3. Teach (unsloth QLoRA)

```bash
pip install unsloth                       # one time
python ai_moat/train_qlora.py             # Qwen3-14B default, ~2-4h
python ai_moat/train_qlora.py --eval-only # re-run the trust gate later
```
OOM ladder (apply in order): `--seq 1536` → `--rank 32` → `--base 7b`.

## 4. Quantize — FULLY AUTOMATIC now

```bash
python ai_moat/quantize_model.py          # merge → GGUF → Q5_K_M, one command
```
- Clones llama.cpp itself, installs converter deps, finds or builds `llama-quantize`.
- Windows easiest path if the auto-build can't find a compiler: download a
  release zip from https://github.com/ggml-org/llama.cpp/releases
  (`llama-bXXXX-bin-win-cuda-x64.zip`), then:
  `python ai_moat/quantize_model.py --llama-bin C:\path\to\llama-quantize.exe`
- Disk: ~56GB peak briefly, intermediates auto-deleted.
- Output: `ai_moat/outputs/moat-qwen3-14b-Q5_K_M.gguf` (~9.9GB — fits fully in
  your 12GB VRAM → 30-50 tok/s in LM Studio).

## 5. Evaluate before trusting it

The teach run prints the held-out eval automatically. Verify by hand:
- gold rows ≈ Adam's verdict (grade within one notch),
- contrastive rows MUST downgrade (this is the whole point),
- reasoning cites the evidence card, not fame.
If contrastive rows don't downgrade → more contrastive repeats (edit `_repeat`)
or lower LR to 5e-5 and re-teach. Do NOT ship a model that fails the gate.

## 6. Wire into the scanner

Same OpenAI-compatible call as gen_silver uses — point it at your own model in
LM Studio and batch-grade the great-business list.

## File map
```
ai_moat/
  build_dataset.py      # builds gold/contrastive/silver_prompts (great-only default)
  gen_silver.py         # FAST teacher labelling (great-only, no-think, parallel)
  train_qlora.py        # QLoRA teaching run + held-out trust gate
  quantize_model.py     # automatic merge → GGUF → Q5_K_M
  rubric_system_prompt.md
  adam_seed_labels.json # Adam's own verdicts (gold source)
  dataset/              # the curriculum
  outputs/              # adapters + final GGUF
```

## PC Safety — researched, not guessed (2026-08-16)

### Teacher phase (Qwen3-14B Q4_K_M fully in VRAM)
- ~9.0GB model + ~0.7GB KV cache at ctx 4096 ≈ 9.7GB of 12GB — safe.
- GPU will run hot (it's compute-bound now, not RAM-bound) — normal.
- If Windows starts swapping: close browsers; LM Studio itself stays ~1GB.

### Student phase (14B QLoRA in unsloth)
- 4-bit base ≈ 8.2GB + adapters/optimizer/activations ≈ 3.3GB → ~11.5GB: tight
  but fits with unsloth gradient checkpointing (enabled in the script).
- First OOM → `--seq 1536`; second → `--rank 32`; last resort `--base 7b`.

### Quantize phase
- CPU/disk-bound, no VRAM needed. Needs ~56GB free disk briefly.

### Duration expectations (so nothing looks "stuck")
| Step | Time |
|---|---|
| gen_silver (137 great-only, 14B in VRAM) | **~1-1.5h** (was 16h+) |
| teach 3 epochs | 2-4h |
| merge + convert + quantize | 30-60min |
