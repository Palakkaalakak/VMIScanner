# 🧠 THE AI SUPERGUIDE — Teach Your Own Moat-Analysis AI
### Written for a complete beginner. Zero AI knowledge assumed.

> **What you'll have at the end:** your own private AI, running on YOUR
> PC, free forever, that analyses company moats the way Adam Khoo does —
> because YOU taught it with the rubric and examples in this repo.

---

## Part 0 — The big picture (read this first)

### What are we actually doing?

We are **teaching**, not training. (Your framing — and it's the correct
one.) Here's the difference in plain words:

- **Training** a model from scratch = raising a baby. Needs trillions of
  words and a datacenter. Not us.
- **Teaching** (what we do) = hiring a smart graduate who already speaks
  English and knows finance, and giving them ~200 worked lessons on ONE
  discipline: Adam's moat-analysis method. The graduate is a free
  open-source model; the lessons are our dataset; the "teaching session"
  is a few hours on your gaming GPU.

Technically this is called **QLoRA fine-tuning**: 99.5% of the model's
brain is frozen; we only adjust tiny "adapter" layers bolted on the side.
That's why it fits on a 12GB graphics card instead of a datacenter.

### The cast of characters

| Character | Who it is | Where it runs |
|---|---|---|
| **The Teacher** | Qwen3-14B (Q4_K_M quant, ~9GB) — an already-smart model | Your PC, inside LM Studio |
| **The Textbook** | Our dataset: gold answers (Adam's own verdicts), contrastive pairs (right-vs-wrong), silver lessons (teacher-written) | This repo, `ai_moat/dataset/` |
| **The Student** | Qwen3-14B (fresh copy) — learns the moat discipline | Your PC, via `train_qlora.py` |
| **The Exam** | Held-out companies the student never saw during teaching | Automatic, inside the training script |
| **The Graduate** | `moat-qwen3-14b-Q5_K_M.gguf` — your finished moat AI | LM Studio, chat with it forever |

### Words you'll see (dummy dictionary)

- **Model / LLM** — the AI file. Literally just a big file of numbers.
- **Quantization / quant** — shrinking that file by storing the numbers
  less precisely. Like JPEG for AI. **Q4/Q5 = visually-lossless JPEG**
  (proven indistinguishable in independent tests). Q2 = potato quality.
- **GGUF** — the file format LM Studio and llama.cpp use. Like .mp4 for AI.
- **VRAM** — your graphics card's memory. You have **12GB**. Everything
  in this guide is sized to fit it.
- **Token** — roughly ¾ of a word. "tok/s" = how fast the AI writes.
- **LoRA adapter** — the small "what I learned" file the teaching
  produces (~200MB). Gets merged into the big model at the end.
- **LM Studio** — a free desktop app that runs AI models with a nice UI.
  Think "VLC player, but for AI models".

### Why these models? (the research, honestly)

We checked everything runnable on a 12GB card, **including quantized
versions of the big new models**. The evidence (independent Quesma study
on the 27B Qwen class: KL divergence, AIME math, blind quality duels):

- **4-bit and above: statistically indistinguishable from the full
  model.** This is the safe zone.
- 3-bit: coin flip (one variant scored 73% on math, its sibling 54%).
- 2-bit: measurably worse (lost ~19 of 20 blind quality duels).

| Candidate | Verdict |
|---|---|
| **Qwen3-14B @ Q4_K_M (~9GB)** | ✅ **WINNER.** Only model that is 4-bit+ (safe zone), fully in your VRAM (fast: 30-50 tok/s), AND teachable via mature QLoRA tooling. |
| Qwen3.8-27B @ 4-bit | ❌ Needs 17-19GB. Doesn't fit. |
| Qwen3.8-27B @ 2-3-bit | ⚠️ Fits (~11-14GB total) but 2-bit quality drop is real, and it's **not teachable** on 12GB regardless. Optional slow teacher only. |
| Qwen3.6-35B-A3B (MoE) @ 2-bit | 🧪 Interesting teacher upgrade IF you have ≥16GB system RAM (MoE offloads cheaply). Experimental; not the default. |

---

## Part 1 — One-time PC setup (~30 minutes)

### Step 1.1 — Install LM Studio
1. Go to **https://lmstudio.ai** → Download for Windows → install → open.

### Step 1.2 — Download the Teacher model
1. In LM Studio, click the 🔍 **Discover/Search** icon (left sidebar).
2. Search: **`Qwen3 14B`**
3. Pick the entry from **unsloth** or **lmstudio-community**.
4. In the file list choose **`Q4_K_M`** (~9.0GB). Click Download.
   - ❗ NOT Q8 (too big for VRAM → slow). NOT Q2/Q3 (dumb).

### Step 1.3 — Load it with the right settings
1. Click the 💬 **Chat** icon → top bar → select the Qwen3-14B model.
2. Before/while loading, open its settings gear:
   - **GPU Offload: MAX** (all layers). This is the single most
     important setting. If it's partial, you get 1-3 tok/s (the old
     16-hour problem). Full = 30-50 tok/s.
   - **Context Length: 8192** — important! LM Studio SPLITS the context
     between parallel requests, and gen_silver uses 2 workers:
     8192 / 2 = 4096 per worker, comfortable. (4096 total → 2048 each →
     "Context size has been exceeded" crashes mid-answer.)
3. Say "hi" to it. If it answers fast, you're done here.

### Step 1.4 — Turn on the local server
1. Click the ↔️ **Developer / Local Server** icon (left sidebar).
2. Click **Start Server**. Note the address — normally
   `http://localhost:1234`. Leave LM Studio running.

### Step 1.5 — Get the repo + Python on your PC
```powershell
# In PowerShell:
git clone https://github.com/Palakkaalakak/VMIScanner.git
cd VMIScanner
# Python 3.10+ from python.org if you don't have it (tick "Add to PATH")
pip install requests
```

### Step 1.6 — (optional) HuggingFace token for faster downloads
A free HF token lifts rate limits on the one-time model downloads.
Drop it in a git-ignored file — the scripts pick it up automatically:
```powershell
Set-Content ai_moat/.hf_token "hf_your_token_here"
```
That file never leaves your PC (it's in .gitignore).

### Updating the project later (NO re-clone needed!)
Whenever I push fixes, just run this inside the VMIScanner folder:
```powershell
git pull
```
That's it — git downloads only what changed (seconds, not minutes) and
keeps your local files: `.hf_token`, generated datasets, trained
adapters, GGUFs. If git ever complains about local changes:
```powershell
git stash        # shelve your local edits
git pull         # update
git stash pop    # put your edits back (optional)
```
Never delete + re-clone — you'd lose your outputs folder.

✅ **Setup done. You never repeat Part 1.**

---

## Part 2 — The pipeline, step by step

Think of it as a school year: 📚 write the textbook → 🏫 teach →
🎓 examine → 📦 package the graduate.

### Step 2.0 — Fresh scan (usually already done)
The dataset is built from the scanner's verdicts, so results must be
current. In the sandbox this runs automatically; the repo's
`public/data/scan_results.json` is kept fresh (540 companies incl. banks
& REITs since 2026-08-16). **You can skip this step on your PC.**

### Step 2.1 — Build the textbook skeleton (seconds, your PC or sandbox)
```powershell
python -m ai_moat.build_dataset
```
What it makes in `ai_moat/dataset/`:
- **gold.jsonl** — Adam's own verdicts with evidence cards. Ground truth.
- **contrastive.jsonl** — right-vs-wrong pairs ("THIS is a moat, THAT
  isn't"). Sharpens boundaries.
- **silver_prompts.jsonl** — empty worksheets for the Teacher to fill,
  **great businesses only** (~155, not all 540 — quality over noise).

### Step 2.2 — The Teacher writes the silver lessons (~1-1.5 hours)
LM Studio running, server on, then:
```powershell
python ai_moat/gen_silver.py
```
- It talks to LM Studio at `localhost:1234`, has the Teacher fill each
  worksheet, and saves after EVERY answer — **interrupting is safe**,
  it resumes where it left off.
- Speed levers already built in: thinking-mode off, 2 parallel workers,
  800-token cap, great-only. This used to take 16h+; now ~1-1.5h.
- Options: `--limit 5` (quick test first — DO THIS ONCE before the full
  run), `--all` (include non-great companies), `--workers 1` (if your
  PC struggles).

### Step 2.2b — Automated quality check (~10-20 min, NO manual review)
The teacher grade-inflates: it hands out 9/10 and 10/10 too easily. In
Adam's framework those scores are reserved for near-monopolies — very
few businesses deserve them. With LM Studio still running:
```powershell
python ai_moat/review_silver.py
```
- Finds every answer scored **above 8/10** and sends it back to the
  teacher in **skeptical second-reviewer mode**: justify the 9-10 from
  the evidence card alone, or downgrade.
- The reviewer can confirm or LOWER a score — never raise it.
- Originals are backed up to `silver.pre_qc.jsonl`; progress checkpoints
  after every review, so interrupting is safe.
- It prints the score distribution before/after so you can see the
  inflation get corrected. That's your quality check — no file-skimming
  needed.

**Already generated silver.jsonl with an older version of the scripts?**
No regeneration needed — `review_silver.py` works directly on your
existing file. Just `git pull` and run it. Same for every step: each
script reads whatever the previous step produced, old version or new.

### Step 2.3 — Teach the Student (~1-2 hours, GPU fans will spin)
```powershell
pip install unsloth
python ai_moat/train_qlora.py
```
- Downloads the student (Qwen3-14B 4-bit, ~9GB, one time), then teaches
  it the curriculum: gold (weighted highest) + contrastive + silver.
- **FIRST: eject the Teacher in LM Studio** (or quit LM Studio). It holds
  ~9GB of VRAM; teaching needs the card to itself. The script checks and
  refuses to start if VRAM is occupied.
- **The exam is automatic:** some companies are held OUT of teaching and
  used as a test. The script prints held-out accuracy at the end.
  - **Rule of thumb: below ~70% held-out agreement → don't trust it, ask
    me and we tune. Above → proceed.**
- Output: `ai_moat/outputs/moat-qwen3-14b-lora/` (the adapter, ~200MB).
- Close games/browsers with video playing — they steal VRAM.

### Step 2.4 — Package the graduate (~30-60 min, ONE command)
```powershell
python ai_moat/quantize_model.py
```
Fully automatic: merges adapter into the base → converts to GGUF →
quantizes to **Q5_K_M (~9.9GB)** → deletes the huge intermediates.
Best practice is built in: at Q4-or-below it auto-computes an
**importance matrix calibrated on OUR OWN moat dataset** (protects
exactly the weights that matter for moat analysis). Q5 skips it because
Q5 is already in the indistinguishable-from-full zone — saves 30 min.

- Needs ~56GB free disk briefly. `--quant Q4_K_M` for a smaller file.
- Windows compiler issues? Download a llama.cpp release zip from
  github.com/ggml-org/llama.cpp/releases and re-run with
  `--llama-bin C:\path\to\llama-quantize.exe` — the script tells you this
  too if it gets stuck.
- Result: **`ai_moat/outputs/moat-qwen3-14b-Q5_K_M.gguf`**

### Step 2.5 — Chat with your own moat AI (forever, free)
1. LM Studio → My Models → import the `.gguf` (or drop it into the LM
   Studio models folder).
2. Load it: GPU Offload MAX, context 4096.
3. Set the **system prompt** to the contents of
   `ai_moat/rubric_system_prompt.md` (this puts it in "moat analyst
   mode").
4. Ask: *"Analyse the moat of Costco"* — it should answer in Adam's
   framework: moat type, evidence, durability, verdict.

---

## Part 3 — When things go wrong (they will, it's fine)

| Symptom | Cause | Fix |
|---|---|---|
| gen_silver: "connection refused" | LM Studio server not started | Developer tab → Start Server |
| gen_silver: HTTP 400 on every ticker | Model not loaded into the SERVER, or just-started server still warming up | Developer/Server tab → make sure the model is selected/loaded THERE (not only in the Chat tab), then re-run |
| "Context size has been exceeded" | Context split across workers: 4096/2 = 2048 each, too small | Eject model → reload with Context Length 8192; or `--workers 1` |
| EVERY ticker "failed the format gate" / SKIPPED | LM Studio ignored our no-thinking API flag: the model spends its whole 800-token budget "thinking" (LM Studio hides it in `reasoning_content`), the visible answer stays empty, so the format check correctly fails | `git pull` — the script now also appends Qwen3's in-band `/no_think` switch to every prompt, which the model itself obeys regardless of API flags. If it STILL happens: click the gear icon next to the loaded model in LM Studio and switch Reasoning OFF |
| Teacher crawls at 1-3 tok/s | GPU Offload partial | Set to MAX; close VRAM-hungry apps; reload model |
| train_qlora: "Some modules are dispatched on the CPU" | Not enough free VRAM for the student. Either LM Studio still holds the Teacher (~9GB), OR you hit the old oversized default: the "unsloth-dynamic" 14B repo is 10.36GB of weights — too big even on an empty 12GB laptop card | `git pull` (default now points at the standard 9.25GB repo, which fits). Eject any LM Studio model, close video tabs, re-run. Still failing? `python ai_moat/train_qlora.py --base qwen3-8b` — guaranteed fit |
| train_qlora: "Pickler._batch_setitems() takes 2 positional arguments but 3 were given" | Python 3.14 changed an internal pickle API; the `dill` library (used by the datasets tool) hasn't caught up yet | `git pull` — the script now auto-patches this at startup (you'll see "py3.14 compat: patched _batch_setitems"). No action needed |
| "CUDA out of memory" during teaching | Something else is using VRAM | Close browsers/games; retry. Persists → tell me, we add `--batch-size 1` |
| Teaching loss not going down | Data/config issue | Screenshot the numbers, send to me. **Don't guess.** |
| Held-out accuracy < 70% | Model didn't learn well enough | Don't ship it. We add lessons or tune together |
| quantize: "llama-quantize not found" | No C++ compiler | Release-zip + `--llama-bin` trick above |
| Answers are generic finance-blah | Forgot the system prompt | Step 2.5.3 |
| Anything interrupted | — | Everything checkpoints. Just re-run the same command |

---

## Part 4 — Iron rules (same as the scanner's)

1. **No invented numbers.** If the AI states a figure not in the
   evidence card, that lesson is wrong — flag it, we fix the dataset.
2. **Your attestation of Adam's views overrides everything**, master doc
   included. If the AI contradicts what you know Adam teaches, it's the
   AI that's wrong.
3. **The AI assists, the checklist decides.** is_great comes from the
   scanner's mechanical checks; the AI explains and reasons about moats.
   Never let it overrule a failed check.
4. **Trust gate before use.** No held-out exam pass → no real-money use.

## Part 5 — Timings & requirements at a glance

| Step | Command | Time | Needs |
|---|---|---|---|
| Textbook | `python -m ai_moat.build_dataset` | seconds | nothing |
| Silver lessons | `python ai_moat/gen_silver.py` | ~1-1.5h | LM Studio running |
| Teaching | `python ai_moat/train_qlora.py` | ~1-2h (packing on) | 12GB GPU, ~20GB disk |
| Packaging | `python ai_moat/quantize_model.py` | ~30-60min | ~56GB disk briefly |
| **Total** | | **~2.5-4h, hands-on time ~15 min** | |

### ⚡ Fast path (if you're impatient)
- **Your attention is only needed for ~15 minutes total** — the rest is
  the machine working while you do something else. Start gen_silver
  before dinner, start teaching before bed, quantize with morning coffee.
- `python ai_moat/train_qlora.py --epochs 2` — cuts teaching ~33%;
  fine if held-out accuracy still passes (the exam tells you).
- `python ai_moat/quantize_model.py --quant Q4_K_M` is NOT faster
  (it adds the imatrix pass) — Q5_K_M default is already the quick one.
- gen_silver `--workers 3` if your GPU isn't maxed (watch tok/s; if it
  drops per-worker, go back to 2).
- What NOT to cut: the held-out exam, the gold tier, or the system
  prompt. Those are the difference between an analyst and a parrot.

*Companion docs: `TRAINING.md` (technical detail), `rubric_system_prompt.md`
(the rubric itself). This guide supersedes nothing — it's the map.*
