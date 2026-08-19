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
- **FIRST: eject the Teacher in LM Studio** (or quit LM Studio). It holds
  ~9GB of VRAM; teaching needs the card to itself.
- The script **measures your free VRAM and auto-picks the student size**
  that actually fits *training* (which needs weights + gradients +
  activations — far more than just loading). On a 12GB laptop card the
  winner is **Qwen3-8B**: newest generation, fully-in-VRAM, ~3GB of
  headroom. A 14B student needs ~12GB free — physically impossible here
  once Windows takes its ~1GB. 8B learns a fixed rubric format nearly
  as well; this is the honest right size, not a downgrade to worry about.
- Downloads the student once (~5.7GB for 8B), then teaches it the
  curriculum: gold (weighted highest) + contrastive + silver.
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
   Studio models folder). If LM Studio doesn't see it, the models folder
   needs the layout `models/<anything>/<anything>/model.gguf` — e.g.
   `models/you/moat/moat-qwen3-8b-Q5_K_M.gguf`.
2. Load it: GPU Offload MAX, **context 8192** (default henceforth — the
   rubric + evidence card + live-research addendum need the room; the
   8B at Q5_K_M leaves plenty of VRAM for it). (The Q5_K_M 8B file is
   ~5.7GB — fits in ~7GB VRAM with context.)
3. Set the **system prompt** to the contents of
   `ai_moat/rubric_system_prompt.md` (this puts it in "moat analyst
   mode"). Without it you get generic finance-blah.
4. If the model "thinks" forever before answering: click the gear icon
   next to the loaded model → Reasoning OFF (or end your question with
   `/no_think`).
5. Ask: *"Analyse the moat of Costco"* — it should answer in Adam's
   framework: moat type, evidence, durability, verdict.

### Step 2.6 — Use it INSIDE the dashboard (AI Moat Evaluator tab)
The dashboard now has a **🤖 AI Moat Evaluator** top-level tab that runs
your trained model automatically — no copy-pasting prompts:

1. LM Studio → **Developer tab → Start Server** (default port 1234), and
   load the moat model INTO THE SERVER (select it in the Developer tab).
2. Get a dashboard that can REACH LM Studio — two ways:
   - **Local (simplest):** double-click `run_dashboard.bat` in the repo
     folder (or `streamlit run streamlit_app.py`). localhost:1234 just
     works because both run on the same PC.
   - **Hosted on share.streamlit.io:** the cloud server cannot see your
     PC's localhost, so expose LM Studio with a tunnel:
     `cloudflared tunnel --url http://localhost:1234`
     (download cloudflared.exe from developers.cloudflare.com, no
     account needed) → it prints a `https://….trycloudflare.com` URL →
     paste that URL **+ `/v1`** into the tab's server-URL box.
     ⚠️ Anyone who has the tunnel URL can use your model while the
     tunnel runs — close the cloudflared window when done.
3. Open the tab — it auto-detects the server and auto-picks any model
   with "moat" in its name (the quantize step names it that way). Green
   banner = your trained model is live.
4. Evaluate one ticker (scanned tickers get the full quantitative
   evidence card — the exact prompt format it was trained on) or hit
   **⚡ Evaluate all GREAT stocks** for a batch run.
5. Every verdict/score is saved to
   `public/data/moat_ai_evaluations.json` and shows up as
   **AI Moat verdict** / **AI Moat score /10** columns in the Scanner
   tab, with dedicated filters (verdict multiselect, min-score slider,
   "only evaluated" toggle). Unevaluated tickers are never silently
   dropped — NA never disqualifies.

**🔍 Live research (staying up to date)**: the Evaluate panel has a
"Live research" toggle (ON by default for single tickers). The dashboard
fetches CURRENT fundamentals (gross/operating/net margin, ROE, revenue
and earnings growth, P/E) plus the latest headlines from Yahoo Finance
and appends them to the evidence card before asking the model. This is
deliberate architecture: **deterministic code does the research, the
model stays a pure judge.** We did NOT teach the model to make its own
tool calls — an 8B QLoRA is excellent at judging evidence it's handed
but unreliable at orchestrating searches, and letting it fetch its own
facts would reintroduce the hallucination risk the evidence card exists
to prevent. Same reason we don't bolt on a second "tool-calling AI"
(e.g. Cactus Needle — a 14MB function-calling model for phones/wearables):
our tools are known in advance (Yahoo lookup), so a Python `if` is
more reliable than any model deciding which tool to call. No numbers
are invented: only fields Yahoo actually returns are included.

**Sharing the model with other people**: the `.gguf` is ~5.7GB, far over
GitHub's 100MB file limit — do NOT `git add` it (outputs/ is
git-ignored anyway). The standard free way is a Hugging Face model repo:
create an account → New Model → upload the `.gguf` (they host multi-GB
GGUFs for free; that's where LM Studio downloads models from). Anyone
then: downloads it, imports into LM Studio, clones this repo, and
follows this Step 2.6. Needs a GPU with ~7GB+ free VRAM for smooth use.

### Step 2.7 — Teach it to RESEARCH ITSELF (tool calling, ~20-40 min)
Your model has a knowledge cutoff. This quick top-up teaches it to CALL
TOOLS so it can fetch fresh facts forever — and it is NOT a retrain:
it **continues from the adapter you already trained**, adding ~73 small
tool-use lessons (~9-12 steps ≈ **20-40 minutes**, vs hours for full
training). It also bakes in the corrected TSLA key-man label (Elon Musk
named explicitly, repeated x4) — so one pass fixes both gaps.

What it learns:
- no evidence card in the prompt → call `research_stock(ticker)` first
- "recent/current events" questions → call `web_search(query)`
- evidence card already provided → answer directly, NO tool spam
- every final answer is still Adam's real verdict — nothing invented

Commands (eject LM Studio's model first, same as training):
```
git pull
python -m ai_moat.build_tool_dataset      # seconds
python ai_moat/teach_tools.py             # ~20-40 min
python ai_moat/quantize_model.py          # auto-picks the new -tools adapter
```
Output: a NEW adapter `outputs/moat-<base>-tools-lora` — your original
judge adapter is never touched. teach_tools ends with a smoke test that
prints whether the model actually emits a `<tool_call>` (should say
"YES ✅"). The quantized file is named `moat-<base>-tools-Q5_K_M.gguf`.

Then in the dashboard's AI Moat Evaluator you get a **Research mode**
choice:
- **🔍 Dashboard research** — Python fetches Yahoo data, model judges
  (always works, even with the plain adapter)
- **🤖 Self-research agent** — the MODEL decides what to research and
  calls research_stock / web_search itself via LM Studio's tool-calling
  API; the dashboard executes the calls and shows you exactly which
  tools it used. If it answers without calling any tool, you're probably
  running the plain adapter — load the -tools GGUF.

Why we didn't bolt on Cactus Needle: it's a 14MB tool-ROUTER for
phones/wearables — great when tools are unknown ahead of time. Ours are
fixed (2 tools), so teaching YOUR model to call them directly is
simpler, and the executor stays deterministic Python either way.

### Step 2.8 — How to READ the held-out eval (don't panic!)
The eval prints two kinds of rows — they are graded OPPOSITE ways:

- **[gold] rows** (META, TSLA, AMZN, INTC, ZM): real evidence, the model
  should match Adam's actual verdict shown on the `>>> EXPECTED:` line.
  TSLA and ZM are IN the gold set on purpose — they're Adam's own
  low/narrow examples (TSLA: failed pricing-power test, no-moat auto
  industry; ZM: "didn't have a very strong moat… easily disrupted",
  6/10, NOT bought). A rubric that only contains 9/10 near-monopolies
  would teach the model that everything is WIDE.
- **[contrastive] rows** (MSFT, GOOGL, AAPL): the evidence card is
  **DELIBERATELY FALSIFIED** — we inject fake margin collapse (~12pp
  gross-margin erosion, ~9pp operating-margin drop, sub-par ROIC) into
  famous names. The CORRECT answer is a LOW score (~NARROW 4/10). If the
  model says "MSFT is wide, everyone knows that", it failed — it's
  reciting reputation instead of reading evidence. A downgrade here is
  the model PASSING its hardest test. The eval output now labels these
  rows "(FAKE degraded evidence — low score = PASS)" so this can't be
  misread again.
- The `max_new_tokens`/`max_length` warning is cosmetic — our 900-token
  limit simply takes precedence. Ignore it.

---

## Part 3 — When things go wrong (they will, it's fine)

| Symptom | Cause | Fix |
|---|---|---|
| gen_silver: "connection refused" | LM Studio server not started | Developer tab → Start Server |
| gen_silver: HTTP 400 on every ticker | Model not loaded into the SERVER, or just-started server still warming up | Developer/Server tab → make sure the model is selected/loaded THERE (not only in the Chat tab), then re-run |
| "Context size has been exceeded" | Context split across workers: 4096/2 = 2048 each, too small | Eject model → reload with Context Length 8192; or `--workers 1` |
| EVERY ticker "failed the format gate" / SKIPPED | LM Studio ignored our no-thinking API flag: the model spends its whole 800-token budget "thinking" (LM Studio hides it in `reasoning_content`), the visible answer stays empty, so the format check correctly fails | `git pull` — the script now also appends Qwen3's in-band `/no_think` switch to every prompt, which the model itself obeys regardless of API flags. If it STILL happens: click the gear icon next to the loaded model in LM Studio and switch Reasoning OFF |
| Teacher crawls at 1-3 tok/s | GPU Offload partial | Set to MAX; close VRAM-hungry apps; reload model |
| train_qlora: "Some modules are dispatched on the CPU" or "No or negligible GPU memory available for fused cross entropy" | TRAINING needs far more VRAM than loading: weights + gradients + activations + loss buffers. A 14B student needs ~12GB FREE — a Windows 12GB laptop card (desktop eats ~1GB) never has that, even with LM Studio closed | `git pull` — the script now MEASURES your free VRAM and auto-picks the largest student that actually fits training (on this laptop: Qwen3-8B). Just run `python ai_moat/train_qlora.py` with no flags |
| train_qlora: any `_batch_setitems` / pickling error ("takes 2 positional arguments but 3 were given", "missing 1 required positional argument: 'obj'") | Python 3.14 changed an internal pickle API; dill/datasets haven't caught up | `git pull` — the script auto-patches at startup (you'll see "py3.14 compat: datasets Pickler (adaptive); Hasher.hash failsafe"). The failsafe means fingerprinting can NEVER crash training again — worst case it just disables dataset caching, which we don't use |
| Training crawls (10+ min per step, ~20h+ estimate, whole PC sluggish) | NVIDIA's Windows driver silently spills GPU memory into system RAM over PCIe ("sysmem fallback") instead of erroring — training "works" but 10-30x slower | Ctrl+C. **NVIDIA Control Panel → Manage 3D Settings → "CUDA - Sysmem Fallback Policy" → "Prefer No Sysmem Fallback"**. Close Chrome. Re-run. The script now warns you automatically after 2 slow steps. Expected healthy speed: ~1-3 min/step, ~1.5-3h total |
| "CUDA out of memory" during teaching | Something else is using VRAM | Close browsers/games; retry. Persists → tell me, we add `--batch-size 1` |
| PC shut down / crashed during or right after training | The script's order is: train (hours) → **save adapter (seconds)** → save tokenizer (seconds) → held-out eval (minutes). If the progress bar finished, the adapter was almost certainly saved — usually only the eval got skipped. And a checkpoint is written at EVERY epoch boundary (steps 30/60/90 on defaults), so even a mid-training crash loses at most part of one epoch | `git pull`, then `python ai_moat/check_training.py` — it inspects `outputs/`, reports COMPLETE / RECOVERED / NOTHING, auto-promotes the newest checkpoint to final if the final save was cut off, and prints your exact next commands. Then run `python ai_moat/train_qlora.py --eval-only` (the skipped trust-gate eval) and `python ai_moat/quantize_model.py` |
| Teaching loss not going down | Data/config issue | Screenshot the numbers, send to me. **Don't guess.** |
| Held-out accuracy < 70% | Model didn't learn well enough | Don't ship it. We add lessons or tune together |
| quantize: "llama-quantize not found" | No C++ compiler | Release-zip + `--llama-bin` trick above |
| teach_tools: `ImportError: cannot import name 'StepSpeedSentinel' from 'train_qlora'` | Old code version — the sentinel used to live *inside* another function, so it couldn't be imported. Fixed: it's now a `make_step_speed_sentinel()` factory | `git pull`, then re-run `python ai_moat/teach_tools.py` |
| After running quantize, streamlit/pandas break with "protobuf 4.25.9 is incompatible" / "numpy 1.26.4 is incompatible" | An older version of quantize_model.py pip-installed llama.cpp's requirements file, which *downgrades* numpy and protobuf. Fixed: it now installs only `gguf` + tokenizer helpers with `--upgrade-strategy only-if-needed` | Repair your environment once: `python -m pip install -U numpy "protobuf>=5.26,<7"` (the version cap matters — protobuf 7.x is *too new* for opentelemetry, which streamlit pulls in) — then `git pull` so it never happens again |
| quantize says `[plain judge]` but you wanted `[tool-calling]` | The `-tools-lora` adapter doesn't exist yet (teach_tools crashed or hasn't run). The detection is telling the truth, not misbehaving | Run `python ai_moat/teach_tools.py` successfully first, THEN re-run quantize — it will automatically pick up the newer `-tools-lora` adapter and name the GGUF `moat-…-tools-….gguf` |
| Every wide-moat stock gets the same 9/10 (AAPL == ABBV) | Score saturation: the model grades each stock in isolation and the top of the scale compresses — an unshakeable multi-source moat and a strong single-source moat both landed at 9. Fixed in the rubric ("SCORE CALIBRATION WITHIN WIDE"): 10 = redundant 4–5-source moats (Heavenly-Queen grade), hard cap of 8 when the moat rests primarily on one source (e.g. pharma patents — Adam: "it doesn't last forever"). The dashboard re-reads the rubric from disk on every evaluation, so this improves immediately WITHOUT retraining | `git pull`, then just re-evaluate. Delete the old saved evals for affected tickers first so the new scores replace them |
| During GGUF conversion: "loading … with an incorrect regex pattern … fix_mistral_regex" | Harmless. transformers prints this for a *Mistral* tokenizer bug — our model is Qwen3, whose tokenizer is unaffected. The GGUF's own tokenizer metadata (visible right after in the log: `tokenizer.ggml.pre = qwen2`, chat template embedded) is written correctly | Ignore it. If you're paranoid, load the GGUF and confirm the KO smoke question still triggers a `<tool_call>` |
| Scanner log: `UnicodeDecodeError: 'charmap' codec can't decode byte 0x8d` and/or `!! DJIA fetch failed ('NoneType' object has no attribute 'rfind') — continuing with S&P500 only` | TWO independent bugs stacked. (1) Windows decodes subprocess output with the locale codec (cp1252) unless told otherwise — a UTF-8 byte in a fetched page crashed the reader thread. Fixed: the scanner's subprocess calls now force `encoding="utf-8", errors="replace"`. (2) Even with (1) fixed, the Dow-30 fetch was dead anyway: Wikipedia's REST HTML no longer carries the DJIA components table at all (verified 2026-08 — names only in a navbox, no tickers). Fixed: the Dow list now comes from stockanalysis.com, the same trusted source as the Nasdaq-100 list | `git pull`, then re-run the scan. Note: your previous scan's results were still valid — it fell back to S&P500+NDX (540 tickers), and nearly all Dow 30 are S&P 500 members anyway, so the actual gap was tiny |
| Rubric calibration changed nothing — AAPL and ABBV STILL both 9/10 (ABBV even said "decaying" and kept its 9) | Prose rubric rules alone don't bind an 8B model — it reads them and then prints 9 anyway. Fixed three ways: (1) **code enforcement** — `ai_moat/calibration.py` now recomputes every saved score deterministically (round DOWN to the floor of the 5 source-score average; 10 requires all five sources ≥8, 9 requires four; −1 penalty when the answer's own DECAY CHECK says decaying/eroding OR the scanner measured op-margin trend ≤ −3pp). Whatever the model prints, the *saved* score obeys — verified on the exact answers that failed: AAPL 9→8, ABBV 9→7. (2) The rubric gained a matching "SCORING ARITHMETIC (mandatory)" section plus a benchmark anchor: every stock is compared against the MA/AAPL/MSFT shelf (MA golden example: ROIC ≥15% ten years running, 85% up-years, +7.0pp op-margin, FCF positive 10/10 — *consistency of growing profits and cash flow is the 9–10 qualifier*). (3) Optional weight-level fix: a quick calibration retrain teaches the model to show the arithmetic itself | `git pull`, delete the old saved evals for affected tickers, re-evaluate — the code enforcement works IMMEDIATELY with your current GGUF, no retrain needed. To also fix the model's printed numbers: eject the model in LM Studio → `python ai_moat/teach_calibration.py` (~15–40 min; `--dry-run` shows the rows first) → `python ai_moat/quantize_model.py` (auto-picks the new `-calib-lora` adapter) → import the new GGUF |
| Batch evaluate only uses dashboard research, I want the agent | Fixed: the Batch evaluate box now has a "🤖 Self-research agent in batch" checkbox — each ticker runs the full tool-calling agent (the model calls research_stock / web_search itself) instead of the pre-baked evidence card. Slower per ticker but deeper; calibration is enforced on those results too | `git pull`, tick the checkbox in the Batch evaluate expander |
| Answers are generic finance-blah | Forgot the system prompt | Step 2.5.3 |
| Anything else interrupted | — | Everything checkpoints. Just re-run the same command |

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
