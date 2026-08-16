"""Fill the silver-tier prompts with a FREE local teacher model — FAST.

The teacher sees ONLY the Adam rubric (system prompt) + the evidence card.
It never sees analyst consensus or third-party moat ratings, so consensus
bias structurally cannot leak into the labels.

WHY THIS IS NOW ~20-50x FASTER THAN BEFORE
------------------------------------------
The old setup (Qwen2.5-32B Q3_K_M on a 12GB card) offloaded half the model
to system RAM -> 1-3 tok/s -> ~16h even for a handful of prompts. Fixes:

1. TEACHER THAT FITS ENTIRELY IN 12GB VRAM (the big one):
     RECOMMENDED: Qwen3-14B  GGUF Q4_K_M (~9.0GB)  -> 30-50 tok/s
     search LM Studio for:  "Qwen3-14B GGUF" (lmstudio-community or unsloth)
     Fully-in-VRAM 14B beats half-offloaded 32B on wall-clock by 20-50x and
     the quality difference on a structured rubric task is small.
     QUALITY OPTION (slower): Qwen3.8-27B UD-Q3_K_XL — newest model, but at
     ~14GB it partially offloads on 12GB; expect ~5-10 tok/s. Use overnight.
   In LM Studio set: GPU offload = MAX, context = 4096 (NOT more — long
   context steals VRAM), and turn OFF "keep model in RAM".

2. GREAT-BUSINESSES-ONLY (default): the moat model's job is to grade moats
   of companies that already pass the great-business scan, so we only label
   those (~150 prompts instead of ~440). Override with --all if ever needed.

3. THINKING SUPPRESSED: Qwen3-series models think by default (hundreds of
   hidden tokens per answer). We suppress it TWO ways, because LM Studio
   builds have been seen ignoring the API flag:
     a) chat_template_kwargs {"enable_thinking": false}  (the API way)
     b) "/no_think" appended to the user message — Qwen3's in-band soft
        switch, honoured by the chat template itself, works everywhere.
   Without this the model burns the ENTIRE max_tokens budget inside
   reasoning_content, content comes back empty, and every ticker fails
   the format gate. Override with --thinking if you want reasoning
   (3-5x slower, marginal quality gain on this structured task).

4. PARALLEL WORKERS: --workers 2 (default) keeps the GPU saturated while
   HTTP round-trips happen. LM Studio queues them safely.

EXPECTED RUNTIME (Qwen3-14B Q4_K_M fully in VRAM, ~150 great-only prompts):
   roughly 20-40s per answer -> ~1-1.5 hours total. Not 16 hours.

Works with any OpenAI-compatible server — LM Studio is the zero-config
option (Developer tab -> Start Server -> default http://localhost:1234).

Usage (on your PC, from the repo root):
  python ai_moat/gen_silver.py --limit 5     # smoke test first — ALWAYS
  python ai_moat/gen_silver.py               # full great-only run
  python ai_moat/gen_silver.py --all         # label every scanned ticker
  python ai_moat/gen_silver.py --workers 1   # if LM Studio misbehaves
  python ai_moat/gen_silver.py --thinking    # allow chain-of-thought (slow)

Output: ai_moat/dataset/silver.jsonl (same chat-messages format as gold).
Progress is checkpointed after EVERY answer — stop and re-run any time.
Uses only the Python standard library — nothing to pip install.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROMPTS = os.path.join(HERE, "dataset", "silver_prompts.jsonl")
SCAN = os.path.join(ROOT, "public", "data", "scan_results.json")
OUT = os.path.join(HERE, "dataset", "silver.jsonl")

REQUIRED_HEADERS = ("MOAT VERDICT:", "SOURCES", "REASONING:", "ACTION")

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

_write_lock = threading.Lock()


def strip_thinking(text: str) -> str:
    """Remove Qwen3-style <think>...</think> blocks (and a dangling open tag)."""
    text = _THINK_RE.sub("", text)
    # model ran out of tokens mid-think: drop everything from the open tag
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


def chat(base_url: str, model: str, messages: list, max_tokens: int,
         thinking: bool, timeout: int = 900) -> str:
    """One OpenAI-compatible /chat/completions call, stdlib only."""
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,        # low temp: consistent, rubric-bound answers
        "max_tokens": max_tokens,
    }
    if not thinking:
        # Belt: understood by llama-server for Qwen3-family templates —
        # but some LM Studio builds IGNORE it (observed: content="" with
        # 798/800 tokens in reasoning_content, finish_reason=length).
        body["chat_template_kwargs"] = {"enable_thinking": False}
        # Braces: Qwen3's in-band soft switch. The chat template itself
        # reads "/no_think" from the latest user turn and disables
        # thinking — this works even when the API flag is dropped.
        # Copy the messages so we never mutate the caller's prompt rows
        # (they get written verbatim into silver.jsonl).
        msgs = [dict(m) for m in messages]
        for m in reversed(msgs):
            if m.get("role") == "user":
                m["content"] = m["content"].rstrip() + "\n\n/no_think"
                break
        body["messages"] = msgs
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.load(r)
    except urllib.error.HTTPError as e:
        # Read the server's actual error text — a bare "400 Bad Request"
        # hides the real reason (model not loaded, context exceeded, ...).
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            detail = ""
        raise RuntimeError(f"HTTP {e.code}: {detail or e.reason}") from None
    msg = payload["choices"][0]["message"]
    content = msg.get("content") or ""
    # LM Studio separates hidden reasoning into reasoning_content. If the
    # model spent its whole token budget thinking, content is empty and
    # the formatted answer was never generated — salvaging the reasoning
    # text is useless, so surface a targeted diagnostic instead.
    if not content.strip() and (msg.get("reasoning_content") or "").strip():
        raise RuntimeError(
            "all tokens went to reasoning_content, answer is empty — the "
            "/no_think switch should stop this; if it persists, open the "
            "model's settings in LM Studio (gear icon next to the loaded "
            "model) and turn Reasoning OFF, then retry")
    return strip_thinking(content)


def looks_valid(answer: str) -> bool:
    """Cheap structural gate: the mandated format headers must be present."""
    return all(h in answer for h in REQUIRED_HEADERS)


def great_tickers() -> set:
    """Tickers that pass the great-business scan (is_great == True)."""
    if not os.path.exists(SCAN):
        print(f"WARNING: {SCAN} missing — cannot filter to great businesses; "
              f"labelling ALL prompts")
        return set()
    with open(SCAN, encoding="utf-8") as f:
        d = json.load(f)
    rows = d["results"] if isinstance(d, dict) else d
    return {r["ticker"] for r in rows
            if isinstance(r, dict) and r.get("is_great")}


def answer_one(p: dict, args) -> tuple:
    """Worker: returns (prompt, answer_or_None)."""
    for attempt in range(1 + args.retries):
        try:
            a = chat(args.base_url, args.model, p["messages"],
                     args.max_tokens, args.thinking)
        except Exception as e:
            msg = str(e)
            if "reasoning_content" in msg:
                print(f"  {p['ticker']}: {msg}")
            elif "context" in msg.lower():
                print(f"  {p['ticker']}: CONTEXT OVERFLOW — in LM Studio, "
                      f"eject the model and reload it with Context Length "
                      f"8192 (each parallel worker gets ctx/workers; "
                      f"4096/2 = 2048 is too small for prompt+answer). "
                      f"Or re-run with --workers 1.")
            else:
                print(f"  {p['ticker']}: request failed ({msg}); "
                      f"check LM Studio: model loaded? server started?")
            time.sleep(5)
            continue
        if looks_valid(a):
            return p, a
        print(f"  {p['ticker']}: attempt {attempt+1} failed the format gate, "
              f"retrying")
    return p, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:1234/v1",
                    help="OpenAI-compatible endpoint (LM Studio default)")
    ap.add_argument("--model", default="local-model",
                    help="model name as shown in LM Studio (any string works "
                         "for LM Studio single-model serving)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N new answers (0 = all; use 5 to smoke-test)")
    ap.add_argument("--retries", type=int, default=2,
                    help="retries per prompt when the answer fails the format gate")
    ap.add_argument("--workers", type=int, default=2,
                    help="parallel in-flight requests (2 keeps the GPU busy; "
                         "use 1 if the server misbehaves)")
    ap.add_argument("--max-tokens", type=int, default=800,
                    help="answer token cap (800 fits the mandated format easily)")
    ap.add_argument("--all", action="store_true",
                    help="label EVERY scanned ticker, not just great businesses")
    ap.add_argument("--thinking", action="store_true",
                    help="allow chain-of-thought (3-5x slower; off by default)")
    args = ap.parse_args()

    if not os.path.exists(PROMPTS):
        sys.exit(f"missing {PROMPTS} — run: python3 -m ai_moat.build_dataset")

    prompts = [json.loads(l) for l in open(PROMPTS, encoding="utf-8")]

    # Great-businesses-only by default — the moat model grades companies that
    # already pass the scan; labelling the rejects wastes hours of GPU time.
    if not args.all:
        g = great_tickers()
        if g:
            before = len(prompts)
            prompts = [p for p in prompts if p["ticker"] in g]
            print(f"great-only filter: {before} prompts -> {len(prompts)} "
                  f"(pass the great-business scan); use --all to override")

    # Resume support: skip tickers already answered.
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            try:
                done.add(json.loads(l)["ticker"])
            except Exception:
                pass
    todo = [p for p in prompts if p["ticker"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(prompts)} prompts in scope, {len(done)} already answered, "
          f"{len(todo)} to do now")
    if not todo:
        print("nothing to do — silver.jsonl is complete for this scope")
        return

    new = skipped = 0
    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as f, \
         ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(answer_one, p, args) for p in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            p, answer = fut.result()
            if answer is None:
                skipped += 1
                print(f"  {p['ticker']}: SKIPPED after retries "
                      f"(re-run the script later to retry it)")
                continue
            row = {"messages": p["messages"] + [
                       {"role": "assistant", "content": answer}],
                   "tier": "silver", "ticker": p["ticker"]}
            with _write_lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()                   # checkpoint after every answer
            new += 1
            rate = (time.time() - t0) / max(1, new)
            left = (len(todo) - i) * rate / 60
            print(f"[{i}/{len(todo)}] {p['ticker']} done "
                  f"({rate:.0f}s/answer, ~{left:.0f}min remaining)")

    print(f"\nwrote {new} new answers -> {OUT}"
          + (f"  ({skipped} skipped — re-run to retry)" if skipped else ""))
    print("NEXT: human spot-review ~10% of silver.jsonl (delete rows whose "
          "verdict contradicts the evidence card), then run train_qlora.py")


if __name__ == "__main__":
    main()
