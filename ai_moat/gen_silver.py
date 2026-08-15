"""Fill the silver-tier prompts with a FREE local teacher model.

The teacher sees ONLY the Adam rubric (system prompt) + the evidence card.
It never sees analyst consensus or third-party moat ratings, so consensus
bias structurally cannot leak into the labels.

Works with any OpenAI-compatible server — LM Studio is the zero-config
option (Developer tab -> Start Server -> default http://localhost:1234).
Teacher model: Qwen2.5-32B-Instruct GGUF, e.g. the community upload
  jorgedelpozolerida/Qwen2.5-32B-Instruct-Q3_K_M-GGUF  (~15.95 GB)
It will offload to system RAM on a 12GB card — slow (roughly 1-3 tok/s,
~1-3 min per answer, so ~443 prompts = one long overnight run, possibly
two nights). Speed does not matter for batch label generation; quality
does. Progress is checkpointed after EVERY answer, so you can stop and
re-run at any time — it resumes where it left off.

Usage (on your PC, from the repo root):
  python ai_moat/gen_silver.py                          # LM Studio default
  python ai_moat/gen_silver.py --base-url http://localhost:1234/v1
  python ai_moat/gen_silver.py --model qwen2.5-32b-instruct  # exact name from LM Studio
  python ai_moat/gen_silver.py --limit 5                # smoke test 5 prompts first

Output: ai_moat/dataset/silver.jsonl (same chat-messages format as gold).
Uses only the Python standard library — nothing to pip install.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "dataset", "silver_prompts.jsonl")
OUT = os.path.join(HERE, "dataset", "silver.jsonl")

REQUIRED_HEADERS = ("MOAT VERDICT:", "SOURCES", "REASONING:", "ACTION")


def chat(base_url: str, model: str, messages: list, timeout: int = 900) -> str:
    """One OpenAI-compatible /chat/completions call, stdlib only."""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.3,        # low temp: consistent, rubric-bound answers
        "max_tokens": 900,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.load(r)
    return payload["choices"][0]["message"]["content"]


def looks_valid(answer: str) -> bool:
    """Cheap structural gate: the mandated format headers must be present."""
    return all(h in answer for h in REQUIRED_HEADERS)


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
    args = ap.parse_args()

    if not os.path.exists(PROMPTS):
        sys.exit(f"missing {PROMPTS} — run: python3 -m ai_moat.build_dataset")

    prompts = [json.loads(l) for l in open(PROMPTS, encoding="utf-8")]

    # Resume support: skip tickers already answered.
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT, encoding="utf-8"):
            try:
                done.add(json.loads(l)["ticker"])
            except Exception:
                pass
    todo = [p for p in prompts if p["ticker"] not in done]
    print(f"{len(prompts)} prompts total, {len(done)} already answered, "
          f"{len(todo)} to go")
    if not todo:
        print("nothing to do — silver.jsonl is complete")
        return

    new = 0
    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as f:
        for i, p in enumerate(todo, 1):
            if args.limit and new >= args.limit:
                break
            answer = None
            for attempt in range(1 + args.retries):
                try:
                    a = chat(args.base_url, args.model, p["messages"])
                except Exception as e:
                    print(f"  {p['ticker']}: request failed ({e}); "
                          f"is the LM Studio server running?")
                    time.sleep(5)
                    continue
                if looks_valid(a):
                    answer = a
                    break
                print(f"  {p['ticker']}: attempt {attempt+1} failed the "
                      f"format gate, retrying")
            if answer is None:
                print(f"  {p['ticker']}: SKIPPED after retries "
                      f"(re-run the script later to retry it)")
                continue
            row = {"messages": p["messages"] + [
                       {"role": "assistant", "content": answer}],
                   "tier": "silver", "ticker": p["ticker"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()                       # checkpoint after every answer
            new += 1
            rate = (time.time() - t0) / new
            left = (len(todo) - i) * rate / 3600
            print(f"[{i}/{len(todo)}] {p['ticker']} done "
                  f"({rate:.0f}s/answer, ~{left:.1f}h remaining)")
    print(f"\nwrote {new} new answers -> {OUT}")
    print("NEXT: human spot-review ~10% of silver.jsonl (delete rows whose "
          "verdict contradicts the evidence card), then run train_qlora.py")


if __name__ == "__main__":
    main()
