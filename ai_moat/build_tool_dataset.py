"""Build the small TOOL-CALLING dataset for the quick teach_tools.py pass.

Purpose: your moat model has a knowledge cutoff. Teaching it to CALL TOOLS
lets it fetch fresh facts forever. This dataset teaches exactly two tools
in Qwen3's NATIVE tool-call format (the <tool_call> JSON format LM Studio
parses into OpenAI-style tool_calls):

  research_stock(ticker) -> the scanner's evidence card + live Yahoo
                            fundamentals and headlines
  web_search(query)      -> web snippets for anything else

Three behaviours are taught (all from EXISTING gold labels — no invented
verdicts, every final answer is Adam's real one):

  A. NO evidence card in the prompt  -> call research_stock first,
     then answer from the tool result.                     (~37 rows)
  B. Evidence card ALREADY provided  -> answer directly, NO tool call
     (prevents pointless tool-spam).                       (~12 rows)
  C. Question mentions recent/current events -> call web_search,
     then answer.                                          (~10 rows)

Output: ai_moat/dataset/tools.jsonl  (chat-messages + tools schema)

Usage:  python3 -m ai_moat.build_tool_dataset
"""
from __future__ import annotations

import json
import os

from ai_moat.build_dataset import (evidence_block, gold_answer, load_scan,
                                   LABELS, SYSPROMPT)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dataset", "tools.jsonl")

# OpenAI-style tool schemas — same ones the dashboard's agent mode sends.
TOOLS = [
    {"type": "function", "function": {
        "name": "research_stock",
        "description": ("Fetch the quantitative evidence card for a stock: "
                        "10-year scanner metrics (margin trends, ROIC, moat "
                        "evidence score) plus CURRENT fundamentals and "
                        "recent headlines. Call this BEFORE judging a moat "
                        "whenever no evidence card was provided."),
        "parameters": {"type": "object", "properties": {
            "ticker": {"type": "string",
                       "description": "Stock ticker symbol, e.g. AAPL"}},
            "required": ["ticker"]}}},
    {"type": "function", "function": {
        "name": "web_search",
        "description": ("Search the web for recent news, events or facts "
                        "not covered by the evidence card — competitive "
                        "threats, regulation, management changes. Use when "
                        "the question involves 'recent', 'current', "
                        "'latest' or anything after your training data."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string",
                      "description": "Search query"}},
            "required": ["query"]}}},
]

ASK_NO_CARD = ("Evaluate this company's economic moat using the rubric. "
               "No evidence card is attached — research it first, then "
               "follow the output format exactly.\n\nTICKER: {t}")
ASK_WITH_CARD = ("Evaluate this company's economic moat using the rubric. "
                 "Use the evidence card; follow the output format "
                 "exactly.\n\n{ev}")
ASK_RECENT = ("Has anything happened recently that changes or threatens "
              "{name}'s ({t}) moat? Check the latest news, then give the "
              "verdict in the exact output format.")


def tc(name, args):
    """One assistant tool-call message (Qwen3 chat-template structure)."""
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {
                "name": name, "arguments": args}}]}


def tool_result(text):
    return {"role": "tool", "content": text}


def main():
    with open(SYSPROMPT, encoding="utf-8") as f:
        system = f.read()
    system += ("\n\n## TOOLS\nYou can call tools. If the user gives you no "
               "evidence card, call research_stock first. For questions "
               "about recent/current events, call web_search. When the "
               "evidence card is already in the message, answer directly "
               "without tools. Never invent numbers — only use what the "
               "card or a tool result actually says.")

    with open(LABELS, encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    scan = load_scan()

    rows = []

    # ---- A. no card -> research_stock -> answer -------------------------
    for lab in labels:
        t = lab["ticker"]
        row = scan.get(t)
        ev = (evidence_block(row, t) if row else
              f"TICKER: {t}\n(no scanner data on file — Yahoo section only)")
        result = (ev + "\n\nLIVE RESEARCH ADDENDUM (Yahoo Finance):\n"
                  "(current fundamentals and headlines appear here at "
                  "inference time)")
        rows.append({"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": ASK_NO_CARD.format(t=t)},
            tc("research_stock", {"ticker": t}),
            tool_result(result),
            {"role": "assistant", "content": gold_answer(lab)},
        ], "tier": "tool_research", "ticker": t})

    # ---- B. card provided -> NO tool call --------------------------------
    with_card = [lab for lab in labels if scan.get(lab["ticker"])][:12]
    for lab in with_card:
        t = lab["ticker"]
        rows.append({"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": ASK_WITH_CARD.format(
                ev=evidence_block(scan[t], t))},
            {"role": "assistant", "content": gold_answer(lab)},
        ], "tier": "tool_none", "ticker": t})

    # ---- C. recent-events question -> web_search -> answer ---------------
    # Search "results" are built from Adam's REAL reasoning text for the
    # ticker — no invented facts; teaches the format, not fake news.
    named = [lab for lab in labels if lab.get("adam_reasoning")][:10]
    for lab in named:
        t = lab["ticker"]
        name = t  # company name not in labels; ticker is fine for training
        reasoning = lab["adam_reasoning"]
        snippets = ("SEARCH RESULTS:\n"
                    f"- Analysis of {t}'s competitive position: "
                    f"{reasoning[:220]}\n"
                    f"- Commentary: investors debate the durability of "
                    f"{t}'s advantages given the points above.")
        rows.append({"messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": ASK_RECENT.format(name=name, t=t)},
            tc("web_search", {"query": f"{t} moat competitive threats "
                                       f"recent news"}),
            tool_result(snippets),
            {"role": "assistant", "content": gold_answer(lab)},
        ], "tier": "tool_search", "ticker": t})

    # ---- Emphasize CORRECTED labels ---------------------------------------
    # The TSLA gold answer changed (key-man now names Elon Musk explicitly).
    # The already-trained adapter learned the OLD bare "flagged" answer, so
    # this top-up must overwrite that habit: repeat every TSLA trajectory x4
    # (same trick as gold x6 weighting in full training — repetition, not
    # invention; the answer text is still Adam's real verdict).
    emphasized = []
    for r in rows:
        emphasized.append(r)
        if r["ticker"] == "TSLA":
            emphasized.extend([r] * 3)
    rows = emphasized

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    a = sum(1 for r in rows if r["tier"] == "tool_research")
    b = sum(1 for r in rows if r["tier"] == "tool_none")
    c = sum(1 for r in rows if r["tier"] == "tool_search")
    print(f"tool dataset: {len(rows)} rows -> {OUT}")
    print(f"  A research_stock trajectories: {a}")
    print(f"  B no-tool (card provided):     {b}")
    print(f"  C web_search trajectories:     {c}")
    print("next: python ai_moat/teach_tools.py   (~20-40 min)")


if __name__ == "__main__":
    main()
