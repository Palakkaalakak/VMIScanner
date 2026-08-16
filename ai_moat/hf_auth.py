"""HuggingFace token auto-loader (shared by train_qlora & quantize_model).

Priority:
  1. HF_TOKEN environment variable (already set? we leave it alone)
  2. ai_moat/.hf_token file (one line, the token) — GIT-IGNORED, stays on
     your PC only. Create it once:
       PowerShell:  Set-Content ai_moat/.hf_token "hf_yourtokenhere"
       (or just make the file in Notepad)

Why: authenticated HF downloads get higher rate limits + faster Xet
transfer. The token never gets committed — .gitignore covers it.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.join(HERE, ".hf_token")


def load_hf_token() -> str | None:
    """Set HF_TOKEN env var from .hf_token file if not already set."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, encoding="utf-8") as f:
            tok = f.read().strip()
        if tok:
            os.environ["HF_TOKEN"] = tok
            # huggingface_hub also honours this legacy name:
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", tok)
            print("HF token loaded from ai_moat/.hf_token "
                  "(faster, higher-rate-limit downloads)")
            return tok
    return None
