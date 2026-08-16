"""Automatic quantization: trained LoRA adapter -> LM Studio-ready GGUF.

Runs the full pipeline with ONE command on your PC:
  1. MERGE   : load base + your LoRA adapter, merge to fp16 safetensors
               (needs ~30GB free disk for a 14B; done on CPU RAM+disk,
               so 16GB system RAM is fine — it streams shards).
  2. CONVERT : llama.cpp's convert_hf_to_gguf.py -> fp16 GGUF (~28GB for 14B).
  3. QUANTIZE: llama-quantize -> Q5_K_M GGUF (~9.9GB for 14B — fits fully
               in your 12GB VRAM for fast inference). Q4_K_M (~8.5GB) also
               supported via --quant.
  4. CLEANUP : deletes the huge fp16 intermediates (keep with --keep-fp16).

llama.cpp is acquired AUTOMATICALLY: the script git-clones it next to the
outputs folder and pip-installs its conversion requirements. llama-quantize
is found from (in order): --llama-bin, PATH, common install locations, or
built from the clone with cmake if you have a compiler. On Windows the
easiest is: download a llama.cpp release zip (llama-bXXXX-bin-win-cuda...)
from https://github.com/ggml-org/llama.cpp/releases and pass
  --llama-bin C:\\path\\to\\llama-quantize.exe

Usage (on your PC, from the repo root, after train_qlora.py finished):
  python ai_moat/quantize_model.py                     # 14B adapter, Q5_K_M
  python ai_moat/quantize_model.py --base 7b           # if you trained 7b
  python ai_moat/quantize_model.py --quant Q4_K_M      # smaller/faster
  python ai_moat/quantize_model.py --llama-bin /path/to/llama-quantize

Result: ai_moat/outputs/moat-<base>-<quant>.gguf
  -> in LM Studio: My Models -> import (or drop the file into the LM Studio
     models folder), then chat with your own moat model at 30-50 tok/s.

Disk needed (14B): ~28GB fp16 merge + ~28GB fp16 GGUF + ~10GB quant.
Intermediates are deleted as soon as they're no longer needed, so peak
usage is ~56GB briefly, ~38GB most of the time. 7B halves everything.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "outputs")

BASES = {
    "qwen3-14b": "unsloth/Qwen3-14B-unsloth-bnb-4bit",
    "14b": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    "7b": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
}


def sh(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def find_llama_quantize(cli_arg: str | None, llama_dir: str) -> str | None:
    """Locate llama-quantize: CLI arg > PATH > common spots > freshly built."""
    if cli_arg:
        if os.path.exists(cli_arg):
            return cli_arg
        sys.exit(f"--llama-bin path does not exist: {cli_arg}")
    p = shutil.which("llama-quantize")
    if p:
        return p
    candidates = [
        os.path.join(llama_dir, "build", "bin", "llama-quantize"),
        os.path.join(llama_dir, "build", "bin", "llama-quantize.exe"),
        os.path.join(llama_dir, "llama-quantize"),
        os.path.join(llama_dir, "llama-quantize.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def ensure_llamacpp(llama_dir: str) -> None:
    """Clone llama.cpp (for the converter script) and install its deps."""
    if not os.path.exists(os.path.join(llama_dir, "convert_hf_to_gguf.py")):
        print(f"cloning llama.cpp -> {llama_dir}")
        sh(["git", "clone", "--depth", "1",
            "https://github.com/ggml-org/llama.cpp", llama_dir])
    # converter deps (gguf, sentencepiece, ...) — small pure-python installs
    req = os.path.join(llama_dir, "requirements",
                       "requirements-convert_hf_to_gguf.txt")
    if os.path.exists(req):
        sh([sys.executable, "-m", "pip", "install", "-q", "-r", req])
    else:
        sh([sys.executable, "-m", "pip", "install", "-q",
            "gguf", "sentencepiece", "protobuf"])


def try_build_quantize(llama_dir: str) -> str | None:
    """Best-effort cmake build of llama-quantize (CPU-only is enough)."""
    if not shutil.which("cmake"):
        return None
    build = os.path.join(llama_dir, "build")
    try:
        sh(["cmake", "-S", llama_dir, "-B", build,
            "-DGGML_CUDA=OFF", "-DBUILD_SHARED_LIBS=OFF",
            "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DCMAKE_BUILD_TYPE=Release"])
        sh(["cmake", "--build", build, "--config", "Release",
            "-j", "--target", "llama-quantize"])
    except subprocess.CalledProcessError:
        return None
    return find_llama_quantize(None, llama_dir)


def merge_adapter(base_key: str, adapter_dir: str, merged_dir: str) -> None:
    """Merge LoRA into the base model, save fp16 safetensors."""
    if os.path.exists(os.path.join(merged_dir, "config.json")):
        print(f"merge already exists -> {merged_dir} (skipping)")
        return
    print("merging LoRA adapter into base (this streams shards; slow but "
          "RAM-safe)...")
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        adapter_dir, max_seq_length=2048, load_in_4bit=True)
    model.save_pretrained_merged(merged_dir, tokenizer,
                                 save_method="merged_16bit")
    print(f"merged fp16 model -> {merged_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=list(BASES), default="qwen3-14b",
                    help="which base you trained (matches train_qlora.py)")
    ap.add_argument("--quant", default="Q5_K_M",
                    choices=["Q5_K_M", "Q4_K_M", "Q6_K", "Q8_0"],
                    help="GGUF quant type (Q5_K_M ~9.9GB for 14B, "
                         "Q4_K_M ~8.5GB)")
    ap.add_argument("--adapter", default=None,
                    help="adapter dir (default: outputs/moat-<base>-lora)")
    ap.add_argument("--llama-bin", default=None,
                    help="path to llama-quantize binary (Windows: point at "
                         "llama-quantize.exe from a release zip)")
    ap.add_argument("--keep-fp16", action="store_true",
                    help="keep the huge fp16 intermediates (default: delete)")
    args = ap.parse_args()

    adapter_dir = args.adapter or os.path.join(OUTDIR,
                                               f"moat-{args.base}-lora")
    if not os.path.exists(adapter_dir):
        sys.exit(f"adapter not found: {adapter_dir}\n"
                 f"run train_qlora.py first (or pass --adapter).")

    merged_dir = os.path.join(OUTDIR, f"moat-{args.base}-merged")
    fp16_gguf = os.path.join(OUTDIR, f"moat-{args.base}-fp16.gguf")
    out_gguf = os.path.join(OUTDIR, f"moat-{args.base}-{args.quant}.gguf")
    llama_dir = os.path.join(OUTDIR, "llama.cpp")

    if os.path.exists(out_gguf):
        sys.exit(f"already done: {out_gguf}\n(delete it to re-run)")

    # ---- step 0: toolchain ----
    ensure_llamacpp(llama_dir)
    quant_bin = find_llama_quantize(args.llama_bin, llama_dir)
    if quant_bin is None:
        print("llama-quantize not found — attempting a CPU-only cmake build "
              "(needs a C++ compiler)...")
        quant_bin = try_build_quantize(llama_dir)
    if quant_bin is None:
        sys.exit(
            "Could not find or build llama-quantize.\n"
            "EASIEST FIX (Windows): download a release zip from\n"
            "  https://github.com/ggml-org/llama.cpp/releases\n"
            "(pick llama-bXXXX-bin-win-cuda-x64.zip or -cpu-x64.zip), unzip,\n"
            "then re-run with:\n"
            "  python ai_moat/quantize_model.py --llama-bin "
            "C:\\path\\to\\llama-quantize.exe")
    print(f"using llama-quantize: {quant_bin}")

    # ---- step 1: merge ----
    merge_adapter(args.base, adapter_dir, merged_dir)

    # ---- step 2: convert to fp16 GGUF ----
    if not os.path.exists(fp16_gguf):
        sh([sys.executable, os.path.join(llama_dir, "convert_hf_to_gguf.py"),
            merged_dir, "--outfile", fp16_gguf, "--outtype", "f16"])
    else:
        print(f"fp16 GGUF already exists -> {fp16_gguf} (skipping convert)")

    # free the merged dir early — it's no longer needed
    if not args.keep_fp16:
        print(f"deleting merged fp16 dir to free disk: {merged_dir}")
        shutil.rmtree(merged_dir, ignore_errors=True)

    # ---- step 3: quantize ----
    sh([quant_bin, fp16_gguf, out_gguf, args.quant])

    # ---- step 4: cleanup ----
    if not args.keep_fp16:
        print(f"deleting fp16 GGUF to free disk: {fp16_gguf}")
        os.remove(fp16_gguf)

    size_gb = os.path.getsize(out_gguf) / 1e9
    print("\n" + "=" * 60)
    print(f"DONE: {out_gguf}  ({size_gb:.1f} GB)")
    print("Load it in LM Studio: My Models -> import, or copy into the "
          "LM Studio models folder.")
    print("It fits fully in your 12GB VRAM -> expect 30-50 tok/s.")
    print("=" * 60)


if __name__ == "__main__":
    main()
