"""Merged model -> q8_0 GGUF -> Ollama model (run after train_orpo.py --merge / --merge-only).

  python -m cutrain.serving.to_ollama
  ollama run nemotron-3-nano-cu:4b

Uses llama.cpp's convert_hf_to_gguf.py (shallow-cloned next to this file on first run) but feeds it config.json
as-is: its loader goes through transformers AutoConfig, and transformers 5 adds MoE defaults to every Nemotron-H
config, so the converter would write an 8-expert `nemotron_h_moe` GGUF that Ollama refuses to load.
The GGUF step exists because Ollama 0.34 only quantizes safetensors via its MLX runtime (absent on Linux) and
only imports GGUFs that are already quantized.
"""
from __future__ import annotations

import argparse
import json
import re
import runpy
import subprocess
import sys
from pathlib import Path

from cutrain.paths import REPO


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "out" / "nemotron-nano-4b-orpo"), help="train_orpo.py --out")
    ap.add_argument("--name", default="nemotron-3-nano-cu:4b", help="Ollama model name")
    ap.add_argument("--outtype", default="q8_0", help="converter type: q8_0 | bf16 | f16 (k-quants need llama-quantize)")
    ap.add_argument("--llama-cpp", default=str(REPO / "llama.cpp"))
    args = ap.parse_args()

    llama = Path(args.llama_cpp)
    if not llama.exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(llama)], check=True)
    out = Path(args.out)
    gguf = out / f"nemotron-3-nano-cu-{args.outtype}.gguf"

    sys.path[:0] = [str(llama), str(llama / "gguf-py")]
    from conversion.base import ModelBase

    ModelBase.load_hparams = staticmethod(
        lambda dir_model, is_mistral_format=False: json.loads((Path(dir_model) / "config.json").read_text()))
    sys.argv = ["convert_hf_to_gguf.py", str(out / "merged"), "--outtype", args.outtype, "--outfile", str(gguf)]
    try:
        runpy.run_path(str(llama / "convert_hf_to_gguf.py"), run_name="__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            raise

    # Modelfile next to this script is the template (renderer/parser/params); point its FROM at the new GGUF
    modelfile = out / "Modelfile"
    template = (REPO / "Modelfile").read_text()
    modelfile.write_text(re.sub(r"(?m)^FROM .*$", lambda _: f"FROM {gguf}", template))
    subprocess.run(["ollama", "create", args.name, "-f", str(modelfile)], check=True)
    print(f"ollama model ready: ollama run {args.name}")


if __name__ == "__main__":
    main()
