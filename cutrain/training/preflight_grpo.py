"""Record hardware/software evidence without loading the 4B model or changing drivers."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return {"command": argv, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": argv, "error": str(exc)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-gpu", action="store_true")
    args = ap.parse_args()
    result = {"host": platform.node(), "platform": platform.platform(), "python": sys.version,
              "gpu": command([shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"]),
              "pci": command(["lspci", "-nn"]), "ram": command(["free", "-h"]),
              "nvidia_module": command(["/usr/sbin/modinfo", "nvidia"])}
    versions = {}
    for name in ("torch", "transformers", "trl", "peft", "bitsandbytes", "kernels", "datasets", "ziglang"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result["versions"] = versions
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["devices"] = [{"name": torch.cuda.get_device_name(i),
                              "total_gib": torch.cuda.get_device_properties(i).total_memory / 1024**3}
                             for i in range(torch.cuda.device_count())]
    except Exception as exc:
        result["cuda_available"] = False
        result["torch_error"] = str(exc)
    try:
        from cutrain.envs.agent_grpo_env import Episode, action_text
        with Episode(19) as ep:
            ep.seed_history()
            for action in ep.oracle():
                ep.step(action_text(action))
            result["sandbox_oracle"] = ep.result()
    except Exception as exc:
        result["sandbox_error"] = str(exc)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if args.require_gpu and not result["cuda_available"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
