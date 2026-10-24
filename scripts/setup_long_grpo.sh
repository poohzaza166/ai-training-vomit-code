#!/usr/bin/env bash
set -euo pipefail
task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if command -v uv >/dev/null 2>&1; then
    task_uv="$(command -v uv)"
elif [[ -x /home/pooh/.grpo-bootstrap/bin/uv ]]; then
    task_uv=/home/pooh/.grpo-bootstrap/bin/uv
else
    python3 -m venv "$task_repo_root/.grpo-bootstrap"
    "$task_repo_root/.grpo-bootstrap/bin/python" -m pip install uv
    task_uv="$task_repo_root/.grpo-bootstrap/bin/uv"
fi
if [[ ! -x "$task_repo_root/.venv-grpo/bin/python" ]]; then
    "$task_uv" venv --python 3.12 "$task_repo_root/.venv-grpo"
fi
"$task_uv" pip install --python "$task_repo_root/.venv-grpo/bin/python" -r "$task_repo_root/requirements-long-grpo.txt"
"$task_repo_root/.venv-grpo/bin/python" - <<'PY'
from pathlib import Path
import shlex
import sys
import ziglang
compiler = Path(ziglang.__file__).parent / 'zig'
wrapper = Path(sys.executable).parent / 'grpo-cc'
wrapper.write_text('#!/bin/sh\nexec ' + shlex.quote(str(compiler)) + ' cc "$@"\n')
wrapper.chmod(0o755)
PY
"$task_uv" pip freeze --python "$task_repo_root/.venv-grpo/bin/python" > "$task_repo_root/requirements-long-grpo.lock.txt"
printf 'Environment ready. GPU driver and Bubblewrap must work before training.\n'
