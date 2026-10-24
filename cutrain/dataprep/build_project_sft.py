"""Verified whole-project demonstrations in opencode format (rendered_sft_v1 rows, one per assistant turn).

Each episode runs in project_env from an empty folder: look, write a spec draft, set up a virtualenv,
write the package one small module at a time, write tests, run them, write the owner guide, try the
command, report. Some episodes include a recovery: a write aimed at the folder itself, or a planted
bug the tests catch and the next edit fixes. Mistaken turns stay in the conversation as context but
produce no training row, so the model learns to recover without learning the mistake. When the
context passes the compaction threshold, the summary request is answered with a filled template from
the episode's real state, and that answer is a training row too.

Only episodes the environment grades as a full success are kept.

  python -m cutrain.dataprep.build_project_sft --model <base> --episodes 60 --out data/project_sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from cutrain.envs import project_tasks as pt
from cutrain.envs.opencode_env import call_message
from cutrain.envs.project_env import ProjectEpisode
from cutrain.serving.serve_adapter import to_template

FINALS = [
    "Built {pkg} in this folder.\n\n- Spec: `SPEC.md` (requirements, design, how it is verified)\n"
    "- Code: {modules}\n- Tests: {tests}\n- Owner guide: `README.md`\n\n"
    "Ran `.venv/bin/python -m pytest -q`: {result}. I also ran `{example}` to check the command works.",
    "Done. The project is organized as:\n{layout}\n\n"
    "Automated checks: `.venv/bin/python -m pytest -q` reports {result}. "
    "Setup, usage and how to run the tests are in `README.md`; the plan I followed is in `SPEC.md`.",
]


def plan_turns(ep: ProjectEpisode, rng: random.Random):
    """Yield (assistant message, trainable) pairs; mistakes are yielded with trainable=False."""
    task, w = ep.task, str(ep.work)
    files = list(task.files.items())
    code = [(r, t) for r, t in files if not r.startswith("tests/") and r != "requirements.txt"]
    tests = [(r, t) for r, t in files if r.startswith("tests/")]
    tool = lambda name, **args: {"role": "assistant", "content": None, "tool_calls": call_message([(name, args)])}

    yield tool("read", path=w), True
    if rng.random() < 0.2:  # the observed failure: writing to the folder itself, then correcting the path
        yield tool("write", path=w, content=task.spec_doc), False
    yield tool("write", path=f"{w}/SPEC.md", content=task.spec_doc), True
    yield tool("write", path=f"{w}/requirements.txt", content=task.files["requirements.txt"]), True
    yield tool("shell", command="python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt"), True
    bug = None
    if rng.random() < 0.3:  # plant a bug in one module that the tests will catch
        candidates = [(r, t) for r, t in code if r.endswith(".py") and not r.endswith(("__init__.py", "__main__.py"))
                      and _mutate(t) != t]
        if candidates:
            bug = rng.choice(candidates)
    for rel, text in code:
        if bug and rel == bug[0]:
            yield tool("write", path=f"{w}/{rel}", content=_mutate(text)), False
        else:
            yield tool("write", path=f"{w}/{rel}", content=text), True
    for rel, text in tests:
        yield tool("write", path=f"{w}/{rel}", content=text), True
    yield tool("shell", command=".venv/bin/python -m pytest -q"), True
    if bug:
        rel, text = bug
        yield tool("read", path=f"{w}/{rel}"), True
        yield tool("edit", path=f"{w}/{rel}", oldString=_mutate(text), newString=text), True
        yield tool("shell", command=".venv/bin/python -m pytest -q"), True
    yield tool("write", path=f"{w}/README.md", content=task.readme), True
    example = next((c.args for c in task.checks if c.ok and c.args and c.args[0] != "__SAMPLE__"), None)
    example_cmd = f"python3 -m {task.package} " + " ".join(_q(a) for a in example) if example else f"python3 -m {task.package} --help"
    yield tool("shell", command=example_cmd), True
    passed = _passed_line(ep)
    modules = ", ".join(f"`{r}`" for r, _ in code if r.endswith(".py"))
    test_list = ", ".join(f"`{r}`" for r, _ in tests)
    layout = "\n".join(f"- `{r}`" for r, _ in code + tests) + "\n- `SPEC.md`, `README.md`"
    final = rng.choice(FINALS).format(pkg=task.package, modules=modules, tests=test_list, result=passed,
                                      example=example_cmd, layout=layout)
    yield {"role": "assistant", "content": final}, True


def _mutate(text: str) -> str:
    """A small, plausible bug: flip a comparison or an arithmetic operator once."""
    for old, new in ((" >= ", " > "), (" <= ", " < "), (" + ", " - "), (" * ", " / ")):
        if old in text:
            return text.replace(old, new, 1)
    return text


def _q(arg: str) -> str:
    return f'"{arg}"' if " " in arg else arg


def _passed_line(ep: ProjectEpisode) -> str:
    for m in reversed(ep.messages):
        if m["role"] == "tool" and " passed" in (m["content"] or ""):
            line = [l for l in m["content"].splitlines() if " passed" in l]
            return line[-1].strip() if line else "all tests passed"
    return "all tests passed"


def summary_for(ep: ProjectEpisode) -> str:
    """A filled opencode summary template from the episode's actual state."""
    task = ep.task
    done = [r for r in ep.created]
    remaining = [r for r in list(task.files) + ["README.md"] if r not in done]
    reqs = [l.strip("- ").strip() for l in task.outline.splitlines() if l.strip().startswith("- ")][:6]
    lines = ["## Objective", f"- Build the {task.package} command-line tool described by the user, in this folder.", "",
             "## Requirements"] + [f"- {r}" for r in reqs] + ["", "## Decisions",
             f"- Python package `{task.package}/` with small modules, pytest tests in `tests/`, virtualenv in `.venv`.", "",
             "## Work State", "### Completed"] + ([f"- Wrote `{r}`" for r in done] or ["- (none)"]) + ["", "### Active",
             f"- Writing `{remaining[0]}`" if remaining else "- Final checks", "", "### Blocked", "- (none)", "",
             "## Next Move"] + [f"{i + 1}. Write `{r}`" for i, r in enumerate(remaining[:6])] + [
             f"{min(len(remaining), 6) + 1}. Run `.venv/bin/python -m pytest -q` and fix failures, then report.", "",
             "## Relevant Files", "- `SPEC.md`: requirements, design and verification plan", "",
             "## Important Context", "- The spec in `SPEC.md` is the source of truth for commands and output formats."]
    return "\n".join(lines)


def render_completion(ep: ProjectEpisode, tok, message: dict) -> tuple[str, str]:
    prompt = ep.prompt(tok)
    full = tok.apply_chat_template(to_template(ep.request_messages() + [message]), tools=ep.tools, tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("rendered turn is not a continuation of the prompt")
    if not full.endswith(tok.eos_token + "\n"):
        raise ValueError("unexpected end of rendered turn")
    return prompt, full[len(prompt):-len(tok.eos_token + "\n")]


def run_episode(seed: int, tok, compact_at: int) -> tuple[list[dict], dict]:
    task = pt.make_task(seed)
    rng = random.Random(seed * 7919)
    rows = []
    with ProjectEpisode(task, compact_at=compact_at) as ep:
        for message, trainable in plan_turns(ep, rng):
            prompt = ep.prompt(tok)  # may switch to a compaction request
            resumed = False
            while ep.compaction:
                summary = {"role": "assistant", "content": summary_for(ep)}
                p, c = render_completion(ep, tok, summary)
                rows.append({"rendered_prompt": p, "completion": c, "kind": "compaction"})
                ep.step(c)
                resumed = True
                prompt = ep.prompt(tok)
            if resumed:  # detail was lost with the old history: reopen the plan before continuing
                reread = {"role": "assistant", "content": None,
                          "tool_calls": call_message([("read", {"path": f"{ep.work}/SPEC.md"})])}
                p, c = render_completion(ep, tok, reread)
                rows.append({"rendered_prompt": p, "completion": c, "kind": "read"})
                ep.step(c)
            p, c = render_completion(ep, tok, message)
            if trainable:
                rows.append({"rendered_prompt": p, "completion": c,
                             "kind": "final" if not message.get("tool_calls") else message["tool_calls"][0]["function"]["name"]})
            ep.step(c)
        result = ep.result()
    for i, r in enumerate(rows):
        r.update(format="rendered_sft_v1", source="project_oracle", source_id=f"project:{seed}:{i}")
    return rows, result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--first-seed", type=int, default=40_000)
    ap.add_argument("--compact-at", type=int, default=9500)
    ap.add_argument("--compact-at-low", type=int, default=5000,
                    help="every third episode compacts this early, so summaries and checkpoint resumes are learned")
    ap.add_argument("--max-tokens", type=int, default=12300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    rows, report = [], {"episodes": 0, "rejected": [], "compaction_rows": 0, "over_max": 0}
    for seed in range(args.first_seed, args.first_seed + args.episodes):
        episode_rows, result = run_episode(seed, tok, args.compact_at_low if seed % 3 == 0 else args.compact_at)
        if not (result["success"] and not result["false_claim"] and not result["invalid"]):
            report["rejected"].append({"seed": seed, "task": result["task"], "spec": result["spec"].get("ok"),
                                       "tests": result["own_tests"].get("passed"), "checks": result["checks"]})
            continue
        for r in episode_rows:
            r["tokens"] = len(tok(r["rendered_prompt"] + r["completion"], add_special_tokens=False).input_ids) + 1
        report["over_max"] += sum(r["tokens"] > args.max_tokens for r in episode_rows)
        rows += [r for r in episode_rows if r["tokens"] <= args.max_tokens]
        report["episodes"] += 1
        report["compaction_rows"] += sum(r["kind"] == "compaction" for r in episode_rows)
        print(f"seed {seed} {result['task']}: {len(episode_rows)} rows, compactions {result['compactions']}", flush=True)
    random.Random(args.first_seed).shuffle(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    report.update(rows=len(rows), kinds=kinds, max_tokens=max(r["tokens"] for r in rows),
                  mean_tokens=round(sum(r["tokens"] for r in rows) / len(rows)))
    Path(args.out).with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
