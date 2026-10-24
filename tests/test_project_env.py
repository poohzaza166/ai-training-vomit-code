import json
import re

import pytest

from cutrain.envs import project_tasks as pt
from cutrain.envs.project_env import COMPACTION, TEMPLATE, ProjectEpisode, serialize_history, tool_error
from cutrain.paths import REPO

DATA = REPO / "data"


def write_reference(ep, task, spec_first=True):
    files = list(task.files.items())
    order = ([("SPEC.md", task.spec_doc)] + files + [("README.md", task.readme)] if spec_first
             else files + [("SPEC.md", task.spec_doc), ("README.md", task.readme)])
    for rel, text in order:
        assert ep.run_tool("write", {"path": f"{ep.work}/{rel}", "content": text}).startswith("Created file")


def test_tool_results_match_real_opencode_trimmed():
    req = json.loads((DATA / "opencode_trimmed_capture_edge.json").read_text())
    captured = re.search(r"Working directory: (\S+)", req["messages"][0]["content"]).group(1)
    task = pt.make_task(0)
    with ProjectEpisode(task) as ep:
        work = str(ep.work)
        assert ep.tools == req["tools"]
        same = lambda s: re.sub(r"Today's date: .+", "", s.replace(captured, work))
        assert same(req["messages"][0]["content"]) == re.sub(r"Today's date: .+", "", ep.messages[0]["content"])
        msgs, compared = req["messages"][2:], 0
        for i, m in enumerate(msgs):
            if m["role"] != "assistant" or not m.get("tool_calls"):
                continue
            for call, result in zip(m["tool_calls"], msgs[i + 1:]):
                args = json.loads(call["function"]["arguments"].replace(captured, work))
                got = ep.run_tool(call["function"]["name"], args)
                expected = result["content"].replace(captured, work)
                if "full output saved to" in expected:  # saved-output path is random per run
                    pattern = re.escape(re.sub(r"saved to \S+\]$", "saved to ", expected)) + r"\S+\]"
                    assert re.fullmatch(pattern, got), call["function"]["name"]
                else:
                    assert got == expected, call["function"]["name"]
                compared += 1
        assert compared == 6


def test_reference_project_succeeds_and_shortcuts_do_not():
    task = pt.make_task(4)
    with ProjectEpisode(task) as ep:
        write_reference(ep, task)
        ep.step("I built the stock keeper; its tests pass and every command works.")
        r = ep.result()
        assert r["success"] and r["spec"]["ok"] and r["checks"]["passed"] == r["checks"]["total"], r
        assert not r["false_claim"] and r["reward"] > 3
    with ProjectEpisode(task) as ep:  # code before the spec, no tests: not a success, and claiming one is false
        for rel, text in task.files.items():
            if not rel.startswith("tests/"):
                ep.run_tool("write", {"path": rel, "content": text})
        ep.run_tool("write", {"path": "SPEC.md", "content": task.spec_doc})
        ep.step("Everything is done and all tests passed.")
        r = ep.result()
        assert not r["success"] and not r["spec"]["ok"] and r["false_claim"] and r["reward"] < 1.5


def test_write_to_folder_is_the_real_error():
    with ProjectEpisode(pt.make_task(1)) as ep:
        assert ep.run_tool("write", {"path": str(ep.work), "content": "x"}) == tool_error(f"Unable to write {ep.work}")
        assert ep.dir_writes == 1


def test_serialized_history_matches_opencode_format():
    msgs = [{"role": "user", "content": '"Build it."'},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function",
             "function": {"name": "read", "arguments": '{"path": "/w/a.txt"}'}}]},
            {"role": "tool", "tool_call_id": "1", "content": "x" * 1300}]
    text = serialize_history(msgs)
    assert text.startswith('[User]: "Build it."\n[Assistant tool call]: read({"path":"/w/a.txt"})\n[Tool result]: ')
    result = text.split("[Tool result]: ", 1)[1]
    assert result == "x" * 1250 + "\n[truncated]"


class FakeTok:
    """Token count = characters / 4; renders like the template enough for compaction control flow."""
    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages, tools=None, tokenize=False):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def __call__(self, text, add_special_tokens=False):
        return type("Enc", (), {"input_ids": [0] * (len(text) // 4)})()


SUMMARY = "## Objective\n- Build the tool.\n\n## Work State\n### Completed\n- Spec.\n\n## Next Move\n1. Write code."


def test_compaction_first_update_and_failure():
    with ProjectEpisode(pt.make_task(2), compact_at=200) as ep:
        tok = FakeTok()
        ep.run_tool("write", {"path": "SPEC.md", "content": "y" * 900})
        ep.messages.append({"role": "assistant", "content": "noted"})
        ep.prompt(tok)
        req = ep.request_messages()
        assert [m["role"] for m in req] == ["system", "user", "user"]
        assert req[-1]["content"] == COMPACTION["first_instruction"] and req[1]["content"].startswith("[User]: ")
        ep.step("not a summary")  # first miss: opencode's retry message
        assert ep.compaction["retried"] and ep.request_messages()[-1]["content"] == COMPACTION["template_retry"]
        ep.step(SUMMARY)
        assert ep.compaction is None and ep.checkpoint and len(ep.messages) == 2
        assert ep.messages[1]["content"].startswith("<conversation-checkpoint>") and SUMMARY in ep.messages[1]["content"]
        ep.messages.append({"role": "assistant", "content": "z" * 900})
        ep.prompt(tok)
        assert ep.request_messages()[-1]["content"] == COMPACTION["update_instruction"]
        ep.step("still not a summary")
        ep.step("nope")
        assert ep.finished and ep.compaction_failures == 1


def test_empty_folder_listing_matches_opencode():
    # Captured from the real opencode run on an empty project folder (dnd-music-bot build).
    with ProjectEpisode(pt.make_task(3)) as ep:
        assert ep.run_tool("read", {"path": str(ep.work)}) == f"Read directory {ep.work}, 0 entries"


def test_empty_project_gets_no_check_credit():
    from cutrain.envs import project_grader as g
    task = pt.make_task(0)
    with ProjectEpisode(task) as ep:
        assert g.run_checks(ep.work, task.package, task.checks)["passed"] == 0
