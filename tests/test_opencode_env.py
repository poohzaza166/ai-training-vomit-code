import json
import re

import pytest

from cutrain.envs.opencode_env import OpencodeEpisode, TEMPLATE, tool_error
from cutrain.paths import REPO

DATA = REPO / "data"
# Captured from real opencode runs: (file, seed, audit lines, setup)
CAPTURES = [("opencode_capture_success.json", 6000001, 200, None),
            ("opencode_capture_errors.json", 6000002, 200, None),
            ("opencode_capture_edge.json", 6000003, 20, "dup")]


@pytest.mark.parametrize("name,seed,audit,setup", CAPTURES)
def test_tool_results_match_real_opencode(name, seed, audit, setup):
    req = json.loads((DATA / name).read_text())
    captured_work = re.search(r"Working directory: (\S+)", req["messages"][0]["content"]).group(1)
    with OpencodeEpisode(seed, audit_lines=audit) as ep:
        if setup == "dup":
            (ep.work / "dup.txt").write_text("a\nb\na\n")
        work = str(ep.work)
        assert ep.messages[1]["content"] == req["messages"][1]["content"]
        assert ep.tools == req["tools"]
        expected_system = req["messages"][0]["content"].replace(captured_work, work)
        assert re.sub(r"Today's date: .+", "", ep.messages[0]["content"]) == re.sub(r"Today's date: .+", "", expected_system)
        compared = 0
        msgs = req["messages"][2:]
        for i, m in enumerate(msgs):
            if m["role"] != "assistant" or not m.get("tool_calls"):
                continue
            results = {r["tool_call_id"]: r["content"] for r in msgs[i + 1:i + 1 + len(m["tool_calls"])]}
            for call in m["tool_calls"]:
                args = json.loads(call["function"]["arguments"].replace(captured_work, work))
                got = ep.run_tool(call["function"]["name"], args)
                expected = results[call["id"]].replace(captured_work, work)
                # The emulator keeps Python's bytecode cache off the workspace (stale-.pyc guard), so
                # directory listings lack the __pycache__ entry real opencode shows after a Python run.
                if expected.startswith("Read directory ") or call["function"]["name"] == "shell":
                    expected = expected.replace("__pycache__/\n", "").replace("__pycache__\n", "")
                    expected = re.sub(r"entries 1-(\d+)", lambda m: f"entries 1-{got.count(chr(10))}", expected) \
                        if expected.startswith("Read directory ") else expected
                assert got == expected, call["function"]["name"]
                compared += 1
        assert compared >= 5


def test_oracle_solution_succeeds_and_honest_failure_is_not_false_claim():
    with OpencodeEpisode(42) as ep:
        for turn in ep.oracle_calls()[:-1]:
            for name, args in turn:
                ep.run_tool(name, args)
        ep.step(ep.oracle_calls()[-1])
        r = ep.result()
        assert r["success"] and r["verified"] and r["tests"] == 1 and not r["false_claim"]
    with OpencodeEpisode(42) as ep:
        ep.step("I could not finish: verify.py failed.")
        r = ep.result()
        assert not r["success"] and not r["false_claim"] and r["reward"] == 0
    with OpencodeEpisode(42) as ep:
        ep.step("Done, all tests passed.")
        r = ep.result()
        assert r["false_claim"] and r["reward"] < 0


def test_verification_is_reset_by_breaking_edit():
    with OpencodeEpisode(42) as ep:
        calls = ep.oracle_calls()
        for turn in calls[:-1]:
            for name, args in turn:
                ep.run_tool(name, args)
        assert ep.verified
        ep.run_tool("write", {"path": "pipeline.py", "content": "def evaluate(x):\n    return [x]\n"})
        assert not ep.verified


def test_step_parses_native_calls_and_paths_stay_in_workspace():
    with OpencodeEpisode(42) as ep:
        ep.step("<think></think>\n<tool_call>\n<function=read>\n<parameter=path>\n/etc/passwd\n</parameter>\n"
                "</function>\n</tool_call>")
        assert ep.messages[-1]["content"] == tool_error("File not found: /etc/passwd")
        assert ep.messages[-2]["tool_calls"][0]["function"]["name"] == "read"
        ep.step("<tool_call>\n<function=shell>\n<parameter=command>\ncat /etc/hostname; ls /home\n</parameter>\n"
                "</function>\n</tool_call>")
        assert "Exited with code" in ep.messages[-1]["content"]


def test_prompt_renders_tools_and_history_like_the_server():
    from transformers import AutoTokenizer
    snapshot = json.loads((DATA.parent / "out/long-grpo-recovery/base_model.json").read_text())["snapshot"]
    tok = AutoTokenizer.from_pretrained(snapshot)
    with OpencodeEpisode(42) as ep:
        ep.step("<tool_call>\n<function=shell>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>")
        prompt = ep.prompt(tok)
        assert "<function>\n<name>shell</name>" in prompt and prompt.endswith("<|im_start|>assistant\n")
        assert "<tool_response>\naudit.log" in prompt
        assert TEMPLATE["tools"][0]["function"]["name"] in prompt


def test_rapid_same_size_rewrite_runs_current_code():
    # Regression: a same-second, same-size rewrite once ran a stale .pyc (oracle seed 3000006).
    with OpencodeEpisode(3000006) as ep:
        calls = ep.oracle_calls()
        good = calls[2][0][1]["newString"]
        wrong = good.replace("x = x * 7", "x = x + 7")
        assert len(wrong) == len(good) and wrong != good
        ep.run_tool("write", {"path": "pipeline.py", "content": wrong + "\n"})
        assert "Exited with code 1" in ep.run_tool("shell", {"command": "python3 verify.py"})
        ep.run_tool("write", {"path": "pipeline.py", "content": good + "\n"})
        assert ep.run_tool("shell", {"command": "python3 verify.py"}) == "3 tests passed\n"
        assert ep.base.grade() == 1.0
