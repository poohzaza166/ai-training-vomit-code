"""Self-check for prepare_data conversion: python "computer use training/test_prepare.py" """
import random
from collections import Counter

import pytest

try:
    from cutrain.dataprep.prepare_data import map_call, step_rows
except ImportError:
    # prepare_data imports the sibling assistant-harness checkout's src/ (HARNESS_ROOT); without it
    # this self-check cannot run, so skip rather than abort the whole test suite.
    pytest.skip("assistant-harness checkout (src/) not present", allow_module_level=True)

assert map_call("computer", {"action": "left_click", "coordinate": [5, 6]}, {}) == ("click", {"x": 5, "y": 6})
assert map_call("computer", {"action": "key", "text": "ctrl+Return"}, {}) == ("hotkey", {"combo": "ctrl+enter"})
state: dict = {}
map_call("computer", {"action": "mouse_move", "coordinate": [9, 9]}, state)
assert map_call("computer", {"action": "scroll", "scroll_direction": "down", "scroll_amount": 2}, state) == \
    ("scroll", {"amount": -2, "x": 9, "y": 9})
assert map_call("computer", {"action": "triple_click", "coordinate": [1, 1]}, {}) is None
assert map_call("Bash", {"command": "ls"}, {}) == ("run_command", {"cmd": "ls"})


def turn(i):  # one assistant tool call + its result, ~300 chars of JSON
    return [{"role": "assistant", "content": [{"type": "tool_use", "id": f"t{i}", "name": "Bash", "input": {"command": f"echo {i}" + " " * 150}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"t{i}", "content": "x" * 100}]}]


msgs = [{"role": "user", "content": "go"}] + [m for i in range(10) for m in turn(i)] + [{"role": "assistant", "content": "all done"}]
rows = list(step_rows([("c", "test", "execution", "sys", None, msgs, {"role": "assistant", "content": "nope"})],
                      random.Random(0), Counter(), max_chars=1000))
assert len(rows) == 11
for r in rows:
    assert [m["role"] for m in r["prompt"][:2]] == ["system", "user"]  # head always kept
    assert r["prompt"][2:3] == [] or r["prompt"][2]["role"] == "assistant"  # no orphan tool result after trim
    assert r["chosen"] != r["rejected"]
assert len(rows[-1]["prompt"]) < len(msgs)  # history window trimmed
assert rows[-1]["rejected"] == [{"role": "assistant", "content": "nope"}] and rows[-1]["rejected_origin"] == "dataset"
assert [t["function"]["name"] for t in rows[0]["tools"]] == ["run_command"]  # no GUI -> only harness tools used


# failed call + successful retry -> error pair; user pushback -> feedback pair
def call(i, cmd):
    return {"role": "assistant", "content": [{"type": "tool_use", "id": f"e{i}", "name": "Bash", "input": {"command": cmd}}]}


def result(i, text, err=False):
    return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"e{i}", "content": text, "is_error": err}]}


msgs = [{"role": "user", "content": "build it"}, call(0, "make"), result(0, "make: no rule", True), call(1, "make all"),
        result(1, "ok"), {"role": "assistant", "content": "Built with make all."},
        {"role": "user", "content": "no, I wanted cmake"}, call(2, "cmake ."), result(2, "ok"),
        {"role": "assistant", "content": "done with cmake"}]
rows = {r["rejected_origin"]: r for r in step_rows([("e", "test", "execution", "sys", None, msgs, None)],
                                                    random.Random(0), Counter(), 10**6)}
args = lambda r, k: r[k][0]["tool_calls"][0]["function"]["arguments"]  # noqa: E731
assert args(rows["outcome:error"], "chosen") == {"cmd": "make all"} and args(rows["outcome:error"], "rejected") == {"cmd": "make"}
fb = rows["feedback:user_correction"]
assert fb["rejected"][0]["content"] == "Built with make all." and args(fb, "chosen") == {"cmd": "cmake ."}
print("ok")
