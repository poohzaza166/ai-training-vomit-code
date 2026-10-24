"""mimo_env / prepare_mix_grpo: label extraction, scoring and MiMo's reward contracts. The docker test runs
only with MIMO_DOCKER_TEST=1 (it needs docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-001457)."""
import os

import pytest

from cutrain.envs.mimo_env import (MathEpisode, math_score, parse_reward_files, patch_touched_files,
                                   reset_test_files_cmd, hub_image)
from cutrain.dataprep.prepare_mix_grpo import math_label

PATCH = """diff --git a/src/a.test.js b/src/a.test.js
index 1..2 100644
--- a/src/a.test.js
+++ b/src/a.test.js
@@ -1 +1,2 @@
 x
+y
diff --git a/mimo_test_command.sh b/mimo_test_command.sh
new file mode 100644
--- /dev/null
+++ b/mimo_test_command.sh
@@ -0,0 +1 @@
+npx jest src/a.test.js
"""


def test_patch_touched_files():
    assert patch_touched_files(PATCH) == ["src/a.test.js", "mimo_test_command.sh"]


def test_reset_cmd_covers_every_touched_file():
    cmd = reset_test_files_cmd(PATCH, "f" * 40)
    assert "src/a.test.js\nmimo_test_command.sh\nEOF_RESET_TEST_FILES" in cmd
    assert reset_test_files_cmd("", "f" * 40) is None
    assert reset_test_files_cmd(PATCH, "") is None


def test_reward_files():
    assert parse_reward_files('{"reward": 0.5, "x": 1}', None) == 0.5
    assert parse_reward_files('{"only": 1}', None) == 1.0
    assert parse_reward_files(None, " 1 \n") == 1.0
    for bad in (('{}', None), (None, ""), (None, None), ('{"a": 1, "b": 2}', None)):
        with pytest.raises(ValueError):
            parse_reward_files(*bad)


def test_hub_image():
    assert hub_image("format-code-task-001457:latest") == \
        "docker.io/xiaomimimo/mimo-v2.6-rl-oss:format-code-task-001457"
    assert hub_image("general-agent-env-12:oss").endswith(":general-agent-env-12")


def test_math_label_boxed_and_choice():
    assert math_label("What is 2+3?", "<think>easy</think>\nThe answer is \\boxed{5}.") == \
        {"type": "boxed", "answer": "5"}
    # Two different boxed answers: no single label.
    assert math_label("q", "<think>.</think> \\boxed{1} or \\boxed{2}") is None
    prompt = ("Q: Beaches and deserts collect large deposits of what?  Choices:  - animals  - plants  "
              "- water  - sand   A: Answer:")
    lab = math_label(prompt, "<think>sand it is</think>\n\nsand")
    assert lab["type"] == "choice" and lab["answer"] == "sand" and "water" in lab["choices"]
    # The answer names two choices: ambiguous, dropped.
    assert math_label(prompt, "<think>.</think> water or sand") is None
    # A boxed answer inside the reasoning only does not count.
    assert math_label("q", "<think>\\boxed{3}</think> I am not sure.") is None


def test_math_score():
    boxed = {"type": "boxed", "answer": "1,024"}
    assert math_score("<think>...</think> so \\boxed{1024}", boxed) == (1.0, True)
    assert math_score("<think>\\boxed{1024}</think> no idea", boxed) == (0.0, False)
    # The model's own habit: a '# Answer' section and no box. Right answer, no format bonus.
    assert math_score("so it is 364.\n# Answer\n\n364", {"type": "boxed", "answer": "364"}) == (1.0, False)
    assert math_score("Final answer: 12", {"type": "boxed", "answer": "12"}) == (1.0, False)
    choice = {"type": "choice", "answer": "sand", "choices": ["animals", "plants", "water", "sand"]}
    assert math_score("\\boxed{Sand}", choice) == (1.0, True)
    assert math_score("It is sand.", choice) == (1.0, False)
    assert math_score("water and sand", choice) == (0.0, False)


def test_math_episode_rewards():
    task = {"id": "t", "kind": "deepseek", "problem": "2+3?", "label": {"type": "boxed", "answer": "5"}}
    ep = MathEpisode(task)
    ep.step("<think></think>\\boxed{5}")
    assert ep.finished and ep.result()["reward"] == pytest.approx(2.2) and ep.result()["success"]
    ep = MathEpisode(task)
    ep.reject_generation("<think>long long")
    r = ep.result()
    assert r["reward"] == pytest.approx(-0.3) and not r["success"] and r["truncated"]


@pytest.mark.skipif(os.environ.get("MIMO_DOCKER_TEST") != "1", reason="needs docker and the MiMo image")
def test_code_task_tools_and_verifier():
    """Real container: tools work, untouched task scores 0, a planted file at a patch-added path does
    not break git apply (MiMo's reset contract)."""
    import json
    import pandas as pd
    from cutrain.envs.mimo_env import MimoEpisode
    c = pd.read_parquet("data/mimo/code.parquet")
    ij = next(json.loads(e["instance_json"]) for e in c.extra_info
              if json.loads(e["instance_json"])["instance_id"] == "format-code-task-001457")
    task = {"id": ij["instance_id"], "kind": "code", "image": hub_image(ij["docker_image"]), "cwd": ij["cwd"],
            "problem": ij["problem_statement"], "test_patch": ij["test_patch"],
            "test_command": ij["test_command"], "verifier_timeout": ij["verifier_timeout_sec"]}
    with MimoEpisode(task) as ep:
        assert ep.run_tool("read", {"path": "package.json"}).startswith("Read file package.json, lines 1-")
        assert "src/" in ep.run_tool("read", {"path": "."})
        assert ep.run_tool("write", {"path": "notes/x.txt", "content": "hello\n"}).startswith("Created")
        assert ep.run_tool("edit", {"path": "notes/x.txt", "oldString": "hello", "newString": "bye"}) \
            == "Edited notes/x.txt (1 replacement)"
        assert "bye" in ep.run_tool("shell", {"command": "cat notes/x.txt"})
        assert "Exited with code 3" in ep.run_tool("shell", {"command": "exit 3"})
        assert "notes/x.txt" in ep.run_tool("glob", {"pattern": "**/*.txt"})
        assert "Found 1 matches" in ep.run_tool("grep", {"pattern": "^bye$", "path": "notes"})
        assert ep.run_tool("shell", {"command": "curl -sS -m 5 https://example.com >/dev/null"}) \
            .endswith(tuple(f"Exited with code {c}" for c in (6, 7, 127)))  # no network
        ep.run_tool("write", {"path": "mimo_test_command.sh", "content": "exit 0\n"})  # planted verifier
        v = ep.verify()
    assert v["score"] == 0.0 and not v.get("error"), v
