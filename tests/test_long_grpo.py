import json
from types import SimpleNamespace
import pytest
import torch
from cutrain.envs.agent_grpo_env import Episode, action_text, parse_action
from cutrain.paths import REPO
from cutrain.serving.serve_adapter import render_turn
from cutrain.training.train_long_grpo import add_feedback, group_advantages, clipped_loss, log_probs, policy_mode, render, create_episode, sft_examples


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    snapshot = json.loads((REPO / "out/long-grpo-recovery/base_model.json").read_text())["snapshot"]
    return AutoTokenizer.from_pretrained(snapshot)


def test_parse_rejects_multiple_and_malformed_calls():
    call = action_text({"tool": "run_command", "args": {"cmd": "pwd"}})
    bare = call.replace("<think></think>", "")
    assert parse_action(call) == {"tool": "run_command", "args": {"cmd": "pwd"}}
    assert parse_action("<think>check first</think>I will look.\n" + bare)  # the template allows text before
    assert parse_action(call + " trailing") is None
    assert parse_action(call + bare) is None
    assert parse_action("<think>not closed " + bare) is None
    assert parse_action(action_text({"tool": "run_command", "args": {"cmd": "pwd", "extra": 1}})) is None
    assert parse_action(action_text({"tool": "task_done", "args": {"summary": "ok", "success": "maybe"}})) is None
    done = parse_action(action_text({"tool": "task_done", "args": {"summary": "ok", "success": False}}))
    assert done["args"]["success"] is False  # the template prints False; it must not come back truthy
    assert parse_action('{"tool":"run_command","args":{"cmd":"pwd"}}') is None  # the retired JSON protocol


def test_oracle_text_is_the_templates_own_rendering(tok):
    with Episode(42) as ep:
        ep.seed_history()
        assert "<tools>" in ep.prompt(tok) and "Reply with exactly one JSON" not in ep.prompt(tok)
        for a in ep.oracle():
            ep.step(action_text(a))
            i = max(j for j, m in enumerate(ep.messages) if m["role"] == "assistant")
            assert render_turn(tok, ep.messages, i, ep.tools)[1] == action_text(a)
        assert ep.result()["success"]


def test_same_history_renders_identically_across_group_members():
    with Episode(42) as a, Episode(42) as b:
        for ep in (a, b):
            ep.seed_history()
            ep.step(action_text(ep.oracle()[0]))
        assert a.messages == b.messages


@pytest.mark.parametrize("seed", [3, 19, 1000001])
def test_complete_real_terminal_episode(seed):
    with Episode(seed, depth=5) as ep:
        ep.seed_history()
        for a in ep.oracle():
            ep.step(action_text(a))
        result = ep.result()
        assert result["success"]
        assert result["tests"] == 1
        assert result["reward"] == pytest.approx(2.46)
        assert result["public_check"] == {"mismatches": [], "first_wrong_stage": None, "returns_list": True}


def test_false_claim_and_verification_gate():
    with Episode(42) as ep:
        ep.step(action_text(ep.oracle()[-1]))
        result = ep.result()
        assert not result["success"]
        assert result["false_claim"]
        assert result["reward"] < 0
    with Episode(42) as ep:
        actions = ep.oracle()
        for a in (actions[1], actions[3]):
            ep.step(action_text(a))
        assert ep.result()["tests"] == 1
        assert not ep.result()["success"]


def test_tampering_cannot_pass():
    with Episode(42) as ep:
        for a in ep.oracle()[:2]:
            ep.step(action_text(a))
        ep.step(action_text({"tool":"run_command", "args":{"cmd":"echo 'print(42)' > verify.py"}}))
        for a in ep.oracle()[2:]:
            ep.step(action_text(a))
        assert not ep.result()["intact"]
        assert not ep.result()["success"]


def test_terminal_is_persistent_and_isolated():
    with Episode(1) as ep:
        out, rc = ep.shell.run("export CHECK_VALUE=42")
        assert rc == 0
        out, rc = ep.shell.run('echo "$CHECK_VALUE"; test ! -e /home/pooh; test ! -e /workspace/../home')
        assert rc == 0 and "42" in out
        out, rc = ep.shell.run("python3 -c 'import socket; s=socket.socket(); s.settimeout(.1); s.connect((\"1.1.1.1\",80))'")
        assert rc != 0


def test_shell_timeout_terminates_episode_shell():
    with Episode(2) as ep:
        out, rc = ep.shell.run("sleep 60", timeout=0.2)
        assert rc == 124
        assert ep.shell.proc.poll() is not None


def test_zero_variance_and_advantage_direction():
    assert group_advantages([1, 1, 1]) == [0, 0, 0]
    assert group_advantages([0, 2]) == [-1, 1]
    lp = torch.zeros(2, requires_grad=True)
    loss = clipped_loss(lp, torch.zeros(2), 1.0, 4)
    loss.backward()
    assert torch.allclose(lp.grad, torch.full((2,), -0.25))


def test_clipping_and_fixed_length_normalization():
    new = torch.tensor([1.0], requires_grad=True)
    clipped_loss(new, torch.zeros(1), 1.0, 4).backward()
    assert new.grad.item() == 0
    a = clipped_loss(torch.zeros(2), torch.zeros(2), 1.0, 8)
    b = clipped_loss(torch.zeros(4), torch.zeros(4), 1.0, 8)
    assert b == 2 * a


def test_chunk_log_probs_and_gradients_match_full_logits():
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(17, 8)
        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(last_hidden_state=self.emb(input_ids))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()
            self.lm_head = torch.nn.Linear(8, 17)
        def get_base_model(self):
            return self
    torch.manual_seed(7)
    model = Model()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    got = log_probs(model, ids, 3, chunk=2, offload=False)
    got.sum().backward()
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    logits = model.lm_head(model.model(ids).last_hidden_state[:, 2:-1]).float()
    want = torch.log_softmax(logits, -1).gather(-1, ids[:, 3:, None]).squeeze()
    want.sum().backward()
    assert torch.allclose(got, want, atol=1e-6)
    for p, g in zip(model.parameters(), grads):
        assert torch.allclose(p.grad, g, atol=1e-6)


def test_early_user_override_is_required():
    with Episode(42, audit_lines=100) as ep:
        ep.seed_history()
        assert f"add {ep.offset}" in ep.messages[1]["content"]
        assert "offset" not in json.loads((ep.work / "contract.json").read_text())
        action = ep.oracle()[1]
        action["args"]["cmd"] = action["args"]["cmd"].replace(f"    x += {ep.offset}\n", "")
        ep.step(action_text(action))
        assert ep.grade() < 1.0


def test_render_does_not_open_unfinished_thinking():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": False}
            return "history\n"
    assert render(Tokenizer(), []) == "history\n<|im_start|>assistant\n"


def test_cpu_embedding_does_not_select_cpu_kernels(monkeypatch):
    import kernels
    calls = []
    monkeypatch.setattr(kernels, "kernelize", lambda base, **kwargs: calls.append(kwargs))
    class Model(torch.nn.Module):
        use_kernels = True
        device = torch.device("cpu")
        def get_base_model(self):
            return self
    model = Model()
    policy_mode(model, False)
    policy_mode(model, True)
    assert all(c["device"].type == "cuda" for c in calls)
    assert calls[0]["mode"] == kernels.Mode.INFERENCE
    assert calls[1]["mode"] == kernels.Mode.TRAINING


def test_context_construction_is_bounded(monkeypatch):
    from cutrain.training import train_long_grpo
    closed = []
    class TinyEpisode:
        messages = []
        def __init__(self, *args, **kwargs): pass
        def seed_history(self): pass
        def prompt(self, tok): return ""
        def close(self): closed.append(True)
    class TinyTokenizer:
        def apply_chat_template(self, *args, **kwargs): return ""
        def encode(self, *args, **kwargs): return [0]
    monkeypatch.setattr(train_long_grpo, "Episode", TinyEpisode)
    args = SimpleNamespace(depth=3, max_new_tokens=10, max_context=2048)
    with pytest.raises(ValueError, match="bounded context"):
        create_episode(42, args, TinyTokenizer(), 1024)
    assert len(closed) == 6


def test_sft_labels_exclude_seeded_context_and_observations(tok):
    with Episode(42) as ep:
        ep.seed_history()
        first = len(ep.messages)
        write = ep.oracle()[1]
        for a in ep.oracle()[:2]:
            ep.step(action_text(a))
        row = {"messages": ep.messages, "tools": ep.tools, "assistant_loss_indices": [first + 2]}
    [(prompt, completion, turns)] = sft_examples(tok, None, 0, [row])
    assert turns == 1 and "<tools>" in prompt and prompt.endswith("<|im_start|>assistant\n")
    assert "cat contract.json" in prompt and completion == action_text(write)
    row["assistant_loss_indices"] = [first + 1]
    with pytest.raises(ValueError, match="loss index"):
        list(sft_examples(tok, None, 0, [row]))


def test_public_check_reports_only_verify_cases_and_errors():
    with Episode(42) as ep:
        check = ep.result()["public_check"]
        assert [m["x"] for m in check["mismatches"]] == [-3, 0, 5]
        assert all(m["got"] == [m["x"]] for m in check["mismatches"])
        assert check["first_wrong_stage"]["number"] == 1 and check["returns_list"]
        ep.step(action_text(ep.oracle()[1]))
        (ep.work / "pipeline.py").write_text(ep.final_pipeline().replace("    values.append(x)\n    return", "    return"))
        check = ep.public_check()
        assert check["first_wrong_stage"] == {"number": 3, **ep.stages[2]}
        (ep.work / "pipeline.py").write_text("def evaluate(x):\n    raise KeyError('boom')\n")
        assert "KeyError" in ep.public_check()["error"]


def test_final_pipeline_does_not_follow_symlinks(tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("host secret")
    with Episode(42) as ep:
        (ep.work / "pipeline.py").unlink()
        (ep.work / "pipeline.py").symlink_to(secret)
        assert "host secret" not in ep.final_pipeline()


def test_feedback_extends_task_message_without_new_turn():
    with Episode(42) as ep:
        ep.seed_history()
        before = [dict(m) for m in ep.messages]
        add_feedback(ep, None)
        assert ep.messages == before
        add_feedback(ep, "Facts: stage 2 is wrong.")
        assert len(ep.messages) == len(before)
        assert ep.messages[1]["content"] == before[1]["content"] + "\n\nFacts: stage 2 is wrong."
        assert ep.messages[2:] == before[2:]
