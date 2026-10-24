import json
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from cutrain.paths import REPO
from cutrain.training.anti_loop import token_loops, apply_loop_reward, loop_preferences
from cutrain.envs.agent_grpo_env import Episode, action_text
from cutrain.dataprep.export_stopping_sft import build_row
from cutrain.training.train_loop_preferences import orpo_loss, preference_example
from cutrain.training.train_long_grpo import group_advantages, clipped_loss


def detect(text):
    return token_loops(list(text), "".join)


def test_detects_cycles_inside_response_not_just_suffix():
    assert detect("x" * 64)["loop_tokens"] == 64
    assert detect("First: " + "I will retry. " * 10 + " End.")["loop_tokens"] >= 64
    assert detect('{"tool":"x"}' * 8)["loop_tokens"] >= 64
    assert not detect("A short normal response without EOS.")["loop_tokens"]


def test_legitimate_json_code_whitespace_are_exempt():
    for text in [" " * 200, ")" * 100, json.dumps({"cmd": "print(1)\n" * 30}),
                 '["same", "same", "same"]', '```python\n' + 'print(1)\n' * 30 + '```',
                 '{"cmd":"' + 'print(1)\\n' * 30]:
        assert not detect(text)["loop_tokens"]


def test_loop_penalty_is_capped_and_length_limit_is_not_repetition():
    record = dict(loop_tokens=64, message_index=2, completion="x"*64, truncated=True)
    r = apply_loop_reward({"reward": 2.46}, [record]*6, 0.5)
    assert r["reward"] == pytest.approx(1.96)
    assert r["token_loop_penalty"] == 0.5
    record["loop_tokens"] = 0
    assert apply_loop_reward({"reward": 0.0}, [record], .5)["token_loop_penalty"] == 0


def test_penalty_gradient_reduces_loop_probability_without_sampling_filters():
    record = dict(loop_tokens=64, message_index=1, completion="retry "*20, truncated=True)
    bad = apply_loop_reward({"reward": 0.0}, [record], .5)["reward"]
    advantages = group_advantages([bad, 0.0])
    logits = torch.zeros(2, requires_grad=True)
    lp = logits.log_softmax(0)
    old = lp.detach()
    loss = sum(clipped_loss(lp[i:i+1], old[i:i+1], advantages[i], 2) for i in range(2))
    loss.backward()
    assert logits.grad[0] > 0 and logits.grad[1] < 0
    updated = (logits.detach() - .1 * logits.grad).softmax(0)
    assert updated[0] < .5  # gradient descent lowers the looping alternative's probability


def test_verifying_after_edit_is_not_a_loop_but_unchanged_repeats_are():
    with Episode(42) as ep:
        verify = action_text(ep.oracle()[2])
        ep.step(verify)
        ep.step(action_text(ep.oracle()[1]))
        ep.step(verify)
        assert ep.repeats == 0
        ep.step(verify)
        assert ep.repeats == 1
        # Unrelated junk does not reset task-state repeat detection.
        ep.step(action_text({"tool":"run_command","args":{"cmd":"echo junk > unrelated.txt"}}))
        ep.step(verify)
        assert ep.repeats == 2


def test_interrupted_output_cannot_execute_even_valid_json():
    with Episode(42) as ep:
        ep.reject_generation(action_text({"tool":"run_command","args":{"cmd":"touch SHOULD_NOT_EXIST"}}))
        assert not (ep.work / "SHOULD_NOT_EXIST").exists()
        assert ep.invalid == 1 and not ep.finished


@pytest.mark.parametrize("mode", ["clean", "recovery", "stop_now", "blocked"])
def test_stopping_examples_are_verified_and_label_only_good_actions(mode):
    row = build_row(800042, mode)
    assert row["verification"]["success"] == (mode != "blocked")
    labels = [row["messages"][i] for i in row["assistant_loss_indices"]]
    assert all(m["role"] == "assistant" for m in labels)
    assert labels[-1]["tool_calls"][0]["function"]["name"] == "task_done"
    assert row["tools"] and all(m.get("tool_calls") for m in labels)
    assert len(labels) == (1 if mode in ("stop_now", "blocked") else 4)
    if mode == "blocked":
        assert row["verified_blocker"]["shell_exited"] and not row["verification"]["premature_stop"]


def test_orpo_sequential_gradient_matches_joint_and_discourages_loop():
    weights = torch.tensor([-1.2, -0.8], requires_grad=True)
    joint = orpo_loss(weights[0], weights[1], .1)
    expected = torch.autograd.grad(joint, weights)[0]
    leaves = [x.detach().requires_grad_() for x in weights]
    coeffs = torch.autograd.grad(orpo_loss(*leaves, .1), leaves)
    for i, coefficient in enumerate(coeffs):
        (weights[i] * coefficient).backward()
    assert torch.allclose(weights.grad, expected)
    assert weights.grad[0] < 0 and weights.grad[1] > 0
    assert torch.isfinite(orpo_loss(torch.tensor(-1e-8), torch.tensor(-100.), .1))


def test_preferences_require_observed_loop_and_exact_same_prefix():
    prompt = [{"role":"user","content":"do task"}]
    chosen = {"role":"assistant","content":None,"tool_calls":[{"id":"call_1","type":"function",
              "function":{"name":"run_command","arguments":json.dumps({"cmd":"pwd"})}}]}
    bad = {"role":"assistant","content":"retry "*80}
    good_result = dict(success=True, verified=True, intact=True, tests=1, token_loop_turns=0)
    bad_result = dict(success=False, token_loop_message_indices=[1], truncated_message_indices=[1],
                      loop_completions=[{"message_index":1,"content":bad["content"]}])
    pairs = loop_preferences([good_result, bad_result], [prompt+[chosen], prompt+[bad]])
    assert len(pairs) == 1 and not pairs[0]["rejected_eos"]
    assert loop_preferences([good_result, bad_result], [prompt+[chosen], [{"role":"user","content":"other"},bad]]) == []
    from transformers import AutoTokenizer
    snapshot = json.loads((REPO / "out/long-grpo-recovery/base_model.json").read_text())["snapshot"]
    tok = AutoTokenizer.from_pretrained(snapshot)
    args = SimpleNamespace(loop_min_tokens=64, max_context=4096)
    prefix, c, r = preference_example(pairs[0], tok, args)
    assert c[-1] == tok.eos_token_id and r[-1] != tok.eos_token_id
    # chosen is the template's native rendering of the transcript turn, prompt includes the tools
    assert tok.decode(c[:-1]) == action_text({"tool": "run_command", "args": {"cmd": "pwd"}})
    assert "<tools>" in tok.decode(prefix)
    fake = deepcopy(pairs[0]); fake["rejected"][0]["content"] = "short ordinary error"
    with pytest.raises(ValueError, match="no detected loop"):
        preference_example(fake, tok, args)


def test_native_parameter_values_are_exempt_like_json_strings():
    code = "print(1)\n" * 30
    assert detect(code)["loop_tokens"] >= 64
    assert detect(action_text({"tool": "run_command", "args": {"cmd": code}}))["loop_tokens"] == 0
