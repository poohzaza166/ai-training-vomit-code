import json
from copy import deepcopy
import pytest
from cutrain.training.agent_critique import parse_review, make_feedback, harvest_verified, TrajectorySelfCritic, grounded, evidence_text


@pytest.mark.parametrize("raw", ['{}', 'not JSON', '{"verdict":"SOUND","flaws":["bad"]}',
                                 '{"verdict":"FLAWED","flaws":[]}',
                                 '{"verdict":"SOUND","flaws":""}'])
def test_malformed_critic_is_not_sound(raw):
    with pytest.raises((ValueError, TypeError)):
        parse_review(raw)


def result(success=False):
    return dict(success=success, verified=success, intact=True, tests=float(success),
                false_claim=not success, reward=2.46 if success else -1,
                initial_message_count=2)


def test_critic_approval_cannot_override_failed_verification():
    r = result()
    original = deepcopy(r)
    feedback = make_feedback(r, {"status": "ok", "verdict": "SOUND", "flaws": []})
    assert "failed independent verification" in feedback and "success=true although" in feedback
    assert r == original  # critique cannot mutate reward or success


def test_failed_review_preserves_objective_feedback():
    feedback = make_feedback(result(), {"status": "error", "flaws": []})
    assert "failed independent verification" in feedback and "reviewer" not in feedback
    critic = object.__new__(TrajectorySelfCritic)
    critic.memory = []
    critic.ask = lambda *args: "garbage"
    review = critic.review([], result())
    assert review["status"] == "error" and "verdict" not in review


def trajectories():
    prompt = [{"role": "system", "content": "tools"}, {"role": "user", "content": "task"}]
    return [prompt + [{"role": "assistant", "content": "fix"}, {"role": "tool", "content": "pass"},
                      {"role": "assistant", "content": "done"}],
            prompt + [{"role": "assistant", "content": "bad"}, {"role": "tool", "content": "fail"}]]


def test_harvest_requires_verified_success_and_masks_tools():
    ts = trajectories()
    assert harvest_verified([result(), result()], ts) == ([], [])
    sft, pairs = harvest_verified([result(True), result()], ts)
    assert sft[0]["assistant_loss_indices"] == [2, 4]
    assert len(pairs) == 1 and pairs[0]["chosen_verification"]["success"]
    invalid = result(True)
    invalid["verified"] = False
    assert harvest_verified([invalid, result()], ts) == ([], [])


def test_preference_pairs_require_identical_conditioning():
    ts = trajectories()
    ts[1] = deepcopy(ts[1])
    ts[1][1]["content"] = "different task or feedback"
    with pytest.raises(ValueError, match="different conditioning"):
        harvest_verified([result(True), result()], ts)


def failed_with_check():
    r = result()
    r.update(final_pipeline="def evaluate(x):\n    return [max(x, 6)]\n", end_reason="turn_budget",
             public_check={"mismatches": [{"x": -3, "got": [9, 9, 9], "expected": [4, 9, 54]}],
                           "first_wrong_stage": {"number": 1, "op": "floor", "value": 4},
                           "returns_list": True})
    return r


def test_feedback_states_correct_facts_only_and_labels_critic_notes():
    feedback = make_feedback(failed_with_check(), {"status": "ok", "flaws": ["The add stage is missing."]})
    assert "stage 1: x = max(x, 4) (the BEFORE-stage-1 override is applied first)" in feedback
    assert "verify.py requires evaluate(-3) == [4, 9, 54]." in feedback
    # Failed code and wrong outputs are copied by the retry policy, so they never appear.
    assert "max(x, 6)" not in feedback and "[9, 9, 9]" not in feedback
    assert "ran out of turns" in feedback
    assert "Unverified reviewer notes" in feedback and "The add stage is missing." in feedback
    assert feedback.index("Facts") < feedback.index("Unverified")
    assert "Recheck" not in feedback and "finish honestly" not in feedback
    without = make_feedback(failed_with_check(), {"status": "ok", "flaws": ["x"]}, critic_notes=False)
    assert "reviewer" not in without
    assert make_feedback({**failed_with_check(), "success": True}, {"status": "ok", "flaws": []}) is None


def test_feedback_for_non_list_and_crashing_code():
    r = failed_with_check()
    r["public_check"] = {"mismatches": [{"x": 0, "got": 5, "expected": [1, 2]}],
                         "first_wrong_stage": {"number": 1, "op": "add", "value": 1}, "returns_list": False}
    assert "must return a list" in make_feedback(r, {"status": "error"})
    r["public_check"] = {"error": "NameError: name 'op' is not defined"}
    assert "NameError" in make_feedback(r, {"status": "error"})


def test_grounding_rejects_copied_values_identifiers_and_user_deferral():
    messages = [{"role": "assistant", "content": '{"cmd": "cat audit.log"}'},
                {"role": "tool", "content": "job=777 `phantom` [1, 2, 3]"},
                {"role": "assistant", "content": "x = max(x, 6)"},
                {"role": "tool", "content": "evaluate(-3) == [4, 9, 54]"}]
    ev = evidence_text(messages, {"tests": 0.0})
    assert grounded("evaluate(-3) should be [4, 9, 54], not max(x, 6).", ev)
    assert not grounded("It returns [4, 9, 9].", ev)
    assert not grounded("The audit job 777 matters.", ev)  # audit dump is not evidence
    assert not grounded("[1, 2, 3] is returned.", ev)
    assert not grounded("The code uses `op` undefined.", ev)
    assert not grounded("The user must provide the intended semantics.", ev)


def test_review_uses_no_cross_task_memory_and_drops_ungrounded_flaws():
    critic = object.__new__(TrajectorySelfCritic)
    critic.memory = ["The contract requires adding 9."]
    critic.seen = __import__("collections").Counter()
    prompts = []
    critic.ask = lambda system, user: prompts.append(system) or json.dumps(
        {"verdict": "FLAWED", "flaws": ["It returns [7, 7].", "It never ran verify.py."]})
    review = critic.review([{"role": "assistant", "content": "cat pipeline.py"}], result())
    assert "adding 9" not in prompts[0]
    assert review["flaws"] == ["It never ran verify.py."]
    assert review["rejected_flaws"] == ["It returns [7, 7]."]
