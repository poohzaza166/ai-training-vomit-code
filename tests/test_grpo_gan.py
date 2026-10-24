"""Self-check for the GRPO self-critique loop: python "computer use training/test_grpo_gan.py" """
import json
import random

from cutrain.training.train_grpo_gan import (ASSISTANT, PASS, SCAFFOLDS, Harvest, ServerCritic, answer_score,
                                            describe_probe, format_score, grounding_score, inject, parse,
                                            round_dataset, scaffold_block, scaffold_score, summarise,
                                            test_score as score_action)

# completion -> unified message, in the format the chat template emits
text = ("<think>\nls first, then read it.\n</think>\nChecking.\n<tool_call>\n<function=run_command>\n"
        "<parameter=cmd>\nls -la\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>")
msg = parse(text)
assert msg["reasoning_content"] == "ls first, then read it." and msg["content"] == "Checking."
assert msg["tool_calls"] == [{"type": "function", "function": {"name": "run_command",
                                                               "arguments": {"cmd": "ls -la", "timeout": 30}}}]
assert parse("<think></think>all done")["content"] == "all done"
assert "tool_calls" not in parse("all done") and "reasoning_content" not in parse("<think>\n\n</think>x")

ref = msg
assert score_action(ref, ref) >= PASS                                          # exact
assert score_action(parse(text.replace("ls -la", "ls")), ref) == 0.75          # right tool, one arg wrong
assert score_action(parse("<think>x</think>done"), ref) == 0.0                 # no call at all
assert score_action(parse(text.replace("run_command", "click")), ref) == 0.0   # wrong tool
stop = {"role": "assistant", "content": "done"}
assert score_action(stop, stop) >= PASS and score_action(ref, stop) == 0.0     # stopping is testable too

# verifiable rows are scored on the ground truth, not on matching the reference prose
assert answer_score("blah\nAnswer: 34", "34") == 1.0
assert answer_score("Answer: $1,234$", "1234") == 1.0           # separators and $ stripped
assert answer_score("so \\boxed{34} follows", "34") == 1.0        # boxed wins when present
assert answer_score("Answer: 7\nAnswer: 34", "34") == 1.0        # last answer counts
assert answer_score("Answer: 35", "34") == 0.0
assert answer_score("I think it is 34", "34") == 0.0             # no answer line at all
ans = {"role": "assistant", "content": "Answer: 34"}
assert score_action(ans, {"role": "assistant", "content": "Answer: 34"}, "34", "Answer: 34") == 1.0
assert score_action(ans, {"role": "assistant", "content": "x"}, "99", "Answer: 34") == 0.0

# a scaffold is only worth having if adherence is measurable
polya_block, polya_sections, polya_shot = SCAFFOLDS["polya"]
assert polya_block and polya_sections and all(s in polya_block for s in polya_sections)
# the worked example has to demonstrate the very headings the reward looks for, or it teaches nothing
assert all(s in polya_shot for s in polya_sections)
assert all(all(s in SCAFFOLDS[k][2] for s in SCAFFOLDS[k][1]) for k in SCAFFOLDS if k != "none")
assert scaffold_score(f"<think>{polya_shot}</think>act", polya_sections) == 1.0  # the shot is a full-credit turn
assert polya_shot not in scaffold_block("polya", False)
assert scaffold_block("none", True) == ""
good = "<think>\nUNDERSTAND: a\nPLAN: b\nCARRY OUT: c\nLOOK BACK: d\n</think>done"
assert scaffold_score(good, polya_sections) == 1.0
assert abs(scaffold_score("<think>UNDERSTAND: a\nPLAN: b</think>x", polya_sections) - 0.65) < 1e-9
# an empty <think> scores nothing, reasoning without the headings still beats it: that spread is the gradient
assert scaffold_score("<think></think>done", polya_sections) == 0.0
assert scaffold_score("<think>\n\n</think>done", ()) == 0.0
assert scaffold_score("<think>let me look at the log first</think>x", polya_sections) == 0.3
assert scaffold_score("<think>let me look</think>x", ()) == 1.0        # no scaffold: thinking is the whole ask
assert scaffold_score("UNDERSTAND: a\nPLAN: b", polya_sections) == 0.0  # no reasoning at all scores nothing
# headings in the body still count, as long as the turn reasoned: that is where the model puts them
assert scaffold_score("<think>thinking</think>UNDERSTAND: a\nPLAN: b\nCARRY OUT: c\nLOOK BACK: d",
                      polya_sections) == 1.0
assert SCAFFOLDS["none"] == ("", (), "")
assert all(SCAFFOLDS[k][1] for k in SCAFFOLDS if k != "none")

# paths the model made up are the baseline's worst habit, so they have to cost it
prompt = "<|im_start|>user\nthe tests in /workspace/app fail<|im_end|>\n"
call = lambda cmd: {"role": "assistant", "tool_calls": [{"type": "function", "function": {  # noqa: E731
    "name": "run_command", "arguments": {"cmd": cmd}}}]}
assert grounding_score(call("ls -la /workspace/app"), prompt) == 1.0
assert grounding_score(call("ls /workspace/app/sub/dir && cat /etc/hosts"), prompt) == 1.0  # sandbox tree
assert grounding_score(call("ls /mnt/driveD/code/unity"), prompt) == 0.5                    # memorised root
assert grounding_score(call("ls /mnt/a/b && ls /srv/c/d"), prompt) == 0.0
assert grounding_score(call("ls /mnt/driveD/code/unity"), prompt + "/mnt/driveD/code/unity") == 1.0  # shown
assert grounding_score({"role": "assistant", "content": "no calls here"}, prompt) == 1.0

assert format_score(text, False) == 1.0
assert format_score(text, True) < 1.0                                          # truncated
assert format_score(text + "\nand now I will also...", False) < 1.0            # text after the call
assert format_score("<think>unclosed" + text, False) < 1.0

# critique injection keeps the prompt renderable and ends on the assistant turn
p = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\ngo<|im_end|>\n" + ASSISTANT
hinted = inject(p, ["clicked without a screenshot", "claimed success with no output"])
assert hinted.endswith(ASSISTANT) and hinted.startswith(p[: -len(ASSISTANT)])
assert "clicked without a screenshot" in hinted and hinted.count(ASSISTANT) == 1

# criticised rows train twice: bare, and with their own flaws pasted in
rows = [{"row_id": "a", "prompt": p}, {"row_id": "b", "prompt": p}]
ds = round_dataset(rows, {"a": ["bad coordinate"]}, True)
assert len(ds) == 3 and sum("bad coordinate" in r["prompt"] for r in ds) == 1
assert len(round_dataset(rows, {"a": ["bad coordinate"]}, False)) == 2

# harvest: flawed sample becomes the rejected side, a clean passing sample the chosen one
row = {"ref": ref, "category": "execution", "tools": [], "prompt": "go" + ASSISTANT,
       "msgs": [{"role": "user", "content": "go"}]}   # as load_rows builds them
side = {"a": dict(row), "b": dict(row)}
h = Harvest()
bad = parse(text.replace("ls -la", "rm -rf /"))
h.add("a", 1.0, ref, [])                      # clean and passing -> on-policy chosen
h.add("a", 0.75, bad, ["deletes the workspace"])
h.add("b", 0.0, bad, ["deletes the workspace"])   # nothing clean -> fall back to the recorded action
h.add("c", 0.5, bad, ["orphan row, no side entry"])
pairs, flaws = h.pairs(side, 0)
got = {r["id"]: r for r in pairs}
assert set(got) == {"gan0:a", "gan0:b"}, got.keys()      # row c dropped: not in the training rows
assert got["gan0:a"]["chosen"] == [ref] and got["gan0:a"]["on_policy_chosen"]
assert got["gan0:b"]["chosen"] == [ref] and not got["gan0:b"]["on_policy_chosen"]
assert all(r["rejected"] == [bad] and r["rejected_origin"] == "gan:self_critique" for r in pairs)
assert flaws == {"a": ["deletes the workspace"], "b": ["deletes the workspace"]}
# pairs are prepare_data.py rows: train_orpo.py reads them with no conversion
assert set(got["gan0:a"]) >= {"id", "source", "category", "tools", "prompt", "chosen", "rejected", "rejected_origin"}
assert json.loads(json.dumps(pairs))  # jsonl-serialisable

# a sample the critic approves makes no pair, and no hint for next round
h2 = Harvest()
h2.add("a", 1.0, ref, [])
assert h2.pairs(side, 0) == ([], {})

# the round-end critique: one call per row, capped by --critic-rows, worst sample first
class Counting:
    def __init__(self):
        self.calls, self.scored = 0, []

    def one(self, context, msg):
        self.calls += 1
        self.scored.append(msg)
        return 0.66, ["invented a path"]


h3 = Harvest()
for rid in ("a", "b"):
    h3.add(rid, 1.0, ref, [])
    h3.add(rid, 0.0, bad, [])
h3.add("c", 0.0, bad, [])                       # not a training row: never worth a call
counted = Counting()
assert h3.critique(counted, side, 1, random.Random(0)) == 1 and counted.calls == 1
assert counted.scored == [bad]                  # the worst sample, not an arbitrary one
pairs, flaws = h3.pairs(side, 0)
assert len(pairs) == 1 and pairs[0]["rejected"] == [bad] and pairs[0]["on_policy_chosen"]
assert list(flaws.values()) == [["invented a path"]]

# the probe gate reports what a per-turn rate implies over a long run, which is the whole point
line = describe_probe({"per_turn": 0.9, "turns": 10, "clean": 1, "episodes": 2}, [5])
assert "per-turn 0.900" in line and "59.05%" in line


# the critic: verdict/flaw parsing and the memory that sharpens it, against a stub server
class Stub:
    def __init__(self, reply):
        self.reply, self.seen = reply, []
        self.chat = type("c", (), {"completions": self})()

    def create(self, **kw):
        self.seen.append(kw["messages"][0]["content"])
        body = type("m", (), {"content": self.reply})()
        return type("r", (), {"choices": [type("c", (), {"message": body})()]})()


def critic(reply, tmp="/tmp/claude-grpo-gan-critic.json"):
    import pathlib
    path = pathlib.Path(tmp)
    path.unlink(missing_ok=True)
    c = ServerCritic.__new__(ServerCritic)
    Critic_init = type(c).__mro__[1].__init__
    Critic_init(c, path, keep=2)
    c.client, c.model, c.workers, c.reasoning = Stub(reply), "stub", 1, "low"
    return c

c = critic('noise {"verdict": "FLAWED", "flaws": ["no screenshot first", "wrong window", "x", "y"]} tail')
score, flaws = c.one("ctx", bad)
assert flaws == ["no screenshot first", "wrong window", "x"], flaws  # capped at 3
assert score == 0.0, score  # three flaws exhaust the critic reward

c.one("ctx", bad)
assert len(c.client.seen) == 1, "identical sample must come from the cache"
c.remember()
assert c.memory == ["no screenshot first", "wrong window"] and json.loads(c.memory_path.read_text()) == c.memory
assert "Flaws this agent repeats" in c.system() and "no screenshot first" in c.system()

assert abs(critic('{"verdict": "FLAWED", "flaws": ["missed the error line"]}').one("ctx", bad)[0] - 0.66) < 1e-9
assert critic('{"verdict": "SOUND", "flaws": ["nitpick"]}').one("ctx", ref) == (1.0, [])  # verdict wins
assert critic('{"verdict": "FLAWED"}').one("ctx", ref)[1], "FLAWED with no flaw list still counts as flawed"
assert critic("not json at all").one("ctx", ref) == (1.0, [])  # unparseable -> neutral, never punishes


class Dead(Stub):
    def create(self, **kw):
        raise RuntimeError("connection refused")


d = critic("x")
d.client = Dead("x")
assert d.one("ctx", ref) == (1.0, []) and d.errors == 1  # a dead critic must not poison the reward
assert "RuntimeError" in d.report_errors() and "neutral 1.0" in d.report_errors()  # and it must say so
assert critic("x").report_errors() == ""

assert summarise({"role": "assistant", "content": ""}) == "(empty answer)"
assert "run_command" in summarise(ref)
print("ok")
