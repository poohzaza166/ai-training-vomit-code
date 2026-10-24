import json
import sys
from pathlib import Path

import pytest
import torch

from cutrain.training import train_long_grpo


class Killed(Exception):
    """Stands in for train_ctl.sh pause stopping the service mid-step."""


class FakeModel(torch.nn.Module):
    use_kernels = False

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(2))

    def get_base_model(self):
        return self

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "weights.pt")


class FakeTok:
    def save_pretrained(self, path):
        pass


class FakeCuda:
    # Only train_long_grpo sees a GPU; torch's own internals (the optimizer) keep the real answer.
    is_available = staticmethod(lambda: True)
    reset_peak_memory_stats = manual_seed_all = set_rng_state_all = staticmethod(lambda *a: None)
    max_memory_allocated = max_memory_reserved = staticmethod(lambda: 0)
    get_rng_state_all = staticmethod(lambda: [])


class TorchWithFakeCuda:
    cuda = FakeCuda

    def __getattr__(self, name):
        return getattr(torch, name)


@pytest.fixture
def harness(monkeypatch, tmp_path):
    log = {"loaded": [], "sft": [], "grpo": [], "adam_steps": [], "kill": None}
    monkeypatch.setattr(train_long_grpo, "torch", TorchWithFakeCuda())

    def load_model(args):
        model = FakeModel()
        log["loaded"].append(args.adapter)
        if args.adapter and (Path(args.adapter) / "weights.pt").exists():
            model.load_state_dict(torch.load(Path(args.adapter) / "weights.pt"))
        return model, FakeTok()

    def sft_warmup(model, tok, opt, args, start=0, save=None):
        for step in range(start, args.sft_warmup_steps):
            opt.zero_grad()
            model.w.sum().backward()
            opt.step()
            log["sft"].append(step)
            log["adam_steps"].append(int(opt.state_dict()["state"][0]["step"]))
            save(step + 1)
            if log["kill"] == ("sft", step):
                raise Killed
        save(args.sft_warmup_steps, force=True)

    def rollout(model, tok, args, seed, target, sample=True, capture_old=True, feedback=None):
        step, member = seed - args.seed, len([g for g in log["grpo"] if g[0] == seed - args.seed])
        log["grpo"].append((step, member))
        if log["kill"] == ("grpo", step, member):
            raise Killed
        result = {"reward": float(member), "task_reward": float(member), "token_loop_penalty": 0.0,
                  "token_loop_turns": 0, "repeats": 0, "truncated_turns": 0, "success": False,
                  "verified": False, "intact": True, "tests": 0}
        return [], result, [{"role": "user", "content": "task"}]

    evaluate = lambda model, tok, args, label: [{"phase": label, "success": True, "repeats": 0, "truncated_turns": 0}]
    for name, fn in (("load_model", load_model), ("sft_warmup", sft_warmup), ("rollout", rollout),
                     ("evaluate", evaluate)):
        monkeypatch.setattr(train_long_grpo, name, fn)
    out = tmp_path / "run"

    def run(*extra, kill=None):
        log["kill"] = kill
        monkeypatch.setattr(sys, "argv", ["train_long_grpo.py", "--out", str(out), "--resume", "--sft-warmup-steps", "5",
                                          "--steps", "4", "--group-size", "2", "--eval-episodes", "1",
                                          "--min-host-available-gib", "0", "--checkpoint-minutes", "0", *extra])
        train_long_grpo.main()
    return run, log, out


def test_pause_mid_sft_and_mid_grpo_resumes_where_it_stopped(harness):
    run, log, out = harness
    with pytest.raises(Killed):
        run(kill=("sft", 2))
    assert log["loaded"] == [str(train_long_grpo.REPO / "out/nemotron-nano-4b-orpo/adapter")]
    with pytest.raises(Killed):
        run(kill=("grpo", 2, 1))
    assert Path(log["loaded"][-1]).parent.name.startswith("state-")  # continued from the resume point
    run()
    # Every SFT step ran exactly once, and AdamW's moments carried over the pause.
    assert log["sft"] == [0, 1, 2, 3, 4] and log["adam_steps"] == [1, 2, 3, 4, 5]
    # GRPO step 2 was killed during its second rollout and ran again in full; nothing else repeated.
    assert [s for s, m in log["grpo"]] == [0, 0, 1, 1, 2, 2, 2, 2, 3, 3]
    summary = json.loads((out / "summary.json").read_text())
    assert [m["step"] for m in summary["training"]] == [0, 1, 2, 3]
    assert [m["attempt"] for m in summary["training"]] == [2, 2, 3, 3]
    assert summary["before"][0]["phase"] == "before" and summary["post_sft"][0]["phase"] == "post_sft"
    assert (out / "adapter" / "weights.pt").exists() and (out / "sft-adapter" / "weights.pt").exists()
    assert len(list(out.glob("state-*"))) == 2 and not (out / "state.tmp").exists()
    loaded = len(log["loaded"])
    run()  # a finished run is not loaded or trained again
    assert len(log["loaded"]) == loaded


def test_resume_refuses_changed_arguments(harness):
    run, log, out = harness
    with pytest.raises(Killed):
        run(kill=("sft", 0))
    with pytest.raises(SystemExit):
        run("--lr", "1e-5")
