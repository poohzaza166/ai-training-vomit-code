# Run notes: the self-critique GRPO loop

## STOPPED: the machine has no working GPU (2026-10-07 22:38)

Nothing is training. `torch.cuda.device_count()` returns 0 for every process. Recovering it needs
root, which this session does not have:

```sh
sudo nvidia-smi --gpu-reset -i 1   # or reboot if the handle is gone and reset refuses
nvidia-smi                         # both cards should list again
```

Then restart on GPU 0 only, with the prompt budget that fits it (see the sizing note below):

```sh
cd "computer use training"
CUDA_VISIBLE_DEVICES=0 nohup python -u train_grpo_gan.py \
  --adapter out/nemotron-think-grpo/adapter \
  --out out/nemotron-think-grpo2 \
  --sources opencode claude hf:arcee-glaive hf:openthoughts-agent hf:nvidia/When2Call hf:dapo-math \
  --rounds 4 --critic-rows 64 --max-prompt-length 1152 \
  --dump out/logs/completions-grpo2.jsonl > out/logs/grpo2.log 2>&1 &
```

If it OOMs again in `_prefill`, the next knob is `--num-generations 2` (and `--grad-accum` stays 4, so
the group constraint still holds). That costs group-size quality, which is why it is second.

### What happened to the hardware

Two separate faults, in order:

1. **GPU 0 (RTX 5060, 8 GB) OOMed** at step 6 of 368, allocating 344 MiB during rollout prefill. Not
   the backward pass - `_generate_and_score_completions` -> `_generate` -> `_prefill`. 409 MiB were
   free; the desktop and two other processes hold ~384 MiB of that card.
2. **GPU 1 (RTX 3060, 12 GB) fell off the bus** 12 steps into the relaunch:
   `torch.AcceleratorError: CUDA error: unspecified launch failure` inside the prebuilt
   `causal-conv1d` kernel (`causal_conv1d_update`, the decode path), with
   `NVRM: rpcRmApiFree_GSP ... status=0x0000000f` in the kernel log. `nvidia-smi` then reported
   `Unable to determine the device handle for GPU1: 0000:87:00.0: Unknown Error`, **and GPU 0 stopped
   working too.** The card took the driver down with it.

   Its own smoke run had passed cleanly beforehand, so a smoke does not clear this card - the fault
   only appears under sustained load. Treat GPU 1 as unreliable for unattended work.

### Sizing GPU 0 for the rollout

Shortening the prompt budget costs no rows, because `--limit 800` is the binding constraint, not the
budget. Measured on the live source mix:

| `--max-prompt-length` | rows kept | p50 | p95 | max |
|---|---|---|---|---|
| 1536 (OOMed) | 800 | 628 | 1540 | 1739 |
| 1280 | 800 | 488 | 1354 | 1458 |
| 1152 | 800 | 459 | 1194 | 1360 |
| 1024 | 800 | 438 | 1089 | 1257 |

Picked **1152**: a 22% cut in the longest prompt against 1536, while keeping the generation group at 4.
The cost is a shorter-context training distribution (p50 459 vs 628), which matters for an agent whose
real runs carry long histories - so this is the smallest cut that looks likely to fit, not the safest
one. 1024 is the fallback if 1152 still OOMs.

Decisions taken while setting up the 2026-10-07 resume run, and the reasoning behind each. Written
because the choices are not all recoverable from the code alone.

## What changed in `train_grpo_gan.py`

**The critic left the reward path.** It was one API call per completion: 800 prompts x 4 generations
is 3200 calls per round. No cloud tier sustains that, and `Critic.one` returns a neutral `1.0` when a
call fails - so a rate-limited round looks healthy and teaches nothing. Now `reward_test` keeps every
sample (it already parses them, so this is free) and `Harvest.critique` runs once at the end of the
round over `--critic-rows` rows, critiquing the worst sample of each. 128 calls per round at the
defaults, counting the held-out test. `--critic-reward` restores the old behaviour.

Cost of the change: the GRPO reward no longer contains a critic term, so within a round the critic
shapes nothing. It only produces `data/gan_pairs.jsonl` and next round's pasted hints. That is where
its signal was worth most anyway - a preference pair is a stronger teacher than a 3-level reward.

**`probe_agent.py` is now the gate.** `run_test` measures exact match on one recorded step, which says
nothing about a run left alone. `run_probe` runs probe_agent's multi-turn episodes after each round and
reports the per-turn pass rate plus `p**turns` for `--probe-horizon`.

**`--sources` matches from the left.** Was `source.split(":")[0]`, so the 2998 `hf:mlabonne/orpo-dpo-mix-40k`
general-chat rows could not be excluded without dropping every other `hf:` row. `hf` still selects all
of them; `hf:dapo` now selects only the verifiable maths.

**`--probe-protocol` defaults to the live config.** `.env` has `TOOL_PROTOCOL=json`. A `native` default
would have gated on a prompt the harness never sends.

## Assumptions in the resume command

- **"Last checkpoint" = `out/nemotron-think-grpo/adapter`** (2026-10-06 00:41), the newest complete one.
  The other candidate, `out/nemotron-nano-4b-grpo-gan/round0/adapter`, is from a run that died in
  round 1 (`round1/` exists and is empty) and is a different lineage - that one came off the ORPO
  adapter, this one off `out/nemotron-think` (SFT-think). Picked the newer, complete lineage.
- **Writing to a new `--out`, not back into `nemotron-think-grpo`.** Resuming in place would overwrite
  `round0/adapter` and `critic_memory.json` from the previous run, which are the only record of it.
- **Dropped `hf:mlabonne/orpo-dpo-mix-40k`** from the source mix (2998 rows, 15% of the pool). General
  chat preference data, nothing agentic; the objective here is tool-driven autonomy.
- **Kept `hf:dapo-math` (1000 rows)** despite being off-domain. They are the only rows carrying a
  ground-truth `answer`, so they are the one reward term in the mix that needs no judge. The previous
  run's `critic_memory.json` is all base-8 subtraction, which shows they can dominate a round's
  critique budget - worth watching, not worth dropping.
- **Moved to GPU 1 (RTX 3060, 12 GB) with `--no-offload`, after the 8 GB card OOMed.** The first
  attempt ran on GPU 0 (RTX 5060, 8 GB) because that path was the measured one. It died at step 6 of
  368 - `torch.OutOfMemoryError` allocating 344 MiB during *rollout prefill*
  (`_generate_and_score_completions` -> `_generate` -> `_prefill`), not during the backward pass. The
  24-row smoke never reached a prompt near the 1536-token cap; the full set does, and four of them
  prefilled at once do not fit. Only 409 MiB were free, and the desktop plus two other processes hold
  ~384 MiB of that card.

  The 3060 (compute 8.6, bf16 fine) was verified with its own smoke before relaunching: prebuilt mamba
  kernels resolve for sm86, and steady-state step time is ~11 s against the 5060's 10-26 s - faster,
  because `--no-offload` drops the CPU round-trip for the input embeddings. Peak is 5990 MiB of 12288,
  so ~6.3 GB of headroom where the 5060 had 0.4 GB.

  Standing option not taken: the fallback the docstring recommends (`--num-generations 2`) would also
  have fit on the 8 GB card, but it halves the GRPO group size, and the group mean *is* the baseline -
  a smaller group means a noisier advantage estimate. Spending the spare card was the cheaper trade.
- **Teacher critic, not `self`.** Verified `localhost:8000` answers to `sk-gemini-gemini-3.1-flash-lite`
  before launching, because a dead critic is silent.

## The eval was lying (fixed before training)

The first gate reading was `per-turn 0.632`, and 5 of its 7 bad turns were "expected run_command, got
no tool call". The transcripts showed why: turn 1 came back as a correct json envelope
(`{"tool": "run_command", ...}`), and every turn after it came back in native syntax
(`<tool_call><function=run_command>`).

`probe_agent.run_scenario` was appending the *parsed* assistant message back into the history, so the
chat template re-rendered the turn as `<tool_call><function=...>` - native syntax, inside a conversation
running the json protocol. The model copied the format it was shown. `src/harness/agent.py:148` already
gets this right (`tool_calls=assistant_calls if not json_mode else None`, keeping the raw envelope as
content - its comment calls it "an in-context format example"); only the probe was wrong.

Measured, same adapter, same 7 episodes:

| | mechanical score |
|---|---|
| json, before the fix | 0.68 |
| native | 0.86 |
| json, after the fix | 0.86 |

So the apparent protocol gap was entirely the eval's own bug. Nothing about the weights or the
`TOOL_PROTOCOL=json` setting needed changing, and the pre-fix number should not be compared against
anything.

**Assumption worth checking:** the fix mirrors `agent.py` by storing the raw envelope as assistant
content with the `<think>` block stripped. `agent.py` stores `raw_text` from the backend, where
reasoning arrives in a separate field; the probe decodes one string, so stripping is the closest
equivalent. If the live backend turns out to feed reasoning back into the history, the probe is
slightly optimistic there.

## The one real deficit, as of the 0.86 baseline

All three remaining failures are the same thing: **it does not stop.** It has the answer and keeps
calling tools.

- `explore` - has `ModuleNotFoundError: No module named 'requests'`, goes looking for a requirements file
- `recover` - `build.sh: 12 files compiled, 0 errors`, runs it again
- `loop_guard` - `12 passed`, decides to also verify the server starts

That is what `hf:nvidia/When2Call` (3000 rows, in the mix) teaches, and what `test_score` already
rewards: a row whose recorded step made no call scores 1.0 for stopping and 0.0 for calling anything.

Caution from the earlier stop-pairs attempt: training "stop sooner" directly taught fabrication
instead - the model stopped and invented a result. Watch `no_false_claim` on the `honesty` episode
(currently clean) as the stop behaviour moves; if honesty degrades while stopping improves, that trade
is the failure, not progress.

## What the gate can and cannot tell you

The target is hands-off overnight work: research into a knowledge base, a project from a spec. That is
200-1000 turns, and a run survives only if every turn does, so the odds are `p**turns`:

| per-turn | 50 turns | 200 turns |
|---|---|---|
| 0.95  | 7.7%  | 0.0%  |
| 0.99  | 60.5% | 13.4% |
| 0.995 | 77.8% | 36.7% |

Coin-flip odds at 200 turns need p = 0.9965. Two limits on believing any number this harness prints:

1. **The gate is 19 turns across 7 episodes** (the `gui` episode skips - no `screenshot`/`click` in the
   live tool set). One bad turn moves measured p by ~0.05. It cannot distinguish 0.95 from 0.995, which
   is the entire range that decides the question.
2. **Tool output is scripted.** Real runs produce output nobody wrote in advance, and that is where
   grounding fails first.

So a PASS here means "did not fail the 19 turns we wrote", not "ready to run unattended". Closing that
gap needs turns, from real execution in a throwaway directory, and most of the remaining reliability
is harness work - step budget, loop detector, forced verification before `task_done`, resumable state
so one bad turn does not cost the whole night - not weights.

## Datasets

Checked the ODSC list of 15 against what is already loaded. `hf:arcee-glaive` (4000) and
`hf:openthoughts-agent` (3000) already cover the multi-turn tool-call training data it recommends, and
`hf:nvidia/When2Call` (3000) covers knowing when *not* to call a tool, which the list does not mention
and which two of the probe's checks test directly. Worth adding, none wired yet:

- `gaia-benchmark/GAIA` - eval. Multi-step tool research with exact answers: the closest public proxy
  for the overnight-research objective, and verifiable, so it costs no judge calls.
- `gorilla-llm/Berkeley-Function-Calling-Leaderboard` - eval. `test_score` is already a homegrown BFCL;
  this gives comparable numbers for free.
- SWE-bench Verified - direction, not a gate. The only one that hard-passes "project from spec"; a 4B
  scores near zero, so it marks the goal rather than measuring progress toward it.

Skipped the web/GUI/embodied seven (WebArena, WebChoreArena, MiniWoB++, WebShop, ALFRED, OmniACT) -
the GUI tools are not in the live harness. ToolBench, ToolAlpaca and API-Bank are REST-API dialogues,
the wrong action space for a bash-and-files agent.
