# Training the policy to stop and avoid repetition

Implemented in `train_long_grpo.py`; enabled by the CU continuation launcher.
These changes affect learning. They do not add an Ollama inference wrapper or
claim that a trained model can never loop.

## Objective and safeguards against reward gaming

The existing independently verified task reward is retained. Subtract one capped
penalty per episode if degenerate token cycles are detected:

```text
reward = task_reward - 0.5 * any_detected_token_loop
```

`--token-loop-penalty` controls the coefficient (0..1; default 0.5).
`--loop-min-tokens` defaults to 64 (minimum 32). The detector searches generated
tokens only for at least four adjacent identical blocks of 1..32 tokens, spanning
at least the configured threshold. It checks the whole response, not only the tail.

Complete JSON objects/arrays, quoted JSON literals, native tool-call parameter values, fenced code, punctuation-only
spans and ambiguous Unicode boundaries are conservatively exempt. This prevents
many code/JSON false positives, but deliberately misses some degenerate output,
including repetition inside those protected regions. It is a transparent heuristic,
not a semantic proof that every detected or missed pattern is useless.

There is no reward for brevity and no token-loop penalty just for reaching the
generation cap. An interrupted response without EOS is recorded as an invalid
unfinished turn and is not executed, even if it happens to parse as valid JSON.
The existing format penalty is separate from the new token-loop penalty.
Finite token and turn limits still bound training rollouts.

Sampling uses no repetition penalty or n-gram suppression. GRPO log probabilities
therefore continue to match the sampled policy (with the configured temperature).
The new signal changes gradients, rather than merely hiding repetition at decoding.

Repeated terminal commands are now penalized only when the same command is seen
with identical before/after task artifacts and identical output. Re-running tests
after an edit is not penalized. A cycle returning to a previously seen transition
can be detected. Unrelated temporary files cannot reset the task fingerprint.
This is an observable-state heuristic; it cannot detect every semantic action loop.

## Verified SFT for stopping, recovery and honest failure

`export_stopping_sft.py` produced `data/long_agent_stopping_sft.jsonl`:

- 24 clean verified repair episodes.
- 16 episodes with an actual failed test before repair.
- 16 examples whose only labelled action is `task_done` after successful verification.
- 8 honest blocked endings after a real terminal timeout closes the session.

Only correct assistant continuations receive labels; failed/contextual actions and
tool observations do not. The SFT trainer appends the actual end-of-turn EOS token.
The validator checks that EOS explicitly. Blocked examples are accepted only with
the unavailable-shell evidence and a non-success, non-false-claim result.

The launcher mixes this dataset with the earlier 64 verified repair episodes.
Thus the combined pool has 128 episodes, with blocked cases comprising 6.25%.
Rows are shuffled uniformly; a short run does not necessarily cover every mode.

## Observed-loop preference training

During GRPO, `loop_preferences.jsonl` is written only when an actual looping
response has an alternative from an independently successful, loop-free trajectory
at the exact same conversation prefix. A critic saying SOUND is insufficient.
No synthetic loop is silently presented as a policy failure, and no unverified
short answer is labelled as the chosen solution.

These are single-assistant-turn preference rows, unlike the older multi-turn
`trajectory_pairs.jsonl` archives. The new optional ORPO warm-up uses the same
completion-only scoring and fixed model lineage. It scores the two alternatives
without graphs, then recomputes one at a time using exact chain-rule coefficients:

```text
c = mean log probability of chosen assistant tokens (including EOS)
r = mean log probability of rejected assistant tokens
loss = -c - beta * log_sigmoid(log_odds(exp(c)) - log_odds(exp(r)))
```

Only assistant tokens receive loss. No reference model or simultaneous pair of
activation graphs is loaded. A truncated negative receives no invented EOS label.
The loader rechecks the positive's verification, detects the negative's repetition,
and rejects examples that exceed context instead of truncating original evidence.

If all group members loop equally, GRPO still has zero relative advantage. Such
groups are logged and skipped. Verified SFT supplies the initial contrast; optional
preference warm-up can use collected pairs when they exist. If there are no verified
alternatives, no preference file is fabricated. More verified demonstrations or
reviewed corrections are needed in that case.

## Commands

The existing launcher now includes both SFT datasets and the new penalty:

```bash
cd '/home/pooh/computer use training'
.venv-grpo/bin/python continue_nemotron_cu.py \
  --out out/nemotron-cu-anti-loop-v1 --steps 40
```

Use `--source pilot` to start from the pilot adapter instead of the original CU
adapter. Use `--dry-run` to inspect the complete command. To extend an existing
direct training command, include:

```text
--sft-data data/long_agent_sft_memory_v2.jsonl data/long_agent_stopping_sft.jsonl
--token-loop-penalty 0.5
--loop-min-tokens 64
```

When a prior run actually contains verified loop pairs, an optional preference
warm-up can be enabled with:

```text
--loop-preference-data out/PRIOR_RUN/loop_preferences.jsonl
--loop-preference-steps 16
--loop-preference-beta 0.1
```

These flags are supported by `train_long_grpo.py`; the launcher exposes the data
and step-count options with beta defaulting to 0.1. Preference learning uses
`--sft-lr`; use an explicitly chosen small learning rate for a controlled experiment.
An empty/missing file is an error, not a successful preference-training run.

## Evaluation and evidence

Before/post-SFT/post-preference/after evaluations use the same held-out task seeds
and decoding settings. They report task success, token-loop episodes, repeated-action
episodes, truncation, abstention and premature stopping separately. A lower loop rate
with worse task completion is not automatically an improvement. Four-mode stopping
data and these metrics do not establish general research-agent or Ollama behavior.

Group metrics retain both the original task rewards and the loop penalties.
Episode logs include loop spans' message indices and original looping responses,
so collected preferences can be traced to the observed failure.

Validation results and GPU smoke-test status are recorded in `GRPO_PLAN_AND_REPORT.md`.
