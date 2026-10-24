# Continuing Nemotron CU and adding research tasks

Repository root: `computer use training/`.

## What is implemented

The original export script names the Ollama model `nemotron-3-nano-cu:4b`.
Its default source is `out/nemotron-nano-4b-orpo`: the original BF16 base plus
the ORPO adapter, merged and converted to Q8 GGUF. Both the adapter and GGUF
are present on this mirror. Ollama itself is not installed on this mirror,
so the model registered on another Ollama server has not been independently verified.

Continue training from the original **base + adapter**, not from the GGUF and
not from the merged ORPO model with the same adapter applied a second time.
The recent pilot already used this lineage. The pilot completed 32 SFT and
two GRPO updates but did not improve its small task-success evaluation.

## One command to continue from the CU checkpoint

```bash
cd '/home/pooh/computer use training'
.venv-grpo/bin/python -m cutrain.training.continue_nemotron_cu --out out/nemotron-cu-terminal-v2
```

This starts a user systemd service with the successfully tested limits:
5 GiB MemoryHigh, 6 GiB MemoryMax, 512 MiB MemorySwapMax. It prints the service
name, log path and exact underlying training command. The service survives an
SSH disconnect. The launcher uses the cached base snapshot when available,
the original ORPO adapter, and both `data/long_agent_sft_memory_v2.jsonl` and
`data/long_agent_stopping_sft.jsonl`. Training-time anti-repetition scoring is now
enabled by default; see [ANTI_LOOP_TRAINING.md](ANTI_LOOP_TRAINING.md).

Defaults use the small pilot budget plus self-critique: 32 SFT updates, two GRPO groups
of two trajectories, six actions, 384 tokens per action, 1K/2K initial
contexts and a 6K total context cap. This is terminal training, not research
training. The corrected v2 contract explicitly defines every operation.
These defaults are a smoke test, not a recommended full training budget.

To inspect the command without loading a model or starting training:

```bash
.venv-grpo/bin/python -m cutrain.training.continue_nemotron_cu --out out/nemotron-cu-terminal-v2 --dry-run
```

To continue from the experimental pilot instead of the original CU weights:

```bash
.venv-grpo/bin/python -m cutrain.training.continue_nemotron_cu \
  --source pilot --out out/nemotron-cu-pilot-continuation
```

`--steps`, `--sft-warmup-steps` and `--group-size` change the budget. Outputs
must be new directories. This restores adapter weights; it does not resume
the old optimizer, step counter or RNG state. For custom task depth or
context settings, use the printed `train_long_grpo.py` command directly.

## Integrated critique, verification and retry loop

`continue_nemotron_cu.py` now defaults to `--critic self`. Pass `--critic none`
for the earlier verification-only behavior. Direct `train_long_grpo.py` keeps
its default `--critic none` for compatibility; explicitly add `--critic self`.

This adapts the original `SelfCritic` backend to complete trajectories. Past
flaws are not fed back into later reviews: in the first full run the critic
copied them verbatim into reviews of unrelated tasks. It is an alternating policy/critique loop, not a classical
GAN with a separately gradient-trained discriminator. The self-critic uses the
same model sequentially; there is no second model resident in GPU memory.

1. Sample a group of independent terminal trajectories on one task.
2. Grade each with executable tests and compute the GRPO update.
3. Review the worst failed trajectory against the original request, tool outputs
   and independent verification summary. Preserve the full evidence; if the
   review exceeds the context budget, log a review error instead of truncating it.
4. Retry that same task in the next training group, in fresh workspaces, with identical
   conditioning for every member. Compare rewards within that group only.
5. `--retry-feedback` sets what the retry sees (default `none`: the plain task again).
   `facts` appends environment-measured facts to the task message, stated as what is
   correct (first wrong contract stage with its correct op/value, verify.py's required
   outputs, import errors, false success claims, invalid replies, turn exhaustion);
   failed code and wrong outputs are never shown because retries copied them.
   `facts+critic` also appends critic flaws that pass grounding: every list, number and
   backticked name cited must appear in the transcript or verifier facts (audit-log
   dump excluded), and no flaw may defer to the user.

   Measured on the 8 failed tasks of `out/full-cu-20261015T002706Z` with its final
   adapter, 4 rollouts each, identical sampling seeds (`out/feedback-ab*`):

   | Retry sees | Successes |
   |---|---|
   | plain task (`none`) | 30/32 |
   | facts inside the task message (`facts`) | 18/32 |
   | facts as a separate user turn | 10/32 |
   | facts + grounded critic, separate turn | 9/32 |
   | failed code + facts, separate turn | 7/32 |
   | original critic feedback, separate turn | 4/32 |

   The policy was never trained on feedback-conditioned prompts, so any feedback text
   is out of distribution for it. Critic reviews still run and are logged.
6. Save critique memory, reviews, verified SFT trajectories and within-group
   success/failure preference archives. A retry does not schedule another retry.

`--steps` counts all groups, including retries. Therefore the default two-group
run can mean one ordinary group and one retry of the same task at the same
context length; it is not two independent task seeds. Before/after evaluation
receives no critique hints. Critic generation has a separate 192-token budget.

Critic scores do not alter task rewards. Incorrect reasoning is trained through
its observable effects and recovery on retried tasks, not through rewarding
long thoughts or an evaluator's stylistic preferences. Critic SOUND cannot override
failed executable checks. Invalid JSON/verdicts are logged as review errors.
Self-critique is not independent verification; the isolated executable grader is.

Artifacts under the new output directory:

- `trajectory_critiques.jsonl`: verdict/status, verifier facts, retry feedback and
  `rejected_flaws` (critic flaws that failed grounding).
- `critic_memory.json`: most frequent grounded critic flaws, for inspection only;
  it is not read back into reviews.
- `verified_rollouts_sft.jsonl`: only independently successful policy trajectories;
  assistant labels exclude all seeded context and tool output.
- `trajectory_pairs.jsonl`: same-condition success/failure trajectory archives.
  These include tool observations and cannot be fed directly into the original
  single-response ORPO loader. A multi-turn preference converter/loss is still needed.

The last two files are created only when qualifying trajectories exist. No
critic-approved failure or oracle fallback is silently labelled as policy success.
The original critic's cloud-server mode is not enabled in this integration.

## How research training fits: proposed extension

```text
Original BF16 base + CU ORPO adapter
    -> SFT: terminal trajectories + research trajectories + tool-call rehearsal
    -> GRPO: alternate terminal groups and research groups
    -> separate terminal / research / retention evaluations
    -> selected adapter -> BF16 merge -> GGUF -> Ollama
```

The new environment would expose search, open-page, calculation and final-answer
actions in the actual harness protocol. Frozen document snapshots make research
rollouts repeatable; live web evaluation follows separately. Do not give the
existing terminal sandbox unrestricted networking to implement this.

Refactor the hard-coded `Episode` construction behind an environment factory.
Both environments should provide initial messages, action execution, completion
state, final reward and a saved transcript. The policy rollout, assistant-only
log probabilities, sequential GPU execution and adapter-saving machinery can
remain shared. Tool-return and final-answer handling must be tested against
the actual harness schema, rather than assuming the current terminal-only parser fits.

Every GRPO group must use the same task, environment and source snapshot for
all its members. Alternate whole groups by task type; do not compare a coding
reward with a research reward inside one group. Choose and monitor reward scales
per environment so one task type does not dominate updates merely by its scale.

Terminal grading remains executable correctness and verification. Research
grading checks extracted facts, benchmark compatibility, arithmetic, citation
support and answer coverage. Fabricated numbers/citations receive substantial
penalties; correct uncertainty is rewarded only when the evidence is insufficient.
Use deterministic checks for structured evidence and arithmetic, with reviewed
rubrics for semantic support. An LLM judge alone is insufficient.

For SFT, retain the existing messages plus explicit `assistant_loss_indices`
format; add task-family and source provenance metadata. Curate successful and
failure-recovery traces. Keep old function-calling examples as rehearsal.
The current SFT loader accepts multiple JSONL files, but simple concatenation
is not a balanced sampler: controlled mixtures need explicit sampling support.

Begin with extraction and two-source comparisons, then add incompatible
benchmarks, missing conditions, conflicting sources and stale evidence.
Increase real tool decisions separately from document/context length. Preserve
an early constraint such as single-user decoding at a specified context length.
Do not train the BC250/3090 answer as a fixed fact; train how to establish it.

Start with the measured 2K/4K context range. The 4K forward/backward profile fits
the RTX 3060, but longer research episodes still need generation and training
memory measurements. Group size two is a memory-conscious starting point;
larger groups require throughput measurement. Advance difficulty once the policy
sometimes succeeds, so GRPO receives evidence of successful behavior.

Split source documents and product/task families before producing trajectories.
Evaluate terminal success, research evidence accuracy, citation support,
comparability, tool-call validity and live-harness compatibility separately.
The research environment, balanced mixture sampler and research rewards are
**not implemented by the new launcher**. Research also needs environment-specific
critic instructions and feedback construction: the current critic adapter expects
terminal verifier fields and explicitly recommends running `python3 verify.py`.

## Pausing training (e.g. to play a game on the same machine)

```bash
bash scripts/train_ctl.sh pause    # stops every running training service: GPU and RAM are free within seconds
bash scripts/train_ctl.sh resume   # starts the paused runs again
bash scripts/train_ctl.sh status   # running / paused / stopped (a run a reboot or crash ended)
```

Works for runs started with `run_mix_grpo.sh` or `run_project_training.sh`; the others
(`run_until_pass.sh`, `run_full_training.sh`) are not converted. `train_long_grpo.py` records a
resume point (`state-*` in its `--out`: adapter, optimizer, RNG, loop state) after every GRPO step
and every `--checkpoint-minutes` (default 10) of SFT, so a pause loses at most the step in progress.
On resume the run scripts skip finished stages, and an interrupted evaluation runs again. Rows of a
repeated step appear twice in `episodes.jsonl`/`metrics.jsonl`; the higher `attempt` is the one that
counted. Raise the service's RAM cap with `MEMORY_HIGH=... MEMORY_MAX=... bash scripts/run_mix_grpo.sh`
(defaults 5500M / 6500M), e.g. for `--activation-offload` on a desktop.

## Exporting a selected trained adapter back to Ollama

The existing scripts support a run directory containing `adapter/`:

```bash
# On an export machine with enough host RAM, the same repo/dependencies,
# and this training output copied into it:
.venv-grpo/bin/python -m cutrain.training.train_orpo \
  --model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --out out/nemotron-cu-terminal-v2 --merge-only
.venv-grpo/bin/python -m cutrain.serving.to_ollama \
  --out out/nemotron-cu-terminal-v2 --name nemotron-cu-terminal-v2:4b
```

The merge loads the unquantized BF16 base on CPU (about 8 GB of weights plus
overhead); the mirror's 7.2 GiB host RAM is not adequate for a comfortable merge.
Use a larger-memory export machine. The export script requires llama.cpp and
Ollama; it converts to Q8 GGUF and creates the specified new model name. Do not
replace the original CU deployment before evaluating the candidate. These
export commands were inspected, not executed during this change.

The repository's Nemotron renderer/parser and raw-config conversion workaround
are preserved. After export, verify both chat and actual tool execution in your
harness: adapter training success alone does not establish serving compatibility.

References: [PEFT checkpoint/merge behavior](https://huggingface.co/docs/peft/developer_guides/checkpoint),
[Ollama GGUF import](https://docs.ollama.com/import).

## Validation of this integration

25 unit/regression tests passed. The four-turn GPU smoke run verified all four
trajectories and harvested 16 assistant actions; equal rewards correctly caused
zero GRPO updates. A separate three-turn test produced a valid, accurate critic
flaw report, one finite GRPO update, and a retry group with identical feedback
for both members. The retries failed and generated no positive SFT examples.
This validates execution and verification gates, not improved reasoning ability.
See `GRPO_PLAN_AND_REPORT.md` for exact artifact paths and measurements.

Training-time token-loop penalties, state-aware repeated-action penalties, verified stopping SFT and optional observed-loop preference warm-up are documented in [ANTI_LOOP_TRAINING.md](ANTI_LOOP_TRAINING.md). These do not install serving-time guards in Ollama.
