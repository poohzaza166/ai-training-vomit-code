# Long-context terminal GRPO: plan and recovery report

Updated: 2026-10-15.
Project directory: `/home/pooh/computer use training`.

## Full mixed-data training script — 2026-10-15

`run_full_training.sh` now uses the existing `build_dataset.py` for source mixing
and deduplication, then `prepare_full_run.py` for native-template SFT rendering.
See [FULL_TRAINING_RUN.md](FULL_TRAINING_RUN.md) for source quotas, commands,
training stages and limitations. The trainer now accepts explicit rendered SFT
rows so reasoning and tool definitions survive without flattening the conversation.
Actual-data preparation retained 4,700 completions, including 600 DeepSeek,
300 GLM and all 440 verified terminal/stopping actions; 440 over-context source
examples were excluded without trimming. Adapter/terminal regression tests:
19 passed in 16.48 seconds (`out/logs/full-script-tests.log`).
The script performs a memory profile, one mixed-SFT epoch, 40 short-context GRPO
groups, then 40 longer-context groups. It uses the original CU adapter, sequential
QLoRA rollouts and self-critique with independently graded rewards. Static DAPO
examples participate only in SFT; no new maths RL environment or research harness
is implied. The full run has not been launched and no quality gain is claimed.

## Training-time anti-repetition implementation — 2026-10-15

See [ANTI_LOOP_TRAINING.md](ANTI_LOOP_TRAINING.md) for the objective, data,
preference-training path, limitations, and commands. New files: `anti_loop.py`,
`train_loop_preferences.py`, `export_stopping_sft.py`, `test_anti_loop.py`.

The task reward now receives a capped 0.5 episode penalty for detected token loops.
Complete JSON, quoted literals and fenced code are conservatively exempt; the
heuristic can miss loops within those regions. No decoding repetition suppression
is used to manufacture a lower measured loop rate. Token-budget exhaustion alone
is not labelled repetition. Unfinished responses are not executed as tool calls.
Repeated commands now require the same observable task-state transition and output;
verification after an edit is not a repeated-action penalty.

New SFT data: 64 verified stopping episodes, 184 labelled assistant examples,
501–2,101 tokens, all within 4K and all checked for the actual EOS token.
There are 56 successful episodes and eight verified unavailable-terminal endings.
Tests: 36 passed in 32.44 seconds, including protected code/JSON, capped rewards,
state-aware repetition, unfinished-call rejection, stopping-data verification,
preference provenance/conditioning, and exact ORPO gradient decomposition.
After adding a direct GRPO gradient-direction test, the focused suite passed
12 tests in 154.67 seconds while the GPU smoke run was active (37 distinct tests
across both runs). The new test verifies that the penalty reduces a looping
alternative's probability without sampling filters. A real Nemotron-tokenizer
check detected 297 repeated tokens in a 301-token repetitive phrase and exempted
repeated code inside a valid JSON call; this is synthetic detector validation.
Evidence: `out/long-grpo-recovery/pytest-anti-loop.log`, `stopping-export.log`,
`stopping-validation.json`, `anti-loop-launcher-check.log`,
`pytest-anti-loop-gradient.log`, and `anti-loop-tokenizer-check.json`.

Observed looping responses are paired only with verified alternatives at an
identical prefix. The optional completion-only ORPO warm-up is separate from GRPO;
its data is not fabricated when no qualifying pair exists. No real looping policy
pair has yet been demonstrated in these smoke runs, so no real-data preference
training or reduced deployment hallucination/repetition rate is claimed.

A GPU smoke run completed with exit status 0 as
`long-grpo-anti-loop-smoke-20261015.service`,
with log `out/logs/long-grpo-anti-loop-smoke.log`, output
`out/long-grpo-anti-loop-smoke`, four SFT steps, two GRPO groups, and one fixed
held-out task evaluated before SFT, after SFT and after GRPO at 1K context.
It continued the experimental pilot adapter with the critic disabled to isolate
this integration. Training rewards were [2.46, 2.46] then [2.46, 0.0]: one flat
group skipped, one finite GRPO update (gradient norm 0.08942), three of four
training trajectories successful. Peak allocated GPU memory was 4.628 GiB.
The same single held-out task passed in all three evaluation phases, with zero
token loops, action loops, truncations or premature stops. There were no observed
token loops in the training samples either, so this run validates execution but
does not demonstrate a reduction in loop rate. No loop preference pairs were
available and no preference warm-up was run. The adapter was saved under
`out/long-grpo-anti-loop-smoke/adapter`; it has not been deployed to Ollama.

## CU continuation command and integrated critique — 2026-10-15

See [PIPELINE_AND_COMMANDS.md](PIPELINE_AND_COMMANDS.md) for the checkpoint
lineage, runnable continuation command, research-training integration design,
and existing merge/GGUF/Ollama export commands.

`continue_nemotron_cu.py` selects the original ORPO adapter behind the repository's
`nemotron-3-nano-cu:4b` export, or the experimental pilot with `--source pilot`.
It launches a memory-limited user service and supports a non-mutating `--dry-run`.
It does not train from the GGUF, overwrite an existing run, or deploy a model.

`agent_critique.py` adapts the original SelfCritic to full terminal trajectories.
`train_long_grpo.py --critic self` now runs verified policy groups, reviews a failed
trajectory, and schedules a fresh feedback-conditioned retry group. All members
of each group receive the same task and feedback. Critic verdicts do not change
executable rewards. Critic failures are recorded as unknown/error, never SOUND.
Only independently verified successes can become SFT examples or chosen preference
trajectories. The exported preference archives need a multi-turn-aware consumer;
they are not directly compatible with the old single-step ORPO loader.

This preserves the original alternating critique idea, not a classical trainable
GAN discriminator. Self-critique does not constitute independent verification.
The executable grader remains independent; research-source verification is still
a proposed environment extension rather than implemented functionality.

Validation: 25 tests passed in 9.11 seconds across the original terminal suite
and the new critic tests (`out/long-grpo-recovery/pytest-critique.log`).
A real two-group GPU smoke test was started as
`long-grpo-critic-smoke-20261015.service`, writing
`out/logs/long-grpo-critic-smoke.log` and `out/long-grpo-critic-smoke/`.
It continues the experimental pilot, uses no SFT warm-up, four actions per rollout,
192 generation tokens, 1K starting context and a 4K cap. It completed successfully:
all four sampled trajectories passed independent verification and were harvested
as four SFT episodes with 16 labelled assistant actions. Peak allocated GPU memory
was 4.166 GiB. Both groups had identical rewards [2.46, 2.46], so GRPO correctly
performed zero optimizer updates. No failure occurred, so no critic review or
retry was needed. These are training-task samples, not a held-out benchmark.
This result used the clarified contract; it cannot be attributed to new GRPO updates.

A separate three-turn control-flow test ran as
`long-grpo-critic-retry-test-20261015.service`, with artifacts in
`out/long-grpo-critic-retry-test/` and log `out/logs/long-grpo-critic-retry-test.log`.
The reduced turn budget deliberately exercises incomplete-task feedback and retry;
it is not a recommended training setting or a model-quality comparison.
Its training loop completed and saved an adapter. The first group had rewards
[0.5, 0.0], yielding one finite GRPO update (gradient norm 0.04792). The live
self-critic returned valid FLAWED JSON and correctly identified addition where
the contract required the minimum-clamp operation. Both members of the next
group received identical feedback and task seed 42 in fresh workspaces.
Their rewards were [0.0, 0.0], so that group correctly skipped its update.
Peak allocation was 4.581 GiB. No verified successes or positive SFT examples
were harvested from this deliberately restricted run. Feedback did not produce
a successful repair: control-flow validation is not a quality-improvement claim.
The explicit four-turn success test and three-turn failure/retry test together
exercise the new integration; the 25 unit/regression tests cover malformed critic
responses, authoritative verification and preference-conditioning checks.

Launcher checks passed for both checkpoint lineages, non-mutating dry runs,
existing-output rejection and invalid-budget rejection.

## Latest attempt after driver repair — 2026-10-14

CUDA now works. `nvidia-smi` reports an RTX 3060 with 12,288 MiB physical
VRAM, driver 595.91.07, and CUDA support 13.2. PyTorch returned
`cuda_available=true`. The sandbox oracle passed again; evidence is in
`out/long-grpo-recovery/preflight-gpu.json` and `preflight-gpu.log`.

A forward/backward profile was launched with 1,024 / 2,048 / 4,096 input tokens
and 384 completion tokens, without CPU embedding or activation offload.
SSH temporarily timed out during startup, including a termination request; that
request did not stop the profiler. Connectivity recovered at 22:54 UTC and the
same process (PID 25192) was alive, with model loading complete and GPU kernel
files downloading. `nvidia-smi` showed 4,104 MiB used by Python. Zero utilization
during setup is not a CUDA failure. No cause of the SSH interruption is established.
The adapter subsequently loaded (17,054,208 trainable parameters), and GPU utilization
reached 100% during the profile. Linux also showed heavy I/O waits, swap reads, and
journal messages reporting memory pressure during startup. The storage device is a
rotating ST3500418AS disk. These observations explain CPU/storage stalls between GPU
bursts; they do not establish a driver failure. The forward/backward profile subsequently completed successfully (results below). The trainer now logs explicit loading,
model-ready and profiling phases for subsequent runs (syntax checked remotely).

On this 7.2 GiB RAM host, avoid CPU activation/embedding offload and concurrent GPU
jobs. A faster disk and more host RAM would address the observed startup bottleneck;
context limits still need to be selected using completed GPU measurements.

### Completed RTX 3060 profile and pilot

The actual completion budget was 384 tokens, with QLoRA and CUDA kernels,
no CPU activation/embedding offload. All three losses and gradient norms were finite.

| Prompt tokens | Peak allocated GiB | Peak reserved GiB | Forward/backward seconds |
| --- | --- | --- | --- |
| 1,024 | 4.746 | 4.838 | 261.24 (cold setup and I/O included) |
| 2,048 | 4.975 | 5.154 | 3.03 |
| 4,096 | 5.932 | 6.092 | 5.45 |

Evidence: `out/long-grpo-mirror-profile/profile.json` and
`out/logs/long-grpo-mirror-profile.log`. This is a gradient/memory check,
not an optimizer update or a demonstration of improved agent success.

A small pilot completed successfully as the user service `long-grpo-pilot-20261014.service`,
with `MemoryHigh=5G`, `MemoryMax=6G`, `MemorySwapMax=512M`. It loads the cached
base snapshot directly and continues the original ORPO adapter. Configuration:
32 verified-trajectory SFT updates, then two GRPO groups of two rollouts,
six turns maximum, 384 tokens per action, 1K/2K starting contexts, 6K context cap.
Greedy evaluations run before SFT, after SFT, and after GRPO, with one episode
per context. This is a smoke test, not a statistically meaningful benchmark.

Logs: `out/logs/long-grpo-mirror-pilot.log`; artifacts: `out/long-grpo-mirror-pilot/`.
Inspect status with `systemctl --user status long-grpo-pilot-20261014`.
The service exited successfully. The final checkpoint is
`out/long-grpo-mirror-pilot/adapter/adapter_model.safetensors` (68,235,624 bytes).
All 32 SFT updates completed; logged loss changed from 0.778 on the first example
to 0.040 on the last (different examples, not a held-out loss comparison).
Both GRPO groups produced finite nonzero gradients and optimizer updates, with no
flat groups. Peak allocation was 4.61 GiB and 5.17 GiB for those groups; runtime
was 41.3 and 60.3 seconds respectively, excluding loading/SFT/evaluation.

**No agent-success improvement was demonstrated.** Greedy task success was 0/2
before SFT, 0/2 after SFT, and 0/2 after GRPO. These are the same single held-out
task seed evaluated at two context lengths, not two independent tasks. The four
sampled training episodes also failed. Both groups had rewards [-1.1, -0.05], so
updates primarily distinguished false/invalid claims from less penalized failures.
They did not provide positive successful-task trajectories.

The model generated genuine inspect/edit/test sequences and attempted recovery,
but often used the wrong operation or incomplete intermediate output. This is a
training-pipeline smoke-test checkpoint, not a deployment recommendation.

### Follow-up from the pilot

Inspection exposed an environment ambiguity: `floor` was graded as `max(x, value)`
but its meaning was not stated in the contract. The model sometimes used integer
division. `agent_grpo_env.py` now includes explicit semantics for every operation.
Historical pilot results above used the old contract; they remain preserved.
New demonstrations were generated separately as
`data/long_agent_sft_memory_v2.jsonl`, with the old dataset retained for reproducibility
(`export_long_agent_sft.py --audit-lines 0 16 48`; the default 0 64 256 overflows 4K).
All 64 oracle episodes passed executable verification. Actual-tokenizer validation
passed for 256 assistant examples spanning 501–1,988 tokens, all within 4K.
The regression suite passed again: 16 tests in 80.12 seconds. Evidence is in
`out/long-grpo-recovery/pytest-contract-v2.log`, `sft-export-v2.log`, and
`sft-validation-v2.json`. No new training run on these corrected contracts has been launched.

Before increasing GRPO steps, use the corrected contracts, train on more verified
traces with failed-test recovery, and evaluate on multiple held-out seeds. Start
with one-stage tasks and progress to deeper pipelines once sampled groups include
both real successes and failures. Repeat baseline/post-SFT/post-GRPO comparisons
on the same corrected environment. Do not equate lower SFT loss or a successful
optimizer step with better long-horizon behavior.

## Mirror restoration status before driver repair

The multi-turn training implementation has been recreated on this mirror, building on
the existing Nemotron ORPO loader. **GPU training is blocked by the new host's missing
NVIDIA kernel module. No completed new GRPO-trained adapter is claimed.**

This mirror is native Ubuntu 26.04.1, kernel `7.0.0-38-generic`, with an RTX 3060
identified by PCI enumeration. It has approximately 7.2 GiB system RAM and 4 GiB swap.
GPU memory capacity is unverified until the driver works. It is not the old WSL A4000
machine, and that machine's memory measurements must not be treated as measurements here.

Driver evidence:

- `nvidia-smi`: cannot communicate with the NVIDIA driver.
- No NVIDIA compute kernel module or `/dev/nvidia*` device was found.
- `modinfo nvidia`: module not found.
- `linux-modules-nvidia-595-server-open-7.0.0-38-generic` has dpkg status `rc`
  (removed, residual configuration), while `nvidia-kernel-source-595-server-open` is installed.
- `sudo -n modprobe nvidia` requires interactive administrator authentication.

Machine-readable evidence and test logs are in `out/long-grpo-recovery/`.

### Validation on this mirror

| Check | Result | Evidence |
| --- | --- | --- |
| New correctness suite | 16 passed in 67.59 seconds; expected NVML warning | `pytest.log` |
| Existing GRPO self-check | Passed (`ok`) | `legacy-grpo.log` |
| Existing SFT loss/gradient self-check | Passed (`ok`) | `legacy-sft.log` |
| Existing data-preparation self-check | Blocked: sibling `src.harness` tree is absent | `legacy-prepare.log` |
| Bubblewrap terminal and hidden grader | Real four-action oracle episode passed, reward 2.46 | `preflight.json` |
| SFT generation | 64 independently verified episodes | `data/long_agent_sft_memory.jsonl` |
| Actual-tokenizer SFT validation | 256 assistant examples, 447–1,934 tokens; all fit 4K | `sft-validation.json` |
| CUDA availability | False; no CUDA devices exposed | `preflight.json` |
| GPU training attempt | Exited 2 at driver preflight; no weights loaded or changed | `gpu-attempt.json` |

The new tests cover executable success, false claims, verification gating, protected-file
tampering, persistent shell state, network/filesystem isolation, command timeout, the early
user-only requirement, bounded context construction, complete assistant-prefix rendering,
CUDA kernel selection with CPU embeddings, SFT label masking, group advantages, clipped
loss behavior, and chunked log-probability values/gradients against a full-logit reference.

## Files and checkpoint lineage

| File | Purpose |
| --- | --- |
| `train_long_grpo.py` | Sequential multi-turn GRPO, optional SFT warm-up, evaluation and memory profiling |
| `agent_grpo_env.py` | Real persistent terminal, disposable Bubblewrap workspace, verified coding tasks |
| `export_long_agent_sft.py` | Export independently verified assistant/tool trajectories for SFT |
| `validate_long_agent_sft.py` | Check actual-tokenizer lengths, preserved requirements and assistant labels |
| `test_long_grpo.py` | Reward, isolation, context, kernel-mode and gradient tests |
| `preflight_grpo.py` | Hardware, CUDA, package and executable-environment checks |
| `setup_long_grpo.sh` | Isolated Python 3.12 environment and local compiler setup |
| `requirements-long-grpo.txt` | Compatible training dependency constraints |
| `requirements-long-grpo.lock.txt` | Exact installed package versions on this mirror |

`train_grpo_gan.py` has a small loader fix: explicitly enable Hub kernels and select
CUDA training kernels before any embedding CPU offload. The original file is saved as
`out/long-grpo-recovery/train_grpo_gan.py.before_long_grpo`. The existing ORPO weights and old datasets are preserved.

The default lineage is the original `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` base plus
`out/nemotron-nano-4b-orpo/adapter`, loaded as a trainable PEFT adapter. Do not load the
already merged ORPO model and apply that same adapter again. The original base is cached
separately; `out/long-grpo-recovery/base_model.json` records the downloaded snapshot when available.
The download completed on this mirror at revision
`dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f`, matching the previously tested base snapshot.

This is adapter continuation. Passing `--adapter` restores weights, not the previous
optimizer/RNG state. New output directories are required, so an existing checkpoint is
not overwritten. No merge, deployment or replacement of the live model is performed.

## What the agent learns

Each task is an executable Python repair problem. The model reads terminal output,
edits `pipeline.py`, runs `python3 verify.py`, then calls `task_done`. It calls
`run_command(cmd, timeout)` / `task_done` as native tool calls: the tools go through the chat
template's `<tools>` block and each turn is one `<tool_call>`, the format every other dataset uses.
(Until 2026-10-21 this environment used the harness's plain-text JSON envelope; the old data is kept
as `data/*.json-protocol.jsonl`.)
The shell retains working directory and environment state between calls.

An early user instruction specifies an input offset that is absent from `contract.json`.
The later code must honor that instruction as well as the ordered stages in the contract.
Actual archived log output extends the history and supplies distractors. This tests
retention of an earlier requirement; it does not establish general long-horizon autonomy.

Each member of a GRPO group starts from an independent copy of the same task. Model
commands execute inside Bubblewrap, without host home directories, credentials, GPU
devices or network access. Hidden grading also runs in a separate sandbox. No generated
Python is imported by the training process. Timeouts stop the terminal process group.
The grader checks outputs against expectations held by the host, and checks that protected
files were preserved. This is a training sandbox, not a proof against arbitrary hostile programs.

The current task family is deliberately narrow. It provides a testable starting point;
it is not an Aider benchmark result or a general repository-repair curriculum.

## GRPO objective

For task x, sample G complete trajectories from the same current policy. Give trajectory i reward:

```text
R_i = 2 * verified_success
    + 0.5 * hidden_test_fraction * protected_files_intact
    - 1 * false_success_claim
    - 1 * protected_file_violation
    - 0.1 * malformed_actions
    - 0.05 * repeated_identical_commands
    - 0.01 * number_of_turns * verified_success

A_i = R_i - mean(R_1, ..., R_G)
ratio_it = exp(log_pi_new(action_token_it | history_it)
             - log_pi_old(action_token_it | history_it))

loss = -sum_i,t min(ratio_it * A_i, clip(ratio_it, 0.8, 1.2) * A_i)
       / (G * max_turns * max_new_tokens)
```

This uses group-relative mean centering without per-group standard-deviation scaling,
and fixed token normalization in the spirit of Dr. GRPO. All generated assistant tokens
receive the trajectory advantage. Tool observations and the seeded prefix receive no loss.
There is one optimizer update per sampled group. Identical-reward groups are skipped and
counted, rather than represented as productive updates. Sampling uses temperature-controlled
full-support sampling, and log-probabilities use the same temperature.

KL coefficient is currently zero: no frozen reference policy occupies VRAM. Use a small
learning rate and regression evaluation; a frozen ORPO-reference KL term is a future extension,
not an implemented feature. No reward is given for reasoning length or merely formatting a plan.

Truncated responses remain failures where appropriate rather than dropping their entire
negative signal. Context limits terminate an episode; the original instructions are not
silently left-truncated. Dense partial credit helps exploration, but frequent flat groups
mean the task is too difficult/easy or the policy needs SFT first.

## Memory and stability changes

- NF4 QLoRA continues the existing rank-16 adapter.
- Rollouts run one at a time; the group is stored on CPU as token IDs and old log-probabilities.
- Backward processes one assistant turn at a time and recomputes its full retained history.
- Gradient checkpointing and 32-position output-logit chunks avoid a full context-by-vocabulary tensor.
- Embeddings stay on GPU by default. Explicit CPU embedding offload reselects CUDA kernels
  after every train/eval mode change, avoiding the CPU-fallback bug found on the old host.
- Activation offload defaults off on this low-RAM mirror. It requires at least 12 GiB
  available host RAM when explicitly enabled. Contexts above 8K also require that headroom.
- WSL disables expandable CUDA allocator segments; native Linux retains the normal setting.
- Context construction has a bounded retry count and fails immediately if sandbox setup fails.
- CPU threads are limited; the sandbox process limit accounts for host-UID PyTorch threads.
- Profiling now uses the full requested completion-token budget, not just a short example call.

These checks reduce avoidable failures but cannot guarantee that a GPU or driver remains
stable under sustained load. The old server's failure cause has not been established.

## Historical A4000 measurements — not mirror results

Observed in this conversation on 2026-10-12, on the old WSL RTX A4000 16 GB / 31 GiB RAM host:

| Input tokens | Peak allocated GPU GiB | Peak reserved GPU GiB | Forward/backward seconds |
| ---: | ---: | ---: | ---: |
| 1,024 | 3.59 | 4.61 | 59.25, including first kernel compilation |
| 4,096 | 4.13 | 4.61 | 2.79 |
| 8,192 | 5.12 | 6.22 | 5.53 |
| 16,384 | 7.27 | 8.35 | 27.34, including compilation effects |
| 32,768 | 11.01 | 14.77 | 25.43 |

Those profiles used only 19 completion tokens, CPU embedding/activation offload, and finite
backward gradients. They were memory smoke tests, not full training runs or quality evaluations.
The figures were recovered from tool output in the conversation; the failed server's raw
log files have not been recovered. The mirror defaults and profiler have since changed.

The slow fallback path at 8K had used approximately 11.92 GiB allocated and 13 seconds.
The optimized kernel path reduced those figures substantially. Initial behavioral probes
showed the ORPO model getting distracted by archived logs and repeatedly inspecting files.
The old pilot completed baseline sampling and at least one SFT optimizer step, but no completed
GRPO run or final improved checkpoint was verified before the server connection failed.

## Driver repair and first GPU run

Administrator action is required before any GPU training. Ubuntu's driver tool listed
the 595-server-open branch as available. A matching module/userspace repair is:

```bash
sudo apt update
sudo apt install nvidia-driver-595-server-open linux-modules-nvidia-595-server-open-generic
sudo modprobe nvidia
nvidia-smi
```

If the installation requires a reboot or Secure Boot enrollment, complete that through
the normal administrator workflow. No reboot or driver-package changes were performed here.
See [Ubuntu's NVIDIA driver guide](https://ubuntu.com/server/docs/nvidia-drivers-installation/).

Then run from this repo:

```bash
cd '/home/pooh/computer use training'
bash setup_long_grpo.sh
.venv-grpo/bin/python preflight_grpo.py --out out/preflight-gpu.json --require-gpu
.venv-grpo/bin/python -m pytest test_long_grpo.py -q
.venv-grpo/bin/python validate_long_agent_sft.py

# Start conservatively on the RTX 3060 / 7.2 GiB RAM mirror.
.venv-grpo/bin/python train_long_grpo.py \
  --out out/long-grpo-mirror-profile --profile \
  --context-schedule 1024 2048 --max-context 4096 --max-new-tokens 384

# Historical pilot template; a completed pilot is reported above.
.venv-grpo/bin/python -u train_long_grpo.py \
  --out out/long-grpo-mirror-pilot \
  --steps 3 --group-size 2 --max-turns 6 --max-new-tokens 384 \
  --context-schedule 1024 2048 --max-context 4096 \
  --sft-warmup-steps 4 --sft-lr 1e-5 --lr 2e-6 \
  --eval-episodes 2 --save-every 1
```

Use a fresh output name when repeating a run. Check `metrics.jsonl` for nonzero updates,
finite gradients, reward spread and GPU peaks; inspect `episodes.jsonl` for actual actions.
`summary.json` separates original baseline, post-SFT evaluation and post-GRPO evaluation,
so improvements from SFT are not incorrectly attributed to GRPO.

To use the exported demonstrations rather than generating oracle episodes during warm-up,
add `--sft-data data/long_agent_sft_memory.jsonl`. Set `--steps 0` for an SFT-only run.
Each dataset row stores the conversation and explicit `assistant_loss_indices`; seeded
assistant context and all tool observations are excluded from those labels.

After the pilot, profile 4K and 8K before increasing the curriculum. Keep generation group
size at 2 initially, then try 4 for a less noisy baseline. Do not launch 16K/32K on this
low-RAM host using historical A4000 settings. Establish sustained stability before longer runs.

## Further training plan

1. **SFT cold start:** build 2,000–5,000 verified trajectories in the actual harness protocol.
   Include inspect → edit → test → interpret failure → repair → verify → stop, with both successful
   and honestly blocked endings. Train only assistant turns; retain tool outputs as context.
   The generated 64-episode seed dataset is for plumbing and initial behavior, not broad capability.
2. **Broaden executable tasks:** add small repository bugs, multi-file changes, JSON/CSV repair,
   failed commands, changed requirements, stale paths, and tests that catch an earlier constraint
   being forgotten. Split by repository/task family before making multiple context-length variants.
3. **GRPO curriculum:** begin with tasks the policy sometimes solves. Progress from 4–8 to 8–16
   actual tool decisions, separately from growing context length. Mix roughly 25% shorter contexts
   into later stages. On larger hardware, consider 2K → 4K → 8K → 16K only after measurement.
4. **Credit assignment:** compare the implemented terminal trajectory reward against verified
   milestone rewards or potential-based progress shaping. Cap shaping so it cannot outweigh
   actual task success. Add penalties for regression of previously passing tests and false claims.
   Do not use an LLM judge as the sole success signal.
5. **Retention:** interleave high-quality original tool-call SFT examples; consider a frozen ORPO
   reference KL term when memory permits. Track tool schema validity, honesty and stopping alongside
   task success, rather than optimizing one aggregate score without inspecting trajectories.

Recommended data sources are your own successful harness traces and executable training tasks
from the current language-specific [SWE-smith datasets](https://huggingface.co/datasets/SWE-bench/SWE-smith).
The older combined SWE-smith card recommends its language-specific successors. Prefer short,
verified trajectories for a 4B model, then add longer failure/recovery episodes.

For Aider, use separate held-out tasks and its actual edit format. The
[Aider polyglot benchmark](https://aider.chat/docs/leaderboards/) covers 225 exercises across six
languages; do not train on its evaluation exercises or expected patches. Terminal-tool success
does not imply good Aider SEARCH/REPLACE editing. Add disjoint edit-format SFT and evaluate both
first-attempt and repair-after-test-feedback success.

For real harness deployment, test against the complete local harness system prompt and tool
registry. This mirror contains the training directory, not the previous sibling `src` harness tree,
so a full live-harness regression run is not claimed. The specific Odysseus repository/protocol
has not been identified; no Odysseus integration is claimed. Normalize trajectories to whichever
adapter it actually accepts once that interface is available.

Use measured whole-episode success at several real turn horizons, with adequate held-out tasks.
Do not present a small per-turn score raised to a power as a measured autonomy result.
The [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer) provides background
on group-relative rewards and fixed-normalization loss variants; this extension uses its own
sequential trajectory loop to integrate the terminal environment.
