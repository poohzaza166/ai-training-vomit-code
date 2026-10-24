# Full CU continuation run

Run commands:

```bash
cd '/home/pooh/computer use training'
bash run_full_training.sh
```

The script is arranged as a settings block, launch helpers, data preparation, and
four numbered stages. Each training option and source quota has its own line;
embedded Python is written as readable blocks. Edit the settings at the top or
use the environment overrides shown below.

The default launches a user systemd service with the tested mirror's 5G memory-high,
6G memory-max and 512M swap limits. It prints the service name, log path and run
folder. The service continues after SSH disconnects. Read progress with
`tail -f 'PRINTED_LOG_PATH'` and status with `systemctl --user status PRINTED_SERVICE`.
Do not start it alongside another GPU training job. A lock prevents concurrent
copies of this script. An error stops the pipeline; it does not silently skip a
failed stage. Existing output directories are refused.

## Data selection uses the existing builder

`build_dataset.py` reads `unified.jsonl`, `hf_agent.jsonl`, and `hf_reasoning.jsonl`,
deduplicates IDs, and selects these source quotas with seed 42:

| Source | Requested rows |
|---|---:|
| OpenCode | 1,000 |
| Claude | 400 |
| OpenThoughts agent | 600 |
| Arcee/Glaive | 600 |
| NVIDIA When2Call | 600 |
| DAPO maths | 400 |
| DeepSeek distillation | 600 |
| GLM reasoning | 300 |
| ORPO/DPO mix | 200 |

These are explicit starting quotas, not an experimentally optimized mixture.
Change the `--quota` arguments in the script to change them. The script rebuilds
from source files rather than stacking overlapping train_mix/train_full exports.

`prepare_full_run.py` then renders that selection with the actual model tokenizer
and chat template, including tool definitions and reasoning. It adds both verified
long-context and stopping episode files and expands their labelled assistant turns.
Only completion tokens receive SFT loss; prompt/tool-output tokens do not.
Completions receive one actual EOS. Examples exceeding 4,096 total tokens are
excluded, with per-source counts in `mixed_sft.report.json`; history and reasoning
are not silently trimmed. Consequently final source proportions differ from the
requested quotas. The builder's legacy optional trimming-based fit check is not
used. Prepared data is tied to the selected tokenizer/template.

DeepSeek, GLM, DAPO and other static examples participate in SFT here. There is no
new DAPO RL environment in this script. Terminal GRPO still uses the executable
pipeline-editing environment; neither broad research verification nor an Aider
benchmark runner has been implemented. Old arbitrary rejected examples are not
assumed to be verified anti-loop preference pairs.

## Stages and defaults

1. Profile a 5,760-token prompt plus 384-token completion through forward/backward.
   Stop on failure before optimization. This is a feasibility gate, not a runtime
   or memory guarantee for every possible batch.
2. Mixed SFT: one pass over all retained completion examples, learning rate 5e-6,
   context cap 4,096, continuing the original CU ORPO adapter. This stage has zero
   GRPO groups. It saves the continued adapter.
3. Shorter-context GRPO: 40 groups, group size four, starting contexts 1,024/2,048,
   total cap 4,096, up to eight actions, 384 generated tokens per action.
4. Longer-context GRPO: another 40 groups, starting contexts 2,048/4,096, total cap
   6,144. Full-history overflow ends the rollout instead of dropping earlier evidence.

Both GRPO stages use learning rate 2e-6, sequential rollouts, QLoRA, checkpointed
activations, 32-token logit chunks, and no CPU/activation offload. They enable the
existing self-critic/feedback retry path and the 0.5 capped token-loop penalty.
Critique does not supply the executable reward. A feedback retry consumes a group;
40 groups does not guarantee 40 optimizer updates, because equal-reward groups skip.

Each training stage evaluates four held-out seeds per configured context before
and after its changes. Compare within-stage results at matching context; the two
GRPO stages do not have identical evaluation context schedules. Evaluate task
success alongside loops, premature stopping and truncation. This terminal-only
check is not evidence of general benchmark gains or hallucination reduction.

The final adapter is `RUN_DIR/grpo-long/adapter`, also recorded in
`RUN_DIR/FINAL_ADAPTER.txt`. Earlier stage adapters remain available if later
training regresses. No merging, GGUF conversion or Ollama deployment is performed.
The script does not promise quality improvement. SFT saves at the end of its stage;
GRPO also saves periodically. Failed stages are not automatically resumed; use a
new run directory and explicit checkpoint selection for a deliberate continuation.

## Configuration and preparation-only check

```bash
SFT_EPOCHS=1 SHORT_STEPS=40 LONG_STEPS=40 GROUP_SIZE=4 EVAL_EPISODES=4 \
  RUN_DIR=out/my-full-cu-run bash run_full_training.sh

# Build/filter data and print every trainer command, without loading GPU weights:
RUN_DIR=out/my-preparation-check bash run_full_training.sh --prepare-only
```

`MODEL` defaults to the cached base snapshot in the recovery manifest;
`START_ADAPTER` defaults to `out/nemotron-nano-4b-orpo/adapter`. `SEED` defaults to 42.
`MODEL` must be a local base-model directory. Never point it at an Ollama GGUF.
These defaults target the current RTX 3060 12GB mirror. Total duration is unmeasured;
a complete SFT pass is substantially longer than the earlier four-step smoke run.

## Validation

The end-to-end `--prepare-only` run completed successfully; all four trainer
commands were generated (`out/logs/full-script-validation-v3-20261015.log`).
The focused adapter and terminal regression suites passed: 19 tests in 16.48s.
The real-data preparation retained 4,700 labelled completions: 600 DeepSeek,
300 GLM, 552 OpenThoughts, 739 OpenCode, 269 Claude, 600 Arcee, 600 When2Call,
400 DAPO, 200 ORPO/DPO-mix and 440 verified terminal/stopping actions.
The 4,700 initially selected source rows lost 440 over-context examples and gained
440 verified actions; the equal initial/final count is coincidental. The largest
retained example has 4,095 tokens. Actual-data checks confirmed preservation of
2,136 reasoning fields and tool names in 2,712 tool-bearing prompts.

No full GPU training job has been launched. The previous anti-loop smoke run
validated the existing SFT/GRPO engine, not this complete mixed-data curriculum.
Full-run improvements and runtime remain unmeasured.
