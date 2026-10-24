# Computer-use agent training

Fine-tunes NVIDIA Nemotron-3-Nano-4B (LoRA) into a tool-calling coding agent: dataset builders,
sandboxed task environments, ORPO / SFT / GRPO trainers, an OpenAI-compatible server for the trained
adapter, and evaluations through real agent harnesses (opencode, Cline, the assistant harness).

## Layout

```
cutrain/                 the Python package (run its entry points as modules, see below)
  paths.py               repository locations (data/, out/, virtualenvs) used by every module
  envs/                  sandboxed tasks the agent acts in, with graders and rewards
  training/              trainers and the helpers they share
  dataprep/              build, convert and validate datasets in data/
  serving/               serve an adapter over an OpenAI-compatible API, export to Ollama
  evaluation/            probe and real-use evaluations of a served adapter
scripts/                 launchers: each starts a pipeline as a detached systemd user service
tests/                   pytest suite
docs/                    design notes, run reports and the command reference
harness_eval/            evaluation through the assistant harness (runs in .venv-harness)
specs/                   app specs and judging criteria for the real-use build evaluations
data/                    datasets (inputs and generated)
out/                     training runs, logs, evaluation evidence
Modelfile                Ollama model template used by cutrain.serving.to_ollama
requirements-*.txt       dependencies for .venv-grpo
```

### What each module does

| Module | Purpose |
|---|---|
| `envs/agent_grpo_env` | Terminal pipeline-repair tasks for long-horizon GRPO, in a bubblewrap sandbox |
| `envs/opencode_env` | Episodes with opencode's exact prompt, tools and tool-result text |
| `envs/project_env`, `project_tasks`, `project_grader` | Whole projects from a plain-language outline, graded independently |
| `envs/mimo_env` | MiMo code / terminal_bench tasks in containers, plus reasoning tasks |
| `training/train_orpo` | ORPO on the unified preference dataset (also defines the chat-template constants) |
| `training/train_sft_think` | SFT that restores reasoning (non-empty `<think>`) |
| `training/train_grpo_gan` | GRPO with a self-critique loop; provides `load_model` for the other trainers |
| `training/train_long_grpo` | Multi-turn GRPO in any env above, with SFT warm-up and pause/resume |
| `training/train_loop_preferences`, `anti_loop` | Loop detection, penalties and loop-vs-verified preference training |
| `training/agent_critique` | Self-critique of failed trajectories and retry feedback |
| `training/continue_nemotron_cu`, `preflight_grpo` | Launch a continuation run; record hardware/software evidence |
| `dataprep/prepare_data`, `prepare_hf` | Claude/opencode transcripts and Hub datasets → unified schema |
| `dataprep/build_dataset`, `prepare_full_run` | Mix and render the SFT corpus |
| `dataprep/build_opencode_sft`, `build_project_sft`, `export_*_sft` | Verified demonstrations from the environments |
| `dataprep/make_think_pairs`, `make_stop_pairs`, `gen_rejected`, `scrub_paths` | Preference pairs and dataset clean-up |
| `dataprep/prepare_mix_grpo`, `validate_long_agent_sft` | Mixed GRPO task pool; SFT data validation |
| `serving/serve_adapter` | Chat server for an adapter; also the shared tool-call protocol (render, parse) |
| `serving/to_ollama` | Merged model → GGUF → Ollama |
| `evaluation/probe_agent` | Scripted probes through the real harness prompt and tools |
| `evaluation/opencode_agent_test`, `build_app_with_opencode`, `build_app_with_cline` | Real-use tests through opencode and Cline |

## Running

Everything runs from the repository root with the `.venv-grpo` interpreter
(create it with `bash scripts/setup_long_grpo.sh`):

```bash
.venv-grpo/bin/python -m cutrain.training.train_long_grpo --help    # any module: python -m cutrain.<area>.<module>
bash scripts/run_mix_grpo.sh                                        # start a pipeline as a user service
bash scripts/train_ctl.sh pause | resume | status                   # free the GPU for something else, then continue
.venv-grpo/bin/python -m pytest                                     # tests
```

The launchers need `loginctl enable-linger $USER` once, and the environments need bubblewrap
(and Docker for `mimo_env`). See [docs/PIPELINE_AND_COMMANDS.md](docs/PIPELINE_AND_COMMANDS.md)
for the full command reference.

## Documentation

- [docs/PIPELINE_AND_COMMANDS.md](docs/PIPELINE_AND_COMMANDS.md): how to continue training, pause/resume, export to Ollama
- [docs/GRPO_PLAN_AND_REPORT.md](docs/GRPO_PLAN_AND_REPORT.md): the long-horizon GRPO design and its results
- [docs/FULL_TRAINING_RUN.md](docs/FULL_TRAINING_RUN.md): the full SFT → GRPO run: data quotas and stages
- [docs/ANTI_LOOP_TRAINING.md](docs/ANTI_LOOP_TRAINING.md): loop penalties, stopping SFT, loop preferences
- [docs/HARNESS_EVAL.md](docs/HARNESS_EVAL.md): the assistant-harness evaluation
- [docs/RUN_NOTES.md](docs/RUN_NOTES.md): decisions and incidents from earlier runs

## Layout change (2026-10-21)

The modules used to sit flat in the repository root. Reports written before this date use the old
commands; translate them as follows:

| Before | Now |
|---|---|
| `python train_long_grpo.py ...` | `python -m cutrain.training.train_long_grpo ...` (area from the table above) |
| `bash run_mix_grpo.sh` | `bash scripts/run_mix_grpo.sh` |
| `test_*.py` in the root | `tests/` |
| `*.md` notes in the root | `docs/` |

The pre-change source is archived in `out/backup-before-reorg-20261021T212813Z.tar.gz`.
