# IndicReasoning

Tamil GSM8K translation, completion-only supervised fine-tuning, TAPO-style reinforcement learning, and reproducible checkpoint evaluation with Qwen3.5-2B.

The public standalone model is [GL3MON/Qwen3.5-2B-Tamil-TAPO](https://huggingface.co/GL3MON/Qwen3.5-2B-Tamil-TAPO), uploaded from RL step 1525. Its [model card](docs/model_card.md) distinguishes uploaded weights from evaluated checkpoints.

## Code layout

```text
indic_reasoning/
  constants.py          Prompt format and supported trainer versions
  config.py             CLI/YAML parsing, validation, sampling controls
  io.py                 Physical JSONL record I/O
  data.py               Official GSM8K alignment and context filtering
  sft_data.py           Worked-solution joins and completion-only targets
  formatting.py         Translation blocks, boxed answers, token boundaries
  rewards.py            ChrF++, correctness, rollout-group diagnostics
  modeling.py           Trainable LoRA loading and vLLM weight synchronization
  trainer.py            TAPO advantages and policy-gradient loss
  checkpoints.py        Complete-checkpoint selection and frozen snapshots
  training/             SFT and RL orchestration
  evaluation/           Split validation, export, scoring, retry handling
  runtime/              RAM supervision, CPU limits, worker cleanup
  translation/          Sarvam API client and resumable translation workflows
configs/                Training and translation configurations
scripts/env.sh          Shared launcher environment; no Conda activation
patches/                Reproducible vLLM/TRL compatibility fixes
```

The two root training scripts are thin compatibility entry points. Evaluation code stays inside `indic_reasoning/evaluation/` and is invoked through `python -m indic_reasoning evaluate`; translation also uses the package CLI. The package imports ML dependencies only when required; the training supervisor starts before them.

## Environment

Use the existing `med` environment. The launchers use your active environment and do not activate Conda.

```bash
conda activate med
python -m indic_reasoning --help
```

The tested environment uses PyTorch 2.11.0+cu129, Transformers 5.6.2, PEFT 0.20.0, TRL 1.9.2, Accelerate 1.14.0, datasets 5.0.1, vLLM 0.22.0+cu129, sacrebleu 2.5.1, math-verify 0.8.0, and W&B 0.28.1. `pyproject.toml` describes package dependencies and optional training/vLLM extras. Run `python -m pip install -e . --no-deps` to install the package into an already configured environment. Installing ML/CUDA dependencies is a separate environment setup step.

The local vLLM/TRL fixes are retained under `patches/`: MIG UUID/NVML handling, eager-mode selection, the Triton GDN backend, and visible gold-answer columns. Apply them with the corresponding `patches/apply_*.py` scripts after changing those libraries. The TAPO sleep-mode policy synchronization lives in `modeling.py` and avoids checkpoint-reload temporary GPU allocations.

## Retained checkpoints

Only these model versions are retained locally:

| Model version | Inference adapter | Resumable training checkpoint |
|---|---|---|
| 2,000-example SFT | `outputs/qwen35_ta_sft_2000/adapter` | `outputs/qwen35_ta_sft_2000/checkpoint-500` |
| Evaluated RL step 1375 | `outputs/qwen35_rl_step_1375_vishal500_eval/adapter` | Adapter snapshot only |
| Uploaded RL step 1525 | `outputs/tapo_ta_qwen35_sft_2000_rl/checkpoint-1525` | Same folder, including optimizer/scheduler state |

Evaluation reports and prediction files remain under `outputs/`. Historical results may reference older adapters that were removed at the user's request. Small data/provenance files from the earlier warm-up remain where needed to verify validation exclusions. Duplicate merged inference copies, smoke-test checkpoints, obsolete runs, and inactive W&B artifacts are removed. [Cleanup manifest](docs/output_cleanup.json) records what was removed and the retained weight checksums.

`data/`, `outputs/`, W&B artifacts, credentials, model weights, and Python caches are excluded from Git. Training data and checkpoint files remain local; Git tracks the code, configs, documentation, and tests.

## Training

Translate official GSM8K TRAIN questions through a locally hosted Sarvam server:

```bash
python -m indic_reasoning translate --concurrency 256
```

The resumable translation writer stores aligned Tamil questions, English originals, gold answers, and stable `gsm8k_train_<index>` IDs in `data/gsm8k_train_ta.jsonl`. The generic CSV/dataset translation tools remain available in `indic_reasoning.translation` using `python -m indic_reasoning.translation.translate_gsm8k`.

Train the 2k SFT preset and evaluate its clean TRAIN validation subset on the second MIG:

```bash
bash run_qwen35_sft_2000.sh
```

This trains one epoch over 2,000 worked solutions, holds out 128 TRAIN questions, saves/evaluates every 50 steps, and selects the checkpoint by validation loss. Gold solution targets are never truncated. Official test questions are excluded from checkpoint selection. An existing completed SFT report skips retraining. If earlier warm-up source IDs are available, validation also excludes them.

Start RL from the selected SFT adapter on the third MIG:

```bash
bash run_qwen35_sft_2000_rl.sh
```

The preset keeps eight rollouts and excludes the 128 validation questions from its 7,345-row RL pool. It logs to W&B in `indic-reasoning-tapo`, including `correct_answer`, per-completion rewards, and all-wrong/all-correct/mixed rollout fractions. TAPO uses separate translation and correctness advantages, alpha 0.25, beta 0, token importance sampling, and the DAPO loss. This is a TRL/LoRA adaptation, not the paper's official implementation.

To resume the uploaded checkpoint without rotating out the retained copy, use a new output directory:

```bash
source scripts/env.sh
python -m indic_reasoning train \
  --config configs/tapo_ta_qwen35_sft_2000_rl.yaml \
  --resume outputs/tapo_ta_qwen35_sft_2000_rl/checkpoint-1525 \
  --output-dir outputs/tapo_ta_qwen35_continued \
  --no-memory-guard
```

The RAM cutoff remains disabled in the launchers as requested. CPU/compile worker caps and process cleanup remain enabled. Remove `--no-memory-guard` for the optional 16 GiB RAM cutoff.

## Evaluation

Recreate the pinned 500-question holdout:

```bash
python -m indic_reasoning prepare-test
```

`Vishal0407/GSM8K_TAMIL` combines 7,473 official training problems and 1,319 test problems. The generator audits row ordering against official answers/calculation signatures, then uses `random.Random(42).sample(range(1319), 500)`. The subset and its revision/hash manifest are saved in `data/`. Questions are unchanged; translation quality can affect scores.

Evaluate the retained RL adapter on the second MIG:

```bash
source scripts/env.sh
CUDA_VISIBLE_DEVICES=MIG-1f695d4f-ec71-5ad3-a117-778dcddf27d1 \
python -m indic_reasoning evaluate --stage evaluate --checkpoint-kind rl \
  --adapter outputs/qwen35_rl_step_1375_vishal500_eval/adapter \
  --output outputs/rl1375_rerun
```

For SFT, use `--checkpoint-kind sft --adapter outputs/qwen35_ta_sft_2000/adapter`. The defaults select the same 500-question manifest and one greedy answer per question, with hidden thinking disabled and a 1,536-token completion cap.

Evaluation automatically exports temporary merged weights inside its output folder and removes them when the run ends. Use `--keep-merged` to retain a cache, or `--stage export --output outputs/exported_model` for a standalone merged model. Results save incrementally; the original `sft_outputs.jsonl` filename remains compatible for both SFT and RL evaluations.

For a running job, freeze the latest complete checkpoint before evaluating:

```bash
python -m indic_reasoning snapshot --run-dir outputs/tapo_ta_qwen35_continued
```

Passing a complete `checkpoint-*` directly to the evaluator also creates a frozen adapter under the evaluation output before loading it. A failure-only retry uses `--retry-failures-from <previous-output>/sft_outputs.jsonl`; it verifies identical questions, gold answers, language, dataset hash, and adapter hash. Retry results are separate from the original one-answer benchmark.

### Recorded one-answer results

Same fixed 500-question Tamil subset, seed 42:

| Model | Correct | Accuracy | Extractable boxed answers |
|---|---:|---:|---:|
| 2k SFT | 180/500 | 36.0% | 97.2% |
| RL step 1125 (historical; weights removed) | 184/500 | 36.8% | 97.8% |
| RL step 1375 | 186/500 | 37.2% | 97.8% |

The small differences do not establish a reliable improvement. Uploaded step 1525 has not been evaluated. These results are not full English GSM8K accuracy, Coverage@8, or combined retry success rates. [Versioned evaluation summary](docs/evaluation_summary.json) records dataset and adapter hashes.

## Validation

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests
python -m indic_reasoning train --self-test
```

The suite checks official-data alignment, held-out splits, Unicode-safe JSONL loading, reward semantics, token segmentation, TAPO gradients versus stock DAPO, trained SFT-to-RL adapter loading, checkpoint snapshots, CLI/YAML precedence, and supervisor cleanup.
