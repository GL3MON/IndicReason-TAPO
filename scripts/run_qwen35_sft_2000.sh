#!/usr/bin/env bash
set -euo pipefail
TAPO_DEFAULT_MIG=MIG-1f695d4f-ec71-5ad3-a117-778dcddf27d1
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/env.sh"
validation_flags=()
if [[ -f outputs/qwen35_ta_sft/train_examples.jsonl ]]; then
    validation_flags=(--exclude-validation-examples outputs/qwen35_ta_sft/train_examples.jsonl)
fi
if [[ ! -f outputs/qwen35_ta_sft_2000/sft_report.json ]]; then
    python -m indic_reasoning sft --no-memory-guard \
        --output-dir outputs/qwen35_ta_sft_2000 --max-samples 2000 --eval-samples 128 \
        --epochs 1 --save-steps 50 --select-best \
        "${validation_flags[@]}" \
        --run-name qwen35-tamil-sft-2000
fi
# Keep official test questions out of checkpoint selection.
python -m indic_reasoning evaluate --stage evaluate --checkpoint-kind sft \
    --adapter outputs/qwen35_ta_sft_2000/adapter --output outputs/qwen35_sft_2000_validation \
    --split train-validation --test outputs/qwen35_ta_sft_2000/eval_examples.jsonl \
    --validation-report outputs/qwen35_ta_sft_2000/dataset_report.json
