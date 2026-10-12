#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/env.sh"
python - <<'VERIFY'
import json,hashlib
from pathlib import Path
root=Path('outputs/qwen35_ta_sft_2000')
r=json.loads((root/'sft_report.json').read_text())
comparison=root/'validation_comparison.json'
c=json.loads(comparison.read_text()) if comparison.exists() else None
assert r['status']=='complete' and r['global_step']==500, 'SFT must complete before RL'
if c is not None:
    assert c['ready_for_rl'], 'SFT did not pass its matched validation comparison'
else:
    v=json.loads(Path('outputs/qwen35_sft_2000_validation/summary.json').read_text())
    assert v['status']=='complete' and v['accuracy']>0
    assert r['after_sft']['after_sft_loss']<r['before_sft']['before_sft_loss']
assert hashlib.sha256((root/'adapter/adapter_model.safetensors').read_bytes()).hexdigest()==r['adapter_sha256']
validation=set(r['dataset']['eval_source_ids'])
training={json.loads(line)['source_id'] for line in (root/'rl_train.jsonl').read_text().splitlines() if line.strip()}
assert not validation & training
assert len(training)==r['dataset']['matched_train_questions']-len(validation)
print('Verified larger SFT and clean validation; starting RL.',flush=True)
VERIFY
exec python -u -m indic_reasoning train --config configs/tapo_ta_qwen35_sft_2000_rl.yaml --no-memory-guard "$@"
