"""Adapter provenance checks and standalone BF16 model export."""
import hashlib
import json
from pathlib import Path
from ..io import save_json


def verify_checkpoint(adapter, kind, split):
    adapter_sha = hashlib.sha256((adapter / 'adapter_model.safetensors').read_bytes()).hexdigest()
    report = None
    if kind == 'sft':
        report = json.loads((adapter.parent / 'sft_report.json').read_text())
        assert report['status'] == 'complete' and report['adapter_sha256'] == adapter_sha
        provenance = {'checkpoint_kind': 'sft', 'global_step': report['global_step']}
    else:
        assert split in ('test', 'dataset-test'), 'RL snapshots require a test split'
        provenance = json.loads((adapter / 'checkpoint_snapshot.json').read_text())
        state = json.loads((adapter / 'trainer_state.json').read_text())
        assert provenance['checkpoint_kind'] == 'rl' and provenance['adapter_sha256'] == adapter_sha
        assert state['global_step'] == provenance['global_step']
    return adapter_sha, report, provenance


def export_checkpoint(adapter, merged, adapter_sha, provenance):
    import torch
    from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer
    from peft import PeftModel
    config = json.loads((adapter / 'adapter_config.json').read_text())
    base = config['base_model_name_or_path']
    assert base == 'Qwen/Qwen3.5-2B'
    model = Qwen3_5ForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16, device_map='cpu', local_files_only=True)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    model = model.merge_and_unload(safe_merge=True)
    model.save_pretrained(merged, safe_serialization=True, max_shard_size='2GB')
    AutoTokenizer.from_pretrained(base, local_files_only=True).save_pretrained(merged)
    from huggingface_hub import hf_hub_download
    import shutil
    for name in ("preprocessor_config.json", "video_preprocessor_config.json"):
        shutil.copy2(hf_hub_download(base, name, local_files_only=True), merged / name)
    save_json(merged.parent / 'export.json', {'base_model': base, 'adapter': str(adapter), 'adapter_sha256': adapter_sha, 'merged_model': str(merged), **provenance})
    print('Exported merged model:', merged, flush=True)
    return json.loads((merged.parent / "export.json").read_text())
