"""Select complete checkpoints and freeze adapters before training rotates them."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from .io import save_json

REQUIRED = ('adapter_model.safetensors', 'adapter_config.json', 'trainer_state.json')

def latest_checkpoint(run_dir):
    for path in sorted(Path(run_dir).glob('checkpoint-*'), key=lambda p:int(p.name.split('-')[-1]), reverse=True):
        if all((path/name).is_file() for name in REQUIRED):
            state=json.loads((path/'trainer_state.json').read_text())
            if state['global_step']==int(path.name.split('-')[-1]):
                return path
    raise ValueError(f'No complete checkpoint in {run_dir}')

def snapshot_checkpoint(source, destination):
    source, destination=Path(source),Path(destination)
    if not all((source/name).is_file() for name in REQUIRED):
        raise ValueError(f'Incomplete checkpoint: {source}')
    state=json.loads((source/'trainer_state.json').read_text())
    if source.name.startswith('checkpoint-') and state['global_step']!=int(source.name.split('-')[-1]):
        raise ValueError('Checkpoint directory and trainer step disagree')
    digest=hashlib.sha256((source/'adapter_model.safetensors').read_bytes()).hexdigest()
    destination.mkdir(parents=True,exist_ok=True)
    existing=destination/'checkpoint_snapshot.json'
    if existing.exists() and json.loads(existing.read_text())['adapter_sha256']!=digest:
        raise ValueError('Snapshot destination already contains a different checkpoint')
    for name in REQUIRED:
        shutil.copy2(source/name,destination/name)
    if hashlib.sha256((destination/'adapter_model.safetensors').read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Checkpoint changed while being copied')
    metadata=dict(source_checkpoint=str(source),global_step=state['global_step'],adapter_sha256=digest,checkpoint_kind='rl')
    save_json(existing,metadata)
    return metadata

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=Path('outputs/tapo_ta_qwen35_sft_2000_rl'))
    parser.add_argument('--output',type=Path,help='Snapshot folder; default includes the saved step')
    args=parser.parse_args()
    source=latest_checkpoint(args.run_dir)
    destination=args.output or Path('outputs/checkpoint_snapshots')/source.name
    print(json.dumps(snapshot_checkpoint(source,destination),indent=2))
    print('Evaluate with --checkpoint-kind rl --adapter',destination)
