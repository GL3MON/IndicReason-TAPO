"""Evaluate or export a frozen SFT/RL checkpoint with reproducible scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from ..io import save_json
from ..checkpoints import snapshot_checkpoint
from .checkpoints import verify_checkpoint, export_checkpoint
from .datasets import load_eval_rows, select_failed_rows
from .scoring import evaluate_batches, summarize_results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-kind', choices=['sft', 'rl'], default='sft')
    parser.add_argument('--split', choices=['test', 'train-validation', 'dataset-test'], default='dataset-test')
    parser.add_argument('--retry-failures-from', type=Path, help='Rerun only incorrect questions from a completed evaluation')
    parser.add_argument('--dataset-manifest', type=Path, default=Path('data/vishal_gsm8k_tamil_test500_seed42.manifest.json'))
    parser.add_argument('--validation-report', type=Path)
    parser.add_argument('--language', choices=['ta', 'en'], default='ta')
    parser.add_argument('--merged-model', type=Path)
    parser.add_argument('--stage', choices=['export', 'evaluate'], required=True)
    parser.add_argument('--adapter', type=Path, default=Path('outputs/qwen35_ta_sft_2000/adapter'))
    parser.add_argument('--test', type=Path, default=Path('data/vishal_gsm8k_tamil_test500_seed42.jsonl'))
    parser.add_argument('--output', type=Path, default=Path('outputs/evaluation'))
    parser.add_argument('--keep-merged', action='store_true', help='Keep exported inference weights instead of using a temporary directory')
    args = parser.parse_args()
    return args


def evaluate(args, merged, exported, rows, manifest, original_n):
    os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
    from vllm import LLM
    llm = LLM(model=str(merged.resolve()), dtype='bfloat16', gpu_memory_utilization=0.55,
              max_model_len=4096, max_num_seqs=8, enforce_eager=True,
              gdn_prefill_backend='triton', limit_mm_per_prompt={'image': 0, 'video': 0}, seed=42)
    metadata = {**exported, 'dataset': 'openai/gsm8k', 'split': args.split, 'test_file': str(args.test),
        'test_sha256': hashlib.sha256(args.test.read_bytes()).hexdigest(), 'n': len(rows),
        'language': args.language, 'decoding': 'greedy', 'enable_thinking': False, 'max_new_tokens': 1536,
        'mig': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'limitation': 'Machine-translated Tamil subset; not the full English benchmark.'}
    if manifest is not None:
        metadata.update(dataset=manifest['dataset'], dataset_revision=manifest['revision'], sample_seed=manifest['seed'],
                        sample_scope=manifest['sample_scope'], dataset_manifest=str(args.dataset_manifest), limitation=manifest['limitation'])
    if args.retry_failures_from:
        metadata.update(retry_of=str(args.retry_failures_from), original_n=original_n,
                        retry_scope='Previously incorrect questions only; one greedy retry with unchanged settings')
    save_json(args.output / 'metadata.json', metadata)
    for results in evaluate_batches(rows, llm, args.language):
        (args.output / 'sft_outputs.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=True)+'\n' for row in results))
        summary = {**metadata, **summarize_results(results), 'status': 'complete' if len(results)==len(rows) else 'running'}
        save_json(args.output / 'summary.json', summary)
        print(f"Evaluation: {len(results)}/{len(rows)}; correct={summary['correct']}; accuracy={summary['accuracy']:.1%}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.checkpoint_kind == 'rl' and not (args.adapter / 'checkpoint_snapshot.json').exists():
        source = args.adapter
        args.adapter = args.output / 'adapter'
        snapshot_checkpoint(source, args.adapter)
    adapter_sha, report, provenance = verify_checkpoint(args.adapter, args.checkpoint_kind, args.split)
    rows, manifest, original_n = load_eval_rows(args, report, adapter_sha)
    temporary = None
    if args.stage == 'export' or args.keep_merged:
        merged = args.merged_model or args.output / 'merged_model'
    elif args.merged_model is not None:
        merged = args.merged_model
    else:
        temporary = tempfile.TemporaryDirectory(prefix='indic-reasoning-eval-')
        merged = Path(temporary.name) / 'merged_model'
    try:
        if args.stage == 'export' or not (merged / 'config.json').exists():
            exported = export_checkpoint(args.adapter, merged, adapter_sha, provenance)
        else:
            exported = json.loads((merged.parent / 'export.json').read_text())
            assert exported['adapter_sha256'] == adapter_sha, 'Merged checkpoint is stale'
        if args.stage == 'export':
            return
        evaluate(args, merged, exported, rows, manifest, original_n)
    finally:
        if temporary is not None:
            temporary.cleanup()
