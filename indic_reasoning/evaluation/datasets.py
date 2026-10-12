"""Evaluation split verification and failure-only retry selection."""
import hashlib
import json
from pathlib import Path
import re
from ..io import read_jsonl

def select_failed_rows(rows, previous):
    if len(rows) != len(previous) or [r['source_id'] for r in rows] != [r['source_id'] for r in previous]:
        raise ValueError("Retry outputs must match the complete original question set in order")
    for row, old in zip(rows, previous):
        if not isinstance(old.get('correct'), bool):
            raise ValueError("Previous results must contain boolean correctness scores")
        if any(row[key] != old[key] for key in ('question', 'english_question', 'answer')):
            raise ValueError("Retry questions and gold answers must be unchanged")
    failed = [row for row, old in zip(rows, previous) if not old['correct']]
    if not failed:
        raise ValueError("There are no failed questions to retry")
    return failed

def load_eval_rows(args, report, adapter_sha):
    manifest = None
    rows = read_jsonl(args.test)
    training_rows = {r['source_id']: r for r in read_jsonl('data/gsm8k_train_ta.jsonl')}
    assert rows and len({r['source_id'] for r in rows}) == len(rows)
    if args.split == 'test':
        train_questions = {r['english_question'] for r in training_rows.values()}
        assert len(rows) == 200
        assert all(re.fullmatch(r'gsm8k_test_\d+', r['source_id']) and r['english_question'] not in train_questions for r in rows)
    elif args.split == 'dataset-test':
        assert args.dataset_manifest, 'External dataset evaluation requires a manifest'
        manifest = json.loads(args.dataset_manifest.read_text())
        assert manifest['test_sha256'] == hashlib.sha256(args.test.read_bytes()).hexdigest()
        assert manifest['n'] == len(rows) and manifest['training_overlap'] == 0
        assert manifest['source_ids'] == [r['source_id'] for r in rows]
        train_questions = {r['english_question'] for r in training_rows.values()}
        assert all(re.fullmatch(r'gsm8k_test_\d+', r['source_id']) and r['english_question'] not in train_questions for r in rows)
    else:
        assert args.validation_report, 'Validation requires an explicit split report'
        validation_report = json.loads(args.validation_report.read_text())
        ids = {r['source_id'] for r in rows}
        assert ids == set(validation_report['eval_source_ids'])
        assert not ids & set(validation_report['train_source_ids'])
        assert not ids & set(report['dataset']['train_source_ids']), 'Evaluated adapter trained on validation questions'
        for r in rows:
            original = training_rows[r['source_id']]
            assert all(r[key] == original[key] for key in ('question', 'english_question', 'answer'))

    original_n = len(rows)
    if args.retry_failures_from:
        assert args.stage == 'evaluate', 'Retry requires an existing merged checkpoint'
        previous_summary = json.loads((args.retry_failures_from.parent / 'summary.json').read_text())
        assert previous_summary['status'] == 'complete'
        assert previous_summary['adapter_sha256'] == adapter_sha, 'Retry must use the same checkpoint'
        assert previous_summary['test_sha256'] == hashlib.sha256(args.test.read_bytes()).hexdigest()
        assert previous_summary['language'] == args.language
        rows = select_failed_rows(rows, read_jsonl(args.retry_failures_from))
        print(f'Retrying {len(rows)} failed questions out of {original_n}.', flush=True)

    return rows, manifest, original_n
