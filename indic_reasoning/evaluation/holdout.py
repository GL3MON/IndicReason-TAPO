#!/usr/bin/env python3
"""Seeded 500-question test subset from the combined Vishal0407 Tamil GSM8K file."""
import hashlib
import json
from pathlib import Path
import random
import re
from datasets import Dataset
from huggingface_hub import hf_hub_download

REPO = 'Vishal0407/GSM8K_TAMIL'
REVISION = '982060cd2f7434e5c07514e22bf13d70b373cd97'

def signature(answer):
    gold = answer.rsplit('####', 1)[-1].strip().replace(',', '')
    expressions = tuple(re.sub(r'[\s\u200b\u200c\u200d\ufeff]+', '', text)
                        for text in re.findall(r'<<([^<>]*)>>', answer))
    return gold, expressions

def main():
    path = Path(hf_hub_download(REPO, 'gsm8k_tamil.json', repo_type='dataset', revision=REVISION))
    rows = json.loads(path.read_text())
    cache = Path.home() / '.cache/huggingface/datasets/openai___gsm8k/main'
    train = Dataset.from_file(str(next(cache.glob('*/*/gsm8k-train.arrow'))))
    test = Dataset.from_file(str(next(cache.glob('*/*/gsm8k-test.arrow'))))
    assert len(rows) == len(train) + len(test) == 8792
    audit = {}
    for split, start, official in [('train', 0, train), ('test', len(train), test)]:
        gold_matches = expression_matches = 0
        mismatches = []
        for index, original in enumerate(official):
            actual, expected = signature(rows[start + index]['answer']), signature(original['answer'])
            gold_matches += actual[0] == expected[0]
            expression_matches += actual == expected
            if actual != expected:
                mismatches.append(index)
        assert gold_matches == len(official), f'Unexpected ordering or gold answers in {split}'
        assert expression_matches / len(official) > 0.98, 'Combined dataset ordering not sufficiently supported'
        audit[split] = dict(n=len(official), gold_matches=gold_matches,
                            calculation_signature_matches=expression_matches, calculation_mismatch_indices=mismatches)
    selected_ids = random.Random(42).sample(range(len(test)), 500)
    training_questions = {row['question'] for row in train}
    selected = []
    for index in selected_ids:
        row = rows[len(train) + index]
        original = test[index]
        assert original['question'] not in training_questions
        gold = signature(row['answer'])[0]
        assert gold == signature(original['answer'])[0]
        selected.append(dict(question=row['question'], english_question=original['question'],
                             answer=gold, language='ta', source_id=f'gsm8k_test_{index}',
                             dataset_row_index=len(train) + index,
                             calculation_signature_matches=signature(row['answer']) == signature(original['answer'])))
    out = Path('data/vishal_gsm8k_tamil_test500_seed42.jsonl')
    out.write_text(''.join(json.dumps(row, ensure_ascii=True) + '\n' for row in selected))
    manifest = dict(dataset=REPO, revision=REVISION, source_file='gsm8k_tamil.json',
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    sampling='random.Random(42).sample(range(1319), 500)', seed=42,
                    sample_scope='Official GSM8K test portion (combined file rows 7473:8792)',
                    n=500, test_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                    source_ids=[r['source_id'] for r in selected], source_test_indices=selected_ids,
                    training_overlap=0, ordering_audit=audit,
                    selected_calculation_signature_mismatches=sum(not r['calculation_signature_matches'] for r in selected),
                    limitation='Dataset split mapping is inferred from full-file gold and calculation signatures; translation quality can affect performance. Published Tamil questions are retained unchanged.')
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('source_ids','source_test_indices')}, indent=2))

if __name__ == '__main__':
    main()
