"""Prompt generation, math verification, and aggregate evaluation metrics."""
from ..constants import SYSTEM
from ..formatting import translation_span, last_boxed
from ..rewards import make_rewards


def evaluate_batches(rows, llm, language, max_new_tokens=1536):
    from vllm import SamplingParams
    from math_verify import parse, verify, LatexExtractionConfig
    results = []
    tokenizer = llm.get_tokenizer()
    prompts = [tokenizer.apply_chat_template([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': row['english_question'] if language == 'en' else row['question']}],
               tokenize=True, return_dict=True, add_generation_prompt=True, enable_thinking=False)['input_ids'] for row in rows]
    assert max(map(len, prompts)) + max_new_tokens <= 4096
    params = SamplingParams(temperature=0, top_p=1, max_tokens=max_new_tokens, seed=42)
    tr_reward, correct_reward = make_rewards()
    for start in range(0, len(rows), 8):
        batch = rows[start:start+8]
        generated = llm.generate([{'prompt_token_ids': p} for p in prompts[start:start+8]], params, use_tqdm=False)
        texts = [g.outputs[0].text for g in generated]
        tr = tr_reward(texts, [r['english_question'] for r in batch])
        cr = correct_reward(texts, [r['answer'] for r in batch])
        for row, text, generation, t, c in zip(batch, texts, generated, tr, cr):
            span = translation_span(text)
            predicted = last_boxed(text[span.end():] if span else text)
            relaxed = False
            if predicted is not None:
                try:
                    conf = [LatexExtractionConfig()]
                    gold = parse('$\\boxed{' + row['answer'] + '}$', extraction_config=conf)
                    pred = parse('$\\boxed{' + predicted + '}$', extraction_config=conf)
                    relaxed = bool(gold and pred and verify(gold, pred))
                except (ValueError, TypeError, TimeoutError):
                    pass
            results.append(dict(row, completion=text, predicted=predicted, correct=bool(c),
                answer_correct_without_format_requirement=relaxed, translation_reward=t,
                valid_translation=span is not None, finish_reason=generation.outputs[0].finish_reason,
                completion_tokens=len(generation.outputs[0].token_ids)))
        yield results


def summarize_results(results):
    n = len(results)
    return dict(evaluated=n, correct=sum(r['correct'] for r in results),
                accuracy=sum(r['correct'] for r in results)/n,
                answer_accuracy_without_format_requirement=sum(r['answer_correct_without_format_requirement'] for r in results)/n,
                translation_reward=sum(r['translation_reward'] for r in results)/n,
                invalid_translation=sum(not r['valid_translation'] for r in results),
                truncated=sum(r['finish_reason']=='length' for r in results),
                extractable_answers=sum(last_boxed(r['completion']) is not None for r in results),
                extractable_answer_fraction=sum(last_boxed(r['completion']) is not None for r in results)/n)
