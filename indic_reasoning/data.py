"""Training-data alignment and context-budget checks."""
import re
from .constants import LANGUAGE_NAMES

def normalize_question(question):
    return " ".join(question.split())


def parse_mathoctopus_prompt(prompt):
    language = re.search(r"Please answer in ([A-Za-z]+)\.", prompt)
    question = re.search(r"### Instruction:\s*(.*?)\s*### Response:\s*$", prompt, re.S)
    if not language or not question:
        raise ValueError("Unexpected MathOctopus prompt format; refusing to guess alignment")
    return language.group(1), question.group(1).strip()


def prepare_rows(raw_rows, original_gsm8k, languages):
    """Validate original release block order and join gold answers to GSM8K train.

    The release orders each English question before its translations. We check
    every English question against the official train split and reject duplicate
    languages within each block. Synthetic `chosen`/`reject` are never rewards.
    """
    gold_by_question = {
        normalize_question(row["question"]): (i, row["answer"].rsplit("####", 1)[-1].strip())
        for i, row in enumerate(original_gsm8k)
    }
    records, current, seen = [], None, set()
    inverse = {name: code for code, name in LANGUAGE_NAMES.items()}
    # Actual release has ONE stray Telugu row. Recognize it while parsing blocks,
    # but do not advertise it as a Telugu training corpus (use local JSONL instead).
    inverse["Telugu"] = "te"
    for row in raw_rows:
        name, question = parse_mathoctopus_prompt(row["prompt"])
        if name not in inverse:
            raise ValueError(f"Unknown language in release: {name}")
        code = inverse[name]
        if code == "en":
            lookup = gold_by_question.get(normalize_question(question))
            if lookup is None:
                raise ValueError("Release English question does not match GSM8K TRAIN")
            source_index, answer = lookup
            current = (question, answer, f"gsm8k_train_{source_index}")
            seen = {"en"}
        else:
            if current is None or code in seen:
                raise ValueError("Release block ordering changed; refusing unsafe question alignment")
            seen.add(code)
        if code in languages:
            english_question, answer, source_id = current
            records.append(dict(question=question, english_question=english_question,
                                answer=answer, language=code, source_id=source_id))
    if not records:
        raise ValueError("No selected-language rows found")
    missing = set(languages) - {r["language"] for r in records}
    if missing:
        raise ValueError(f"Languages unavailable in released training data: {sorted(missing)}")
    return records


def filter_context_rows(dataset, tokenizer, context_length, completion_length, chat_template_kwargs=None):
    """Keep complete chat prompts with room for the full generation budget."""
    budget = context_length - completion_length
    if budget < 1:
        raise ValueError("Context length must exceed the completion budget")
    before = len(dataset)

    def fits(batch):
        tokenized = tokenizer.apply_chat_template(
            batch["prompt"], add_generation_prompt=True, tokenize=True, return_dict=True,
            **(chat_template_kwargs or {}))
        return [len(ids) <= budget for ids in tokenized["input_ids"]]

    filtered = dataset.filter(fits, batched=True, batch_size=128,
                              desc=f"Filtering chat prompts to {budget} tokens")
    print(f"Context filter: kept {len(filtered)}/{before} rows; skipped {before - len(filtered)} "
          f"overlength prompts (prompt budget={budget}, completion budget={completion_length}, "
          f"context={context_length}).", flush=True)
    if not len(filtered):
        raise ValueError("No rows fit the context budget; increase vllm_max_model_length "
                         "or reduce max_completion_length")
    return filtered
