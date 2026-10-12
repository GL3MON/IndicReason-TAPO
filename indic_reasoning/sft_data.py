"""Official worked-solution joins, clean validation splits, and SFT tokenization."""
import random
import re
from .constants import OPEN, CLOSE, SYSTEM
from .data import normalize_question

def prepare_sft_rows(records, official_train):
    """Join only original GSM8K TRAIN rationales; verify question and final answer."""
    rows, seen = [], set()
    for row in records:
        if row.get("language") != "ta":
            continue
        source_id = row["source_id"]
        match = re.fullmatch(r"gsm8k_train_(\d+)", source_id)
        if match is None or source_id in seen:
            raise ValueError(f"Expected unique GSM8K train source IDs: {source_id}")
        seen.add(source_id)
        index = int(match.group(1))
        if index >= len(official_train):
            raise ValueError(f"Train source index out of bounds: {source_id}")
        source = official_train[index]
        if normalize_question(row["english_question"]) != normalize_question(source["question"]):
            raise ValueError(f"English question mismatch: {source_id}")
        rationale, separator, gold = source["answer"].rpartition("####")
        if not separator or not rationale.strip():
            raise ValueError(f"Missing official worked solution: {source_id}")
        if str(row["answer"]).strip().replace(",", "") != gold.strip().replace(",", ""):
            raise ValueError(f"Gold answer mismatch: {source_id}")
        # Remove calculator annotations while retaining the worked arithmetic.
        rationale = re.sub(r"<<[^<>]*>>", "", rationale).strip()
        gold = str(row["answer"]).strip()
        completion = f"{OPEN}\n{row['english_question']}\n{CLOSE}\n{rationale}\n\\boxed{{{gold}}}"
        rows.append({**row, "completion": completion})
    if not rows:
        raise ValueError("No matched Tamil train examples")
    return rows


def split_sft_rows(rows, train_count, eval_count, seed, exclude_eval_ids=()):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    if len(rows) < train_count + eval_count:
        raise ValueError("Not enough unique training questions for disjoint SFT train/eval subsets")
    excluded = set(exclude_eval_ids)
    validation = [r for r in rows if r["source_id"] not in excluded][:eval_count]
    if len(validation) != eval_count:
        raise ValueError("Not enough unseen examples for validation")
    eval_ids = {r["source_id"] for r in validation}
    training = [r for r in rows if r["source_id"] not in eval_ids][:train_count]
    return training, validation


def tokenize_sft_row(row, tokenizer, max_length):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": row["question"]}]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True,
                                            add_generation_prompt=True, enable_thinking=False)
    prompt_ids = encoded["input_ids"]
    completion_ids = tokenizer.encode(row["completion"] + tokenizer.eos_token, add_special_tokens=False)
    if len(prompt_ids) + len(completion_ids) > max_length:
        return None  # Never truncate the gold answer or a worked solution.
    if not completion_ids or completion_ids[-1] != tokenizer.eos_token_id:
        raise ValueError("SFT target must end with the model EOS")
    ids = prompt_ids + completion_ids
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(prompt_ids) + completion_ids}
