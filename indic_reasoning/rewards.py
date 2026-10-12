"""Translation/correctness rewards and rollout-group diagnostics."""
import math
from .formatting import text_of, translation_span, last_boxed

def correctness_group_metrics(values, group_size):
    """Measure whether each rollout group provides a correctness contrast."""
    if group_size < 2 or not values or len(values) % group_size:
        raise ValueError("Expected complete nonempty correctness rollout groups")
    if any(v not in (0, 1) for v in values):
        raise ValueError("Correctness rewards must be binary")
    counts = [sum(values[i:i + group_size]) for i in range(0, len(values), group_size)]
    n = len(counts)
    return {
        "tapo/correctness_all_wrong_fraction": sum(c == 0 for c in counts) / n,
        "tapo/correctness_all_correct_fraction": sum(c == group_size for c in counts) / n,
        "tapo/correctness_mixed_fraction": sum(0 < c < group_size for c in counts) / n,
        "tapo/correctness_any_correct_fraction": sum(c > 0 for c in counts) / n,
        "tapo/correct_completions_per_group": sum(counts) / n,
    }


def grouped_standardize(values, group_size, eps=1e-4):
    """Population std (explicit choice); zero-variance groups give zero advantage."""
    if group_size < 2 or len(values) % group_size:
        raise ValueError("Rewards must contain complete groups of at least 2 rollouts")
    result = []
    for start in range(0, len(values), group_size):
        group = values[start:start + group_size]
        mean = sum(group) / group_size
        std = math.sqrt(sum((v - mean) ** 2 for v in group) / group_size)
        result.extend((v - mean) / (std + eps) for v in group)
    return result


def make_rewards():
    from sacrebleu.metrics import CHRF
    from math_verify import parse, verify, LatexExtractionConfig
    metric = CHRF(word_order=2)  # ChrF++, not plain character-only ChrF
    extraction = [LatexExtractionConfig()]

    def translation_reward(completions, english_question, **kwargs):
        values = []
        for completion, reference in zip(completions, english_question):
            match = translation_span(text_of(completion))
            values.append(metric.sentence_score(match.group(1).strip(), [reference]).score / 100.0
                          if match else 0.0)
        return values

    def correctness_reward(completions, answer, log_extra=None, **kwargs):
        # TRL supplies one gold answer per sampled completion, already repeated
        # and aligned across rollouts. Its extra columns also reach W&B tables.
        if log_extra is not None:
            log_extra("correct_answer", [str(gold) for gold in answer])
        values = []
        for completion, gold in zip(completions, answer):
            text = text_of(completion)
            span = translation_span(text)
            # The answer MUST be in the reasoning segment, not in the translation.
            predicted = last_boxed(text[span.end():]) if span else None
            if predicted is None:
                values.append(0.0)
                continue
            try:
                gold_expr = parse("$\\boxed{" + str(gold) + "}$", extraction_config=extraction)
                pred_expr = parse("$\\boxed{" + predicted + "}$", extraction_config=extraction)
                values.append(float(bool(gold_expr and pred_expr and verify(gold_expr, pred_expr))))
            except (ValueError, TypeError, TimeoutError):
                values.append(0.0)
        return values
    return translation_reward, correctness_reward
