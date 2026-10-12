"""Translation blocks, boxed answers, and token-level segment boundaries."""
from .constants import OPEN, CLOSE, TRANSLATION_RE

def text_of(completion):
    if isinstance(completion, str):
        return completion
    return "".join(message.get("content", "") for message in completion)


def translation_span(text):
    """Reject empty/multiple/unbalanced translation blocks."""
    if text.count(OPEN) != 1 or text.count(CLOSE) != 1:
        return None
    match = TRANSLATION_RE.search(text)
    return match if match and match.group(1).strip() else None


def last_boxed(text):
    """Extract the last balanced \\boxed{...}, including nested braces."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    start += len("\\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip() or None
    return None


def translation_token_mask(tokenizer, ids):
    """Locate the end of the translation on ACTUAL sampled token IDs.

    Re-encoding is used only to obtain offsets, and only if IDs round-trip exactly.
    Otherwise use prefix decoding. IDs fed to the policy loss are never replaced.
    The wrapper/preamble through the closing tag belongs to the translation segment.
    """
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    span = translation_span(text)
    if span is None:
        return [False] * len(ids)
    boundary = span.end()
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if list(encoded["input_ids"]) == list(ids):
            return [start < boundary and end > start for start, end in encoded["offset_mapping"]]
    except (NotImplementedError, ValueError):
        pass
    # Handles tokenizer normalization/non-round-tripping without guessing IDs.
    for end in range(1, len(ids) + 1):
        prefix = tokenizer.decode(ids[:end], skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False)
        if CLOSE in prefix:
            return [i < end for i in range(len(ids))]
    raise RuntimeError("Decoded translation closing tag has no token boundary")
