"""Tests for prompts module."""

from indic_reasoning.translation.prompts import build_gsm8k_prompt


def test_import():
    """Verify build_gsm8k_prompt can be imported."""
    assert build_gsm8k_prompt is not None


def test_returns_string():
    """Verify the function returns a string."""
    prompt = build_gsm8k_prompt("Hindi")
    assert isinstance(prompt, str)


def test_contains_key_elements():
    """Verify prompt contains expected keywords."""
    prompt = build_gsm8k_prompt("Hindi")
    keywords = ["numbers", "mathematical symbols", "units", "logical structure", "natural Hindi"]
    for kw in keywords:
        assert kw.lower() in prompt.lower(), f"Missing keyword: {kw}"
