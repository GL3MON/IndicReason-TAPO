"""Tests for VLLMTranslator module."""

from indic_reasoning.translation.vllm_translator import VLLMTranslator


def test_import():
    """Verify VLLMTranslator can be imported."""
    assert VLLMTranslator is not None


def test_default_params():
    """Verify default parameters are set."""
    t = VLLMTranslator(
        model_name="sarvamai/sarvam-translate",
        vllm_url="http://localhost:8000/v1/chat/completions",
    )
    assert t.default_params["max_tokens"] == 1024
    assert t.default_params["temperature"] == 0.01
    assert t.default_params["top_p"] == 0.95
    assert t.default_params["n"] == 1


def test_translate_method_exists():
    """Verify translate method exists and is callable."""
    t = VLLMTranslator(
        model_name="sarvamai/sarvam-translate",
        vllm_url="http://localhost:8000/v1/chat/completions",
    )
    assert callable(t.translate)
