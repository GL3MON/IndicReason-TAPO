"""Tests for DatasetTranslator module."""

from indic_reasoning.translation.dataset_translator import DatasetTranslator
from indic_reasoning.translation.vllm_translator import VLLMTranslator
from indic_reasoning.translation.prompts import build_gsm8k_prompt


def test_import():
    """Verify DatasetTranslator can be imported."""
    assert DatasetTranslator is not None


def test_initialization():
    """Verify DatasetTranslator can be initialized with a VLLMTranslator and prompt builder."""
    translator = VLLMTranslator(
        model_name="sarvamai/sarvam-translate",
        vllm_url="http://localhost:8000/v1/chat/completions",
    )
    prompt_fn = build_gsm8k_prompt("Hindi")
    dt = DatasetTranslator(translator=translator, system_prompt_builder=prompt_fn)
    assert dt is not None
    assert dt.translator is translator
    assert dt.system_prompt_builder == prompt_fn


def test_system_prompt_builder_returns_string():
    """Verify the prompt builder returns a non-empty string."""
    prompt = build_gsm8k_prompt("Hindi")
    assert isinstance(prompt, str)
    assert len(prompt) > 0
