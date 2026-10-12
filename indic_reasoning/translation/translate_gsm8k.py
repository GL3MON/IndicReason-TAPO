"""
translate_gsm8k.py

Entry point for translating the GSM8K dataset (or any text column)
using a vLLM-hosted sarvam-translate model.

Usage:
    python translate_gsm8k.py --config config_gsm8k.yaml
"""

import argparse
import os
import yaml
from datasets import load_dataset

from .vllm_translator import VLLMTranslator
from .dataset_translator import DatasetTranslator
from .prompts import build_gsm8k_prompt


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Translate GSM8K dataset using vLLM-hosted sarvam-translate"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # -------------------------
    # vLLM translator setup
    # -------------------------
    vllm_translator = VLLMTranslator(
        model_name=config.get("model_name", "sarvamai/sarvam-translate"),
        vllm_url=config.get("vllm_url", "http://localhost:8000/v1/chat/completions"),
        default_params={
            "max_tokens": config.get("max_tokens", 1024),
            "temperature": config.get("temperature", 0.01),
            "top_p": config.get("top_p", 0.95),
            "n": config.get("n", 1),
        },
    )

    # -------------------------
    # Dataset translator setup
    # -------------------------
    dataset_translator = DatasetTranslator(
        translator=vllm_translator,
        system_prompt_builder=build_gsm8k_prompt,
    )

    # -------------------------
    # Load dataset
    # -------------------------
    dataset_name = config.get("dataset_name", "gsm8k")
    dataset_config = config.get("dataset_config")  # may be None
    split = config.get("split", "train")
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    text_column = config.get("text_column", "question")
    target_language = config.get("target_language", "Hindi")
    batch_size = config.get("batch_size", 8)
    output_dir = config.get("output_dir", f"./translated_gsm8k_{target_language.lower()}")

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Translate
    # -------------------------
    print(f"Translating column '{text_column}' → {target_language}")
    translated_dataset = dataset_translator.translate_column(
        dataset=dataset,
        column_name=text_column,
        target_language=target_language,
        batch_size=batch_size,
    )

    # -------------------------
    # Save
    # -------------------------
    translated_dataset.save_to_disk(output_dir)
    print(f"Translated dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()