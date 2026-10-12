"""
dataset_translator.py

Generic utilities for translating a column of a Hugging Face Dataset
using a VLLMTranslator instance.
"""

from typing import List, Callable, Optional
from datasets import Dataset


class DatasetTranslator:
    """
    Translates one or more text columns of a Hugging Face Dataset.

    Parameters
    ----------
    translator: VLLMTranslator
        Instance that knows how to call the vLLM server.
    system_prompt_builder: Callable[[str], str]
        Function that receives a target language name and returns the
        full system prompt (including any examples) to be used for translation.
    """

    def __init__(
        self,
        translator: "VLLMTranslator",
        system_prompt_builder: Callable[[str], str],
    ):
        self.translator = translator
        self.system_prompt_builder = system_prompt_builder

    def _translate_one(self, text: str, target_language: str) -> str:
        """
        Low‑level translation of a single string.
        """
        system_prompt = self.system_prompt_builder(target_language)
        # The system prompt already contains the instruction and example;
        # we only need to send the raw text as the user message.
        return self.translator.translate(system_prompt=system_prompt, user_text=text)

    def translate_column(
        self,
        dataset: Dataset,
        column_name: str,
        target_language: str,
        new_column_name: Optional[str] = None,
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> Dataset:
        """
        Return a new dataset with an additional column containing the translation.

        Parameters
        ----------
        dataset: Dataset
            Input HF dataset.
        column_name: str
            Name of the column to translate.
        target_language: str
            Target language (used to build the system prompt).
        new_column_name: str, optional
            Name for the translated column. If None, defaults to
            f"{column_name}_{target_language.lower()}".
        batch_size: int
            Number of examples to process before printing a progress update.
        show_progress: bool
            Whether to print progress to stdout.

        Returns
        -------
        Dataset
            The dataset with the translated column added.
        """
        if column_name not in dataset.column_names:
            raise ValueError(f"Column '{column_name}' not found in dataset.")

        if new_column_name is None:
            new_column_name = f"{column_name}_{target_language.lower()}"

        total = len(dataset)
        translated: List[str] = []

        for i, example in enumerate(dataset):
            text = example[column_name]
            # Guard against non‑string entries (e.g., None)
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            translated.append(self._translate_one(text, target_language))

            if show_progress and (i + 1) % batch_size == 0:
                print(f"  translated {i + 1}/{total} examples")

        if show_progress:
            print(f"Finished translating {total} examples.")

        # Add the new column and return a new dataset object
        return dataset.add_column(new_column_name, translated)

    def translate_multiple_columns(
        self,
        dataset: Dataset,
        columns: List[str],
        target_language: str,
        batch_size: int = 8,
    ) -> Dataset:
        """
        Convenience wrapper to translate several columns sequentially.
        """
        for col in columns:
            dataset = self.translate_column(
                dataset=dataset,
                column_name=col,
                target_language=target_language,
                batch_size=batch_size,
            )
        return dataset