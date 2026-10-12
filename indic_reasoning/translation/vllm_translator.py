"""
vllm_translator.py

Handles communication with a vLLM server hosting a chat-style model.
"""

import json
import requests
from typing import Dict, Any, Optional


class VLLMTranslator:
    """
    Thin wrapper around the vLLM OpenAI-compatible chat completions endpoint.
    """

    def __init__(
        self,
        model_name: str,
        vllm_url: str = "http://localhost:8000/v1/chat/completions",
        default_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Parameters
        ----------
        model_name: str
            Model identifier as known to the vLLM server.
        vllm_url: str
            Full URL to the chat/completions endpoint.
        default_params: dict, optional
            Default generation parameters (max_tokens, temperature, top_p, n).
        """
        self.model_name = model_name
        self.vllm_url = vllm_url
        self.default_params = default_params or {
            "max_tokens": 1024,
            "temperature": 0.01,
            "top_p": 0.95,
            "n": 1,
        }

    def translate(
        self,
        system_prompt: str,
        user_text: str,
        **override_params,
    ) -> str:
        """
        Send a single chat completion request and return the assistant's reply.

        Parameters
        ----------
        system_prompt: str
            System-level instruction.
        user_text: str
            The content to translate (or any user message).
        **override_params:
            Generation parameters to override the defaults.

        Returns
        -------
        str
            The model's generated text.
        """
        params = {**self.default_params, **override_params}
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            **params,
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(self.vllm_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        # Assuming a single choice
        return result["choices"][0]["message"]["content"].strip()