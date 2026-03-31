# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

import logging
import os
from typing import List, Optional, Dict, Any

import openai.types.chat

from algodisco.base.llm import LanguageModel

# Try to import anthropic, if not available, we will raise an error when the class is used.
try:
    import anthropic
except ImportError:
    anthropic = None

logging.getLogger("httpx").setLevel(logging.WARNING)


class ClaudeAPI(LanguageModel):
    def __init__(
        self,
        model: str,
        base_url: str = None,
        api_key: str = None,
        **anthropic_init_kwargs,
    ):
        super().__init__()
        if anthropic is None:
            raise ImportError(
                "The 'anthropic' package is required for ClaudeAPI. Please install it with 'pip install anthropic'."
            )

        # If base_url is set to None, find 'ANTHROPIC_BASE_URL' in environment variables
        if base_url is None:
            base_url = os.environ.get("ANTHROPIC_BASE_URL")

        # If api_key is set to None, find 'ANTHROPIC_API_KEY' in environment variables
        if api_key is None:
            if "ANTHROPIC_API_KEY" not in os.environ:
                raise RuntimeError(
                    'If "api_key" is None, ANTHROPIC_API_KEY must be set.'
                )
            else:
                api_key = os.environ["ANTHROPIC_API_KEY"]

        self._model = model
        self._client = anthropic.Anthropic(
            api_key=api_key, base_url=base_url, **anthropic_init_kwargs
        )

    def chat_completion(
        self,
        message: str | List[openai.types.chat.ChatCompletionMessageParam],
        max_tokens: Optional[int] = 1024,
        timeout_seconds: Optional[float] = None,
        *args,
        **kwargs,
    ) -> str:
        """Send a chat completion query to the Claude API.
        Return the response content.

        Args:
            message: The message in str or Anthropic/OpenAI format.
            max_tokens: The maximum number of tokens to generate.
            timeout_seconds: The timeout seconds.
        """
        if isinstance(message, str):
            messages = [{"role": "user", "content": message.strip()}]
        else:
            messages = message

        # Claude requires 'system' to be a top-level parameter, not in messages
        system_msg = ""
        actual_messages = []
        for msg in messages:
            # Handle potential dict or object access
            role = (
                msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            )
            content = (
                msg.get("content")
                if isinstance(msg, dict)
                else getattr(msg, "content", None)
            )

            if role == "system":
                if system_msg:
                    system_msg += "\n" + content
                else:
                    system_msg = content
            else:
                actual_messages.append({"role": role, "content": content})

        # Prepare parameters
        completion_kwargs = {
            "model": self._model,
            "messages": actual_messages,
            "max_tokens": max_tokens if max_tokens is not None else 1024,
        }
        if timeout_seconds is not None:
            completion_kwargs["timeout"] = timeout_seconds

        if system_msg:
            completion_kwargs["system"] = system_msg

        # Update with any additional kwargs
        completion_kwargs.update(kwargs)

        response = self._client.messages.create(**completion_kwargs)
        # Assuming we want the text content of the first content block
        return response.content[0].text

    def embedding(
        self,
        text: str | List[str],
        dimensions: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        **kwargs,
    ) -> List[float] | List[List[float]]:
        """Claude currently does not have a native embedding API.
        This method will raise a NotImplementedError.
        """
        raise NotImplementedError("ClaudeAPI does not support embeddings.")
