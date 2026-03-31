# Copyright (c) 2026 Rui Zhang
# Licensed under the MIT license.

from typing import List, Optional, TypedDict, Sequence


class ChatMessage(TypedDict, total=False):
    role: str
    content: str


ChatPrompt = Sequence[ChatMessage]


class LanguageModel:
    """Base class for language model interface."""

    def chat_completion(
        self,
        message: str | ChatPrompt,
        max_tokens: int,
        timeout_seconds: float,
        *args,
        **kwargs,
    ):
        """
        Send a chat completion query to the language model server.
        Return the response content.

        Args:
            message: The message in str or openai format.
            max_tokens: The maximum number of tokens to generate.
            timeout_seconds: The timeout seconds.
        """
        pass

    def embedding(
        self,
        text: str | List[str],
        dimensions: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        **kwargs,
    ) -> List[float] | List[List[float]]:
        """
        Generate embeddings for the given text(s) using the model specified during initialization.

        Args:
            text: The text or a list of texts to embed.
            dimensions: The number of dimensions for the output embeddings.
            timeout_seconds: The timeout seconds.

        Returns:
            The embedding for the text, or a list of embeddings for the list of texts.
        """
        pass

    def close(self):
        """Release resources (if necessary)."""
        pass

    def reload(self):
        """Reload the language model (if necessary)."""
        pass

    def __del__(self):
        self.close()
