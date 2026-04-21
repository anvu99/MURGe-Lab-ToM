"""
Abstract base class for all memory architectures.
Each concrete memory class defines its own internal entry format.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from configs.configs import Conversation


class BaseMemory(ABC):
    """
    Base class for agent memory.

    Each memory architecture (episodic, semantic, ToM, etc.) subclasses this
    and defines its own internal storage format. The two core operations are
    update (write) and retrieve (read).
    """

    def __init__(self, name: str):
        """
        Args:
            name: Human-readable name for this memory type
                  (e.g., "EpisodicMemory", "ToMMemory").
        """
        self.name = name

    @abstractmethod
    def update_memory(
        self,
        conversation: Conversation,
        result: Any = None,
        **kwargs,
    ) -> None:
        """
        Process the conversation history and store relevant information.

        Each memory architecture decides *what* to extract and *how* to store it.

        Args:
            conversation: The full conversation so far (list of RoundEntry).
            result: The outcome of the debate (e.g., correct answer, score).
            **kwargs: Additional architecture-specific arguments.
        """
        ...

    @abstractmethod
    def retrieve_memory(self, query: Optional[str] = None, **kwargs) -> str:
        """
        Retrieve relevant memories and return them as a formatted string
        ready for prompt injection.

        Args:
            query: The retrieval query (e.g., the current question or context).
            **kwargs: Additional architecture-specific arguments.

        Returns:
            A string to be injected into the agent's prompt.
        """
        ...

    def get_instruction(self) -> str:
        """
        Return an explicit prompt instruction telling the agent how to use this memory.
        Defaults to an empty string for memories that don't need explicit instructions.
        """
        return ""

    @abstractmethod
    def clear(self) -> None:
        """Reset all stored memories."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
