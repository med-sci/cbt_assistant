from enum import Enum
from typing import List, Dict, Optional


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

    def formatted(self) -> str:
        role_map = {
            Role.USER: "User",
            Role.ASSISTANT: "Assistant",
            Role.SYSTEM: "System"
        }
        return role_map[self]


class JsonDatasetParser:
    """
    A class for parsing JSON dataset into a format compatible with the finetuning API.
    """

    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize the parser with an optional system prompt.

        Args:
            system_prompt (Optional[str]): The system prompt to use for the parser.
        """
        self._system_prompt = system_prompt

    def parse(self, messages: List[Dict[str, str]]) -> str:
        """
        Parse a list of messages into a format compatible with the finetuning API.

        Args:
            messages (List[Dict[str, str]]): The list of messages to parse.

        Returns:
            str: The parsed messages.
        """
        parts = []

        if self._system_prompt:
            parts.append(f"System: {self._system_prompt}")

        for msg in messages:
            role = self._parse_role(msg["role"])
            content = msg["content"]
            parts.append(f"{role}: {content}")

        return "\n".join(parts)

    def _parse_role(self, role: str) -> str:
        return Role(role).formatted()
