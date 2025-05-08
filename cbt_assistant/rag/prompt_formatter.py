from typing import Optional
from enum import Enum


class PromptStyle(str, Enum):
    PLAIN = "plain"


class PromptFormatter:
    def __init__(self, style: PromptStyle = PromptStyle.PLAIN):
        self._style = style

    def build(self, context: str, question: str) -> str:
        context = context.strip()
        question = question.strip()

        if self._style == PromptStyle.PLAIN:
            return (
                "Use the context below to answer the question.\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{question}\n\n"
                "Answer:"
            )

        raise ValueError(f"Unknown prompt style: {self._style}")