from typing import Optional
from enum import Enum


class PromptStyle(str, Enum):
    PLAIN = "plain"
    CHAT_INSTRUCT = "chat_instruct"
    CHAT_FORMATTED = "chat_formatted"


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

        elif self._style == PromptStyle.CHAT_INSTRUCT:
            return (
                f"Context:\n{context}\n\n"
                f"User: {question}\n"
                "Assistant:"
            )

        elif self._style == PromptStyle.CHAT_FORMATTED:
            return (
                f"<|context|>\n{context}\n\n"
                f"<|user|>\n{question}\n\n"
                "<|assistant|>\n"
            )

        else:
            raise ValueError(f"Unknown prompt style: {self._style}")