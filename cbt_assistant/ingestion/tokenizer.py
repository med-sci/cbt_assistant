import tiktoken
from typing import List


class TokenizerWrapper:
    """
    A simple wrapper for tiktoken tokenizer interface.
    """

    def __init__(self, encoding: str = "cl100k_base"):
        """
        :param encoding: Encoding name supported by tiktoken (e.g. 'gpt2', 'cl100k_base')
        """
        self._enc = tiktoken.get_encoding(encoding)

    def encode(self, text: str) -> List[int]:
        return self._enc.encode(text)

    def decode(self, tokens: List[int]) -> str:
        return self._enc.decode(tokens)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))