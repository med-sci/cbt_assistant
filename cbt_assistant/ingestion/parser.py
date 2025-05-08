from abc import ABC, abstractmethod
import fitz
from typing import List
from ..rag.document_store import Document, MetaField
from .tokenizer import TokenizerWrapper


class BaseParser(ABC):
    @abstractmethod
    def parse(self, path: str, source: str = "") -> List[Document]:
        """
        Parse a file and return a list of Documents.

        :param path: Path to the file
        :param source: Optional string identifying the origin
        :return: List of Document objects
        """
        pass


class PDFParser(BaseParser):
    """
    PDF parser that extracts text page-by-page and splits into token-aware chunks.
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 40, encoding: str = "cl100k_base"):
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._tokenizer = TokenizerWrapper(encoding)

    def parse(self, path: str, source: str = "") -> List[Document]:
        doc = fitz.open(path)
        documents = []
        doc_id = 0

        for i, page in enumerate(doc):
            full_text = page.get_text().strip()
            if not full_text:
                continue

            chunks = self._split_into_token_chunks(full_text)

            for j, chunk in enumerate(chunks):
                token_count = self._tokenizer.count_tokens(chunk)

                documents.append(
                    Document(
                        id=doc_id,
                        text=chunk,
                        meta={
                            MetaField.SOURCE: source or path,
                            MetaField.PAGE: i + 1,
                            MetaField.CHUNK: j + 1,
                            MetaField.TYPE: "pdf",
                            MetaField.TOKENS: token_count,
                        }
                    )
                )
                doc_id += 1

        return documents

    def _split_into_token_chunks(self, text: str) -> List[str]:
        tokens = self._tokenizer.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = start + self._chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self._tokenizer.decode(chunk_tokens).strip()

            if self._tokenizer.count_tokens(chunk_text) > 10:
                chunks.append(chunk_text)

            start += self._chunk_size - self._overlap

        return chunks