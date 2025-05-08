from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
from tinydb import TinyDB, Query
from typing import Optional
from enum import Enum


class Document(BaseModel):
    id: int
    text: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class DocumentRecord(Enum):
    id = "id"
    text = "text"
    meta = "meta"

class MetaField(Enum):
    SOURCE = "source"
    PAGE = "page"
    CHUNK = "chunk"
    TYPE = "type"
    TOKENS = "tokens"


class BaseDocumentStore(ABC):
    @abstractmethod
    def __enter__(self):
        """Context manager entry"""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        pass

    @abstractmethod
    def add(self, doc_id: int, text: str, meta: Optional[Dict[str, Any]] = None):
        pass

    @abstractmethod
    def get(self, doc_id: int) -> Optional[str]:
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def max_id(self) -> int:
        pass


class TinyDocumentStore(BaseDocumentStore):
    """
    A simple document store that uses TinyDB to store documents.
    """

    TABLE_NAME = "documents"

    def __init__(self, path: str):
        """
        Initialize the document store.

        :param path: Path to the TinyDB database file.
        """
        self._path = path

    def __enter__(self):
        """
        Context manager entry.
        """
        self._db = TinyDB(self._path)
        self._table = self._db.table(self.TABLE_NAME)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()

    def add(self, doc: Document):
        """
        Add a document to the store.
        """
        self._table.insert(doc.model_dump())

    def get(self, doc_id: int) -> Optional[str]:
        """
        Retrieve text by doc_id.
        """
        q = Query()
        result = self._table.get(q.id == doc_id)
        return result[DocumentRecord.text.value] if result else None

    def max_id(self) -> int:
        """
        Get the maximum document ID.
        """
        all_docs = self._table.all()

        if not all_docs:
            return 0

        return max(doc["id"] for doc in all_docs) + 1

    def save(self):
        self._db.close()


class DocumentStoreList(Enum):
    TINY = TinyDocumentStore
