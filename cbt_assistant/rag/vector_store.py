from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import faiss
from enum import Enum
import os
from .embedder import BaseEmbedder
from ..rag.document_store import BaseDocumentStore, Document


class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, texts: List[str]):
        """
        Add a list of documents to the store.
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int = 5) -> List[str]:
        """
        Search for the top-k closest documents.

        :param query: Query string.
        :param k: Number of nearest documents to return.
        :return: List of document strings.
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save index to disk.
        """
        pass


def ensure_dir_exists(path: str):
    """
    Ensure the parent directory for a file path exists.
    """
    dir_path = os.path.dirname(path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-based vector store with external document storage.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        docstore: BaseDocumentStore,
        index_path: Optional[str] = None,
    ):
        """
        Initialize the vector store.

        :param embedder: Embedding model.
        :param docstore: Document store.
        :param index_path: Path to the FAISS index file.
        """
        self._embedder = embedder
        self._docstore = docstore
        self._next_id = self._docstore.max_id()
        self._index = self._get_index(index_path, self._embedder.embedding_dim())

    def _get_index(self, index_path: Optional[str], dim: int):
        """
        Get or create the FAISS index.

        :param index_path: Path to the FAISS index file.
        :param dim: Dimension of the embedding vectors.
        :return: FAISS index.
        """
        if index_path and os.path.exists(index_path):
            return faiss.read_index(index_path)

        return faiss.IndexFlatL2(dim)

    def add_documents(self, texts: List[str], metas: Optional[List[dict]] = None):
        """
        Add documents to the vector store.

        :param texts: List of document texts.
        :param metas: List of document metadata.
        """
        vectors = self._embedder.embed(texts).astype("float32")
        self._index.add(vectors)

        for i, text in enumerate(texts):
            meta = metas[i] if metas and i < len(metas) else {}
            doc = Document(id=self._next_id, text=text, meta=meta)
            self._docstore.add(doc)
            self._next_id += 1

    def search(self, query: str, k: int = 5) -> List[Document]:
        vec = self._embedder.embed([query]).astype("float32")
        _, indices = self._index.search(vec, k)

        return [
            self._docstore.get(i)
            for i in indices[0]
            if self._docstore.get(i) is not None
        ]

    def save(self, index_path: str):
        ensure_dir_exists(index_path)
        faiss.write_index(self._index, index_path)
        self._docstore.save()


class VectorStoreList(Enum):
    FAISS = FAISSVectorStore
