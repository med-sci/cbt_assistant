import os
import logging
from typing import Optional

from tqdm import tqdm

from ..rag.document_store import TinyDocumentStore
from ..rag.vector_store import FAISSVectorStore
from ..rag.embedder import SentenceTransformersEmbedder
from ..configuration.templates import RAGConfig
from .parser import PDFParser
from ..rag.document_store import Document


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def add_file_to_store(path: str, config: Optional[RAGConfig] = None):
    """
    Parses the given file and adds resulting Documents to the vector and document stores.

    :param path: Path to a supported file (currently only PDF).
    :param config: Optional RAGConfig. If not provided, defaults will be used.
    """

    docs = PDFParser().parse(path)

    if not docs:
        logger.warning(f"No text found in file: {path}")
        return

    logger.info(f"Parsed {len(docs)} documents from {path}")

    embedder = SentenceTransformersEmbedder(config.embed_model)

    with TinyDocumentStore(config.faiss_index_path.replace(".faiss", ".json")) as docstore:
        store = FAISSVectorStore(
            embedder=embedder,
            docstore=docstore,
            index_path=config.faiss_index_path
        )

        texts = []
        metas = []

        for doc in tqdm(docs, desc="Embedding & storing"):
            texts.append(doc.text)
            metas.append(doc.meta)

        store.add_documents(texts, metas)
        store.save(config.faiss_index_path)

    logger.info(f"✅ Ingested {len(docs)} documents from {path} into vector + doc store.")