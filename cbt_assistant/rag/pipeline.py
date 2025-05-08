from ..configuration.templates import RAGConfig
from ..llm.models import Model
from .embedder import SentenceTransformersEmbedder
from .vector_store import FAISSVectorStore
from .document_store import TinyDocumentStore
from .prompt_formatter import PromptFormatter, PromptStyle
from typing import Optional


class RAGPipeline:
    def __init__(
        self,
        config: RAGConfig,
        model: Model,
        formatter: Optional[PromptFormatter] = None,
    ):
        self._config = config
        self._model = model

        self._embedder = SentenceTransformersEmbedder(config.embed_model)
        self._docstore = TinyDocumentStore()
        self._store = FAISSVectorStore(
            embedder=self._embedder,
            docstore=self._docstore,
            index_path=config.faiss_index_path,
        )

        self._formatter = formatter or PromptFormatter(style=PromptStyle.PLAIN)

    def run(self, query: str) -> str:
        with self._docstore:
            documents = self._store.search(query, k=self._config.top_k)

        context = "\n".join(doc.text for doc in documents if doc is not None)

        prompt = self._formatter.build(context=context, question=query)

        return self._model.generate(prompt)