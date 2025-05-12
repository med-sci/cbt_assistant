from typing import Optional
from .configuration.templates import AppConfig
from .llm.models import HuggingFaceModel
from .rag.pipeline import RAGPipeline
from .semantic_filtering.semantic_filters import SimilaritySemanticFilter

class Pipeline:
    """
    Main pipeline class that orchestrates the generation of responses.
    """

    def __init__(self, config: AppConfig):
        """
        Initialize the pipeline with the given configuration.
        :param config: App configuration.
        """
        self._config = config
        self._llm = HuggingFaceModel(config.model)
        self._rag_pipeline = self._create_rag_pipeline()
        self._semantic_filter = self._create_semantic_filter()

    def _create_rag_pipeline(self) -> Optional[RAGPipeline]:
        if self._config.rag is not None:
            return RAGPipeline(
                config=self._config.rag,
                model=self._llm,
            )
        return None

    def _create_semantic_filter(self) -> Optional[SimilaritySemanticFilter]:
        if self._config.semantic_filter is not None:
            return SimilaritySemanticFilter(self._config.semantic_filter)
        return None

    def generate(self, query: str) -> str:
        if self._semantic_filter and not self._semantic_filter.is_relevant(query):
            return self._semantic_filter.get_rejection_message()

        if self._rag_pipeline:
            return self._llm.generate(query, self._rag_pipeline.run(query))

        return self._llm.generate(query)