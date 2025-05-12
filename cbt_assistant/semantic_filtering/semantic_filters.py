from abc import ABC, abstractmethod
from typing import List, Optional
import torch
import torch.nn.functional as F
from cbt_assistant.rag.embedder import BaseEmbedder
from cbt_assistant.semantic_filtering.rejection import RejectionMessageGenerator


class BaseSemanticFilter(ABC):
    @abstractmethod
    def is_relevant(self, query: str) -> bool:
        """
        Check if the given query is relevant.
        :param query: User's question or input
        :return: True if relevant, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def explain(self, query: str) -> str:
        """
        Provide explanation or diagnostic about the relevance decision.
        :param query: User's question or input
        :return: Human-readable explanation string
        """
        raise NotImplementedError


class SimilaritySemanticFilter(BaseSemanticFilter):
    def __init__(self, topics: List[str], embedder: BaseEmbedder, threshold: float = 0.45):
        """
        Initialize the SimilaritySemanticFilter.
        :param topics: List of topics to check relevance against
        :param embedder: Embedding model
        :param threshold: Similarity threshold for relevance
        """
        self._topics = topics
        self._embedder = embedder
        self._threshold = threshold
        self._topic_vectors = torch.tensor(self._embedder.embed(self._topics), dtype=torch.float32)
        self._last_decision: Optional[tuple[float, bool]] = None
        self._rejection_generator = RejectionMessageGenerator()

    def is_relevant(self, query: str) -> bool:
        """
        Check if the given query is relevant to any of the predefined topics.
        :param query: User's question or input
        :return: True if relevant, False otherwise
        """
        query_vector = torch.tensor(self._embedder.embed([query])[0], dtype=torch.float32)
        sims = F.cosine_similarity(query_vector.unsqueeze(0), self._topic_vectors)
        max_score = sims.max().item()
        self._last_decision = (max_score, max_score >= self._threshold)

        return self._last_decision[1]

    def explain(self, query: str) -> str:
        """
        Provide explanation or diagnostic about the relevance decision.
        :param query: User's question or input
        :return: Human-readable explanation string
        """
        if self._last_decision is None:
            raise RuntimeError("is_relevant() must be called before explain()")

        score, decision = self._last_decision

        return (
            f"Query: '{query}'\n"
            f"Max similarity to topics: {score:.2f} (threshold: {self._threshold})\n"
            f"Decision: {'Relevant' if decision else 'Irrelevant'}\n"
            f"Topics checked: {', '.join(self._topics)}"
        )

    def get_rejection_message(self) -> str:
        return self._rejection_generator.generate()
