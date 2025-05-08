from enum import Enum
from sentence_transformers import SentenceTransformer
from typing import List, Union
from ..configuration.templates import RAGConfig
import numpy as np
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Compute embedding(s) for input text(s).

        :param text: One or more texts.
        :type text: Union[str, List[str]]
        :return: Embedding(s) as np.ndarray
        :rtype: np.ndarray
        """
        raise NotImplementedError

    @abstractmethod
    def embedding_dim(self) -> int:
        """
        Return dimensionality of embeddings.
        """
        raise NotImplementedError


class SentenceTransformersEmbedder(BaseEmbedder):
    """
    Wrapper around a sentence-transformers embedding model.
    """

    def __init__(self, model_name: str):
        """
        :param model_name: Name of the sentence-transformers model to use.
        :type model_name: SentenceTransformersEmbedderModels
        """
        self._model = SentenceTransformer(model_name)

    def embed(self, text: List[str]) -> np.ndarray:
        """
        Generate embeddings for one or more texts.

        :param text: A list of strings.
        :type text: List[str]
        :return: Embedding(s) as a numpy array.
        :rtype: np.ndarray
        """
        return np.array(self._model.encode(text, normalize_embeddings=True))

    def embedding_dim(self) -> int:
        """
        Return dimensionality of embeddings.
        """
        return self._model.get_sentence_embedding_dimension()


class SentenceTransformersEmbedderModels(str, Enum):
    E5_SMALL = "intfloat/e5-small-v2"
    E5_BASE = "intfloat/e5-base-v2"