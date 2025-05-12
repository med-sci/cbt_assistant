from pydantic import BaseModel, validator
from typing import Optional
from enum import Enum
from ..llm.models import AvailableHuggingFaceModels
from ..rag.embedder import SentenceTransformersEmbedderModels


class Device(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"


class ModelConfig(BaseModel):
    device: Device
    model_id: str = AvailableHuggingFaceModels.PHI2.value
    system_prompt: str = ""
    max_tokens: int = 200
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None

    @validator("model_id")
    def validate_model_id(cls, v):
        allowed = [m.value for m in AvailableHuggingFaceModels]

        if v not in allowed:
            raise ValueError(f"Invalid model_id: {v}. Must be one of: {allowed}")
        return v


class RAGConfig(BaseModel):
    faiss_index_path: str
    document_store_path: str
    embed_model: str = SentenceTransformersEmbedderModels.E5_SMALL.value
    top_k: int = 5

    @validator("embed_model")
    def validate_embed_model(cls, v):
        allowed = [m.value for m in SentenceTransformersEmbedderModels]

        if v not in allowed:
            raise ValueError(f"Invalid embed_model: {v}. Must be one of: {allowed}")
        return v


class SemanticFilterConfig(BaseModel):
    threshold: float = 0.45
    topics_path: str = "data/cbt_topics.txt"


class AppConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    rag: Optional[RAGConfig] = None
    semantic_filter: Optional[SemanticFilterConfig] = None
