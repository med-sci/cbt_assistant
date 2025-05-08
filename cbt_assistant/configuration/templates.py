from pydantic import BaseModel, validator
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

    @validator("model_id")
    def validate_model_id(cls, v):
        allowed = [m.value for m in AvailableHuggingFaceModels]

        if v not in allowed:
            raise ValueError(f"Invalid model_id: {v}. Must be one of: {allowed}")
        return v


class RAGConfig(BaseModel):
    faiss_index_path: str
    embed_model: str = SentenceTransformersEmbedderModels.E5_SMALL.value
    top_k: int = 5

    @validator("embed_model")
    def validate_embed_model(cls, v):
        allowed = [m.value for m in SentenceTransformersEmbedderModels]

        if v not in allowed:
            raise ValueError(f"Invalid embed_model: {v}. Must be one of: {allowed}")
        return v


class AppConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    rag: RAGConfig = RAGConfig()