from cbt_assistant.pipeline import Pipeline
from cbt_assistant.configuration.templates import AppConfig


class GeneratorService:
    def __init__(self):
        config = AppConfig(llm_config=..., rag_config=None, semantic_filter_config=None)
        self._pipeline = Pipeline(config)

    def generate(self, query: str) -> str:
        return self._pipeline.generate(query)