import argparse
import logging

from cbt_assistant.configuration.templates import RAGConfig
from cbt_assistant.llm.models import HuggingFaceModel, AvailableHuggingFaceModels
from cbt_assistant.rag.pipeline import RAGPipeline
from cbt_assistant.rag.prompt_formatter import PromptFormatter
from cbt_assistant.rag.prompt_formatter import PromptStyle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    parser = argparse.ArgumentParser(description="Ask a question using RAG pipeline.")
    parser.add_argument("--query", type=str, required=True, help="Your input question")
    args = parser.parse_args()
    prompt = args.query

    model_id = AvailableHuggingFaceModels.PHI2.value
    logging.info(f"Starting model: {model_id}")

    try:
        config = RAGConfig()
        model = HuggingFaceModel(model_id=model_id, system_prompt="")
        formatter = PromptFormatter(style=PromptStyle.PLAIN)

        pipeline = RAGPipeline(config=config, model=model, formatter=formatter)

        logging.info(f"Running RAG pipeline for query: {prompt}")
        response = pipeline.run(prompt)

        print(f"\nPrompt:\n{prompt}\n")
        print(f"Response:\n{response}\n")
        logging.info("RAG response generation complete.")

    except Exception as e:
        logging.error(f"Error during RAG pipeline execution: {e}", exc_info=True)