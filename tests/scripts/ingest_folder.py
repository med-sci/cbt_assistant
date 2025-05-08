import argparse
import logging
import os

from tqdm import tqdm

from cbt_assistant.configuration.templates import RAGConfig
from cbt_assistant.ingestion.utils import add_file_to_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingest all supported files from folder into RAG store")
    parser.add_argument("--folder", type=str, required=True, help="Path to folder with documents")
    args = parser.parse_args()

    folder_path = args.folder

    config = RAGConfig()

    pdf_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files
    ]

    logger.info(f"📂 Found {len(pdf_files)} PDF files in {folder_path}")

    for file_path in tqdm(pdf_files, desc="Ingesting files"):
        try:
            add_file_to_store(file_path, config=config)
        except Exception as e:
            logger.warning(f"❌ Failed to ingest {file_path}: {e}")


if __name__ == "__main__":
    main()