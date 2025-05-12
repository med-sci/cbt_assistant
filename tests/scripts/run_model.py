import argparse
import logging
from cbt_assistant.configuration.templates import AppConfig
from cbt_assistant.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Ask a question using the Pipeline.")
    parser.add_argument("--query", type=str, required=True, help="Your question")
    parser.add_argument("--config", type=str, default="app_config.yaml", help="Path to the configuration file")
    args = parser.parse_args()

    config = AppConfig(args.config)
    pipeline = Pipeline(config)

    response = pipeline.run(args.query)
    print(response)


if __name__ == "__main__":
    main()