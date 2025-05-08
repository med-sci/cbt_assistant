import logging
from cbt_assistant.llm.models import HuggingFaceModel, HuggingFaceModels

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    model_id = HuggingFaceModels.PHI2.value
    logging.info(f"Starting model: {model_id}")

    try:
        model = HuggingFaceModel(model_id=model_id, system_prompt="")
        logging.info("Model loaded.")

        prompt = "What is cognitive behavioral therapy?"

        logging.info(f"Generating response for prompt: {prompt}")
        response = model.generate(prompt, max_tokens=200)

        print(f"\nPrompt:\n{prompt}\n")
        print(f"Response:\n{response}\n")
        logging.info("Generation complete.")

    except Exception as e:
        logging.error(f"Error during generation: {e}", exc_info=True)


if __name__ == "__main__":
    main()
