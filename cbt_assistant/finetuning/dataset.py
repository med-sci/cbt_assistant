from datasets import load_dataset, Dataset
from .utils import JsonDatasetParser

class ChatDatasetLoader:
    """
    Loads a JSONL dataset and applies a parser to format the data
    into a text field suitable for language model training.
    """

    def __init__(self, parser: JsonDatasetParser):
        """
        :param parser: An instance of JsonDatasetParser to format chat messages.
        """
        self._parser = parser

    def load_jsonl(self, file_path: str) -> Dataset:
        """
        Loads a JSONL file containing 'messages' lists and applies the parser.

        Example input (JSONL line):
        {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]}

        :param file_path: Path to the JSONL file.
        :return: Hugging Face Dataset with a 'text' column ready for tokenization.
        """
        dataset = load_dataset("json", data_files=file_path, split="train")

        dataset = dataset.map(
            lambda x: {"text": self._parser.parse(x["messages"])},
            remove_columns=dataset.column_names
        )

        return dataset