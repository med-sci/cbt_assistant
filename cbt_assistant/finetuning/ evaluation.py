import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


class PerplexityEvaluator:
    def __init__(self, model_path: str):
        """
        :param model_path: Путь к сохраненной модели (или имя модели в HF Hub).
        """
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self._model.eval()
        if torch.cuda.is_available():
            self._model.to("cuda")

    def compute_perplexity(self, dataset: Dataset) -> float:
        """
        Вычисляет перплексию на предоставленном датасете.
        Датасет должен содержать поле 'text'.

        :param dataset: Hugging Face Dataset.
        :return: Значение перплексии.
        """
        total_loss = 0.0
        total_tokens = 0

        for example in dataset:
            inputs = self._tokenizer(example["text"], return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                total_loss += loss.item() * inputs["input_ids"].size(1)  # Умножаем на число токенов
                total_tokens += inputs["input_ids"].size(1)

        average_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(average_loss)).item()
        return perplexity