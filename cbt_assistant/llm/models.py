from abc import ABC, abstractmethod
from typing import Optional
from enum import Enum
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import torch


class Model(ABC):
    def __init__(self, model_id: str, device: Optional[str] = "cuda"):
        self._model_id = model_id
        self._device = device

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        pass


class HuggingFaceModel(Model):
    """
    Universal model runner for HuggingFace causal LLMs.
    """

    def __init__(self, model_id: str, device: str = "cuda", system_prompt: str = ""):
        """
        :param model_id: Model ID or local path.
        :param device: Device to load model on, e.g. "cuda", "cpu", or "auto".
        :param system_prompt: Text to prepend to each prompt.
        """
        super().__init__(model_id, device)
        self._system_prompt = system_prompt
        self._tokenizer = self._load_tokenizer()
        self._model = self._load_model()

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load and return the tokenizer.

        :return: HuggingFace tokenizer
        :rtype: PreTrainedTokenizer
        """
        return AutoTokenizer.from_pretrained(self._model_id)

    def _load_model(self) -> PreTrainedModel:
        """
        Load and return the model.

        :return: HuggingFace causal language model
        :rtype: PreTrainedModel
        """
        return AutoModelForCausalLM.from_pretrained(
            self._model_id,
            device_map=self._device,
            torch_dtype=torch.float32,
            load_in_4bit=True,
        )

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """
        Generate a response from the model.

        :param prompt: Input text
        :type prompt: str
        :param max_tokens: Maximum number of new tokens to generate
        :type max_tokens: int
        :return: Model's textual response
        :rtype: str
        """
        full_prompt = self._system_prompt + prompt

        inputs = self._tokenizer(full_prompt, return_tensors="pt").to(
            self._device
        )
        outputs = self._model.generate(**inputs, max_new_tokens=max_tokens)
        decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

        return decoded[len(full_prompt):].strip()


class HuggingFaceModels(Enum):
    PHI2 = "microsoft/phi-2"
