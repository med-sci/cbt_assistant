from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import Dataset
from cbt_assistant.config.fine_tuning import FineTuningConfig


class StudentTrainer:
    def __init__(self, config: FineTuningConfig):
        """
        :param config: Training settings, including model, dataset, and hyperparameters.
        """
        self._config = config
        self._tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(config.model_id, trust_remote_code=True)

    def train(self, dataset: Dataset):
        """
        Starts training on the provided dataset.
        """
        args = TrainingArguments(
            output_dir=self._config.output_dir,
            num_train_epochs=self._config.epochs,
            per_device_train_batch_size=self._config.batch_size,
            learning_rate=self._config.learning_rate,
            logging_steps=self._config.logging_steps,
            save_steps=self._config.save_steps,
            save_total_limit=self._config.save_total_limit,
            report_to="none"
        )

        trainer = SFTTrainer(
            model=self._model,
            train_dataset=dataset,
            tokenizer=self._tokenizer,
            args=args
        )

        trainer.train()

        # Save the model and tokenizer
        trainer.model.save_pretrained(self._config.output_dir)
        trainer.tokenizer.save_pretrained(self._config.output_dir)
