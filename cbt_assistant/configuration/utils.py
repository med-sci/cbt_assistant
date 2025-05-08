import yaml
from .templates import AppConfig


def load_config(path: str) -> AppConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return AppConfig(**data)