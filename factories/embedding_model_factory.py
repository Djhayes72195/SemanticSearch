from typing import Callable
from sentence_transformers import SentenceTransformer

class EmbeddingModelFactory:
    def __init__(self):
        self._registry: dict[str, Callable[[], SentenceTransformer]] = {
            "all-MiniLM-L6-v2": lambda: SentenceTransformer("all-MiniLM-L6-v2"),
        }

    def get_model(self, model_name: str) -> SentenceTransformer:
        if model_name not in self._registry:
            raise ValueError(f"Unsupported model: {model_name}")
        return self._registry[model_name]()
