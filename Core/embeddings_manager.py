import time
import os
from logger import logger
from annoy import AnnoyIndex
from pathlib import Path
from tqdm import tqdm
from factories.embedding_model_factory import EmbeddingModelFactory

PATH_TO_EMBEDDINGS_BASE = Path(
    "C:\\Users\\Djhay\\OneDrive\\Desktop\\Projects\\Hackathon\\Hackathon\\ProcessedData"
)

class EmbeddingManager:
    def __init__(self):
        self._annoy_index = self._set_up_annoy()
        emf = EmbeddingModelFactory()
        self._model = emf.get_model("all-MiniLM-L6-v2")

    def _set_up_annoy(self):
        embedding_dim = 384  # TODO: Take this out
        metric = 'angular'  # TODO: Take this out
        annoy_index = AnnoyIndex(embedding_dim, metric)
        return annoy_index

    def save_embeddings(self, processed_data_dir):
        """
        Saves Annoy index to disk, creating the directory if it does not exist.
        """
        embedding_path = processed_data_dir / "embeddings.ann"

        n_trees = 10  # TODO: Pass in dynamically
        self._annoy_index.build(n_trees=n_trees)
        self._annoy_index.save(str(embedding_path))

        return str(embedding_path)


    def generate_and_store_embedding(self, id, split):
        embedding = self._model.encode(split)
        self._annoy_index.add_item(id, embedding)
