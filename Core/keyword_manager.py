import pickle
import re
import os
import unicodedata
from pathlib import Path
from rank_bm25 import BM25Okapi
import logging


# Setup logger
logger = logging.getLogger(__name__)


class KeywordManager:
    def __init__(self, dataset_name):
        self._dataset_name = dataset_name

    def save_index(self, tokenized_chunks, processsed_data_dir, processed_corpus_id):
        """
        Creates, loads, or caches a BM25 keyword index for the dataset.

        Args:
            tokenized_chunks (list): Tokenized text chunks for BM25.
            processed_corpus_id (str): Unique identifier for the processed dataset.

        Returns:
            BM25Okapi: Precomputed BM25 index.
        """
        bm25 = BM25Okapi(tokenized_chunks)
        index_dir = processsed_data_dir / Path(processed_corpus_id)
        index_path = index_dir / "bm25_index.pkl"

        # Ensure the directory exists
        os.makedirs(index_dir, exist_ok=True)

        with open(index_path, "wb") as f:
            pickle.dump(bm25, f)
            logger.info(f"Saved BM25 index for {self._dataset_name} to {index_path}")

        return bm25

