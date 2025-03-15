import pickle
import re
import os
import unicodedata
import json
from pathlib import Path
from rank_bm25 import BM25Okapi
import logging
from Core.tokenizer import Tokenizer

import unicodedata
import re
import nltk
from nltk.corpus import stopwords


# Setup logger
logger = logging.getLogger(__name__)


class KeywordManager:
    def __init__(self, dataset_name):
        self._dataset_name = dataset_name

    def save_index(self, tokenized_chunks, processed_data_dir):
        """
        Creates, loads, or caches a BM25 keyword index for the dataset.

        Args:
            tokenized_chunks (list): Tokenized text chunks for BM25.

        Returns:
            BM25Okapi: Precomputed BM25 index.
        """

        # Just so we can see what went into bm25
        testing_dict = {}
        for i, chunk in enumerate(tokenized_chunks):
            testing_dict[i] = list(chunk)
        with open("check_tokenized_chunks.json", "w") as f:
            json.dump(testing_dict, f, indent=4)

        bm25 = BM25Okapi(tokenized_chunks)
        index_path = processed_data_dir / "bm25_index.pkl"

        with open(index_path, "wb") as f:
            pickle.dump(bm25, f)
            logger.info(f"Saved BM25 index for {self._dataset_name} to {index_path}")

        return bm25
