import pickle
import re
import unicodedata
from pathlib import Path
from rank_bm25 import BM25Okapi
import logging

PATH_TO_KEYWORD_INDEX = Path(
    "C:\\Users\\Djhay\\OneDrive\\Desktop\\Projects\\Hackathon\\Hackathon\\KeywordIndexes"
)

# Setup logger
logger = logging.getLogger(__name__)


class KeywordManager:
    def __init__(self, dataset_name):
        self._dataset_name = dataset_name
        self._cache = {}
        self._cache_dir = Path(PATH_TO_KEYWORD_INDEX)
        self._cache_dir.mkdir(parents=True, exist_ok=True)  # Ensure cache directory exists

    def generate_or_load_keyword_model(self, corpus):
        """
        Creates, loads, or caches a BM25 keyword index for the dataset.

        Args:
            corpus (CorpusData): Corpus object.

        Returns:
            BM25Okapi: Precomputed BM25 index.
        """
        cache_path = self._cache_dir / f"{self._dataset_name}.pkl"

        if self._dataset_name in self._cache:
            return self._cache[self._dataset_name]

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                logger.info(f"Loading cached BM25 index for {self._dataset_name} from {cache_path}")
                bm25 = pickle.load(f)
                self._cache[self._dataset_name] = bm25
                return bm25

        logger.info(f"Building BM25 index for dataset: {self._dataset_name}")

        tokenized_corpus = [self._tokenize(doc) for doc in corpus.data.values()]

        bm25 = BM25Okapi(tokenized_corpus)

        with open(cache_path, "wb") as f:
            pickle.dump(bm25, f)
            logger.info(f"Saved BM25 index for {self._dataset_name} to {cache_path}")

        self._cache[self._dataset_name] = bm25

        return bm25

    def _tokenize(self, text):
        """
        # TODO: Put this in helper function module
        Tokenizes and normalizes text (lowercase, removes punctuation, removes accents, and optionally removes stop words).

        Args:
            text (str): Input text.

        Returns:
            list: A list of tokenized words.
        """
        # Normalize accents
        text = ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))

        # Tokenize words
        tokens = re.findall(r"\b\w+\b", text.lower())

        return tokens
