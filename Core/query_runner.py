from pathlib import Path
import numpy as np
from factories.embedding_model_factory import EmbeddingModelFactory
from TestRunner.config import PROCESSED_DATA_PATH
from annoy import AnnoyIndex
from Core.tokenizer import Tokenizer
import pickle


class QueryRunner:

    def __init__(self, processed_data_id, config):
        (
            self._annoy_index,
            self._keyword_index
         ) = self._load_resources(processed_data_id)
        self._tokenizer = Tokenizer()

        embedding_model_name = config["embedding_model"]
        emf = EmbeddingModelFactory()
        self._embedding_model = emf.get_model(embedding_model_name)

    def query(self, query):
        annoy_results = self._query_annoy(query)
        keyword_results = self._query_keyword(query)

        normalized_annoy = self._normalize_scores([1 - d for d in annoy_results[1]])  # Convert distances to similarities
        normalized_bm25 = self._normalize_scores(keyword_results[1])

        final_annoy_scores = (annoy_results[0], normalized_annoy)
        final_keyword_scores = (keyword_results[0], normalized_bm25)
        return final_annoy_scores, final_keyword_scores

    def _normalize_scores(self, scores):
        min_score = min(scores)
        max_score = max(scores)
        return [(s - min_score) / (max_score - min_score + 1e-9) for s in scores]  # Normalize to 0-1

    def _query_annoy(self, query):
        embedded_query = self._embedding_model.encode(query, convert_to_tensor=True)
        raw_results = self._annoy_index.get_nns_by_vector(
                    embedded_query, 5, include_distances=True
                )
        return raw_results

    def _query_keyword(self, query):
        tokenized_query = self._tokenizer.tokenize(
            query
        )
        raw_results = self._keyword_index.get_scores(tokenized_query)

        # format to better match annoy output
        top_indices = np.argsort(raw_results)[-5:][::-1]  # TODO: Extract top k to config
        top_values = raw_results[top_indices]
        return (top_indices.tolist(), top_values.tolist())


    def _query_documents_keyword(self, query):
        query_tokens = self.tokenize(query)
        keyword_results = self._keyword_index.get_scores(query_tokens)
        return keyword_results

    def _load_resources(self, processed_data_id):
        resources_dir = PROCESSED_DATA_PATH / Path(
            processed_data_id
        )
        annoy_index = self._load_annoy_index(resources_dir)
        keyword_index = self._load_keyword_index(resources_dir)
        return annoy_index, keyword_index

    def _load_annoy_index(self, resources_dir):
        """
        Load the Annoy index for similarity search.
        """
        path = resources_dir / Path("embeddings.ann")
        embedding_dim = 384 # Extract to config
        annoy_index = AnnoyIndex(embedding_dim, "angular")
        annoy_index.load(str(path))
        return annoy_index

    def _load_keyword_index(self, resources_dir):
        path = resources_dir / Path("bm25_index.pkl")
        with open(path, 'rb') as f:
            keyword_index = pickle.load(f)
        return keyword_index

