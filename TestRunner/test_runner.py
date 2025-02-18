import re
import json
from rank_bm25 import BM25Okapi
import pickle
import nltk
from nltk.corpus import stopwords
import hashlib

from pathlib import Path
from logger import logger
from annoy import AnnoyIndex
from .models import TestMetadata
import unicodedata
from .config import TEST_RESULTS_PATH, PROCESSED_DATA_PATH
from Core.results_processors import TestingResultProcessor
from Core.query_runner import QueryRunner
from Core.ranker import Ranker
from EmbeddingGeneration.config import PATH_TO_EMBEDDINGS
from factories.embedding_model_factory import EmbeddingModelFactory


class TestRunner:
    """
    TestRunner is responsible for executing tests using embeddings and configurations
    provided by the EmbeddingsManager and processing the results.
    """

    #         dataset_name=dataset_name,
    # corpus=corpus,
    # id_mapping=id_mapping,
    # config=config,
    # embedding_manager=embedding_manager,
    # qa=question_answer,
    def __init__(
        self,
        dataset_name,
        corpus,
        processed_corpus_id,
        config,
        qa,
        similarity_calculator=None,
    ):
        """
        Initialize the TestRunner.

        Parameters:
        -----------
        dataset_name : str
            The name of the dataset being tested.
        corpus : CorpusData
            The corpus object containing test data.
        id_mapping : dict
            A mapping of embedding IDs to document metadata.
        embedding_manager : EmbeddingsManager
            The manager responsible for embedding handling.
        embedding_time : float
            Time taken to generate embeddings.
        config : dict
            The configuration dictionary for the test.
        similarity_calculator : Callable, optional
            Custom similarity calculation function (if needed).
        """
        self._dataset_name = dataset_name
        self._corpus = corpus
        self._processed_corpus_id = processed_corpus_id
        self._qr = QueryRunner(processed_corpus_id, config)
        (self._id_mapping, self._metadata) = self._load_resources()
        self._ranker = Ranker(config, self._id_mapping, self._corpus)
        self._config = config
        self._qa = qa

        self._embedding_dim = 384  # TODO: Extract into config

        self._similarity_calculator = similarity_calculator

        model_name = config["embedding_model"]
        emf = EmbeddingModelFactory()
        self._embedding_model = emf.get_model(model_name)

        self._results_processor = TestingResultProcessor(corpus)

    def _load_resources(self):
        resources_dir = PROCESSED_DATA_PATH / Path(self._processed_corpus_id)
        id_mapping = self._load_json_resource(resources_dir, "id_mapping")
        metadata = self._load_json_resource(resources_dir, "metadata")
        return (id_mapping, metadata)

    def _load_json_resource(self, resources_dir, name):
        path = resources_dir / Path(f"{name}.json")
        with open(path, "rb") as f:
            resource = json.load(f)
        return resource

    def run_test(self):
        """
        Execute the tests and save the results.
        """
        case_by_case_results = []

        for test_case in self._qa.question_answer:
            query = test_case.get("query", "")
            ground_truth = {
                "doc": test_case.get("answer_doc", ""),
                "position": test_case.get("answer_position", ""),
                "text": test_case.get("answer_text", ""),
            }

            if not query:
                print(f"Skipping test case: Missing query in {test_case}")
                continue

            annoy_scores, keyword_scores = self._qr.query(query)
            ranking_matrix = self._ranker.rank(annoy_scores, keyword_scores)
            formatted_top_hits = self._format_for_results_processor(ranking_matrix)

            processed_results = self._results_processor.process(
                formatted_top_hits, query, ground_truth
            )
            case_by_case_results.append(processed_results)

        final_results = {
            "metadata": self._metadata,
            "results": case_by_case_results,
        }
        logger.info("Test complete, writing results.")
        self._write_results(final_results)

    def _format_for_results_processor(self, ranking_matrix):
        top_hits_ids = list(ranking_matrix["ID"])
        combined_similarity = list(ranking_matrix["Combined_Score"])
        semantic_similarity = list(ranking_matrix["Semantic_Score"])
        keyword_similarity = list(ranking_matrix["Keyword_Score"])
        top_hits_data = [self._id_mapping[str(id)] for id in top_hits_ids]

        for i, res in enumerate(top_hits_data):
            res.update({"similarity": combined_similarity[i]})
            res.update({"semantic_similarity": semantic_similarity[i]})
            res.update({"keyword_similarity": keyword_similarity[i]})
            res.update(
                {
                    "text": self._corpus.find_passage(
                        res.get("location"), res.get("char_range")
                    )
                }
            )
        return top_hits_data

    def _rank_results(self, top_hits_data):
        weights = self._config.get("semantic_vs_keyword_weights", [0.5, 0.5])
        semantic_weight, keyword_weight = (
            weights if isinstance(weights, list) and len(weights) == 2 else [0.5, 0.5]
        )
        for hit in top_hits_data:
            hit["general_sim"] = (
                hit["similarity"] * semantic_weight
                + hit["keyword_score"] * keyword_weight
            )
        top_hits_data = sorted(
            top_hits_data,
            key=lambda x: x["general_sim"],
            reverse=False,
        )[
            :10
        ]  # Keep top 10
        return top_hits_data

    # def _append_keyword_scores(self, top_hits_data, query):
    #     query_tokens = self._tokenize(query)
    #     for hit in top_hits_data:
    #         keyword_score = self._calculate_keyword_score(query_tokens, hit["text"])
    #         hit.update(
    #             {"keyword_score": 1 - keyword_score}
    #         )  # Invert keyword score to match annoy distance metric
    #     return top_hits_data

    # def _calculate_keyword_score(self, query_tokens, hit_text):
    #     """
    #     Computes the BM25 relevance score for a given passage.

    #     Args:
    #         query_tokens (list): Tokenized query.
    #         hit_text (str): The retrieved text.

    #     Returns:
    #         float: BM25 score (higher = more relevant).
    #     """
    #     passage_tokens = self._tokenize(hit_text)
    #     return self._keyword_ranking_model.get_scores(query_tokens)

    # def OLD_IMPLEMENTATION_calculate_keyword_score_jaccard(self, query_tokens, passage_text):
    #     """
    #     Computes keyword overlap using Jaccard similarity.

    #     Args:
    #         query_text (str): The search query.
    #         passage_text (str): The retrieved text.

    #     Returns:
    #         float: Jaccard similarity score (0 to 1).
    #     """
    #     passage_tokens = self._tokenize(passage_text)

    #     if not query_tokens or not passage_tokens:
    #         return 0.0

    #     intersection = len(query_tokens & passage_tokens)
    #     union = len(query_tokens | passage_tokens)
    #     return intersection / union

    def _extract_top_hits_data(self, annoy_output):
        top_hits_ids, similarities = annoy_output[0], annoy_output[1]
        top_hits_data = [self._id_mapping[str(id)] for id in top_hits_ids]

        for i, res in enumerate(top_hits_data):
            res.update({"similarity": similarities[i]})
            res.update(
                {
                    "text": self._corpus.find_passage(
                        res.get("location"), res.get("char_range")
                    )
                }
            )
        return top_hits_data

    def _query_documents(self, embedded_query):
        """
        Query the Annoy index with the embedded query.

        Parameters:
        -----------
        embedded_query : np.array
            The embedding vector for the query.

        Returns:
        --------
        list
            List of nearest neighbors and their distances.
        """
        return self._annoy_index.get_nns_by_vector(
            embedded_query, 20, include_distances=True
        )

    def _write_results(self, final_results):
        """
        Write the test results to a JSON file.

        Parameters:
        -----------
        final_results : dict
            The final results dictionary containing metadata and results.
        """
        file_name = self._generate_test_filepath()
        file_name.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        with open(file_name, "w") as json_file:
            json.dump(final_results, json_file, indent=4)
        logger.info(f"Results written to {file_name}")

    def _generate_test_filepath(self, extension="json"):
        """
        Generate a file path for saving test results.

        Parameters:
        -----------
        extension : str
            The file extension (default: "json").

        Returns:
        --------
        Path
            The path for saving the results.
        """
        name = self.generate_unique_filename()
        return TEST_RESULTS_PATH / f"{name}.{extension}"

    def generate_unique_filename(self, prefix="config", extension="json"):
        """
        Generate a unique filename from a configuration dictionary.

        Args:
            config (dict): The configuration dictionary.
            prefix (str): Prefix for the filename (default: "config").
            extension (str): File extension (default: "json").

        Returns:
            str: A unique filename based on the config.
        """
        # Convert dictionary to a sorted string
        config_str = json.dumps(self._config, sort_keys=True, separators=(",", ":"))

        # Hash the string for uniqueness
        config_hash = hashlib.md5(config_str.encode()).hexdigest()

        # Construct filename
        filename = f"{prefix}_{config_hash}.{extension}"
        
        return filename

