import os
import json
from pathlib import Path
from torch import nn
from annoy import AnnoyIndex
from .models import TestMetadata
from .config import TEST_RESULTS_PATH
from Core.results_processors import TestingResultProcessor
from EmbeddingGeneration.config import PATH_TO_EMBEDDINGS
from sentence_transformers import SentenceTransformer

        # dataset_name=dataset_name,
        # corpus=corpus,
        # id_mapping=id_mapping,
        # embedding_manager=embedding_manager,
        # embedding_time=embedding_time,
        # config=config,
        # qa=question_answer
class TestRunner:
    """
    TestRunner is responsible for executing tests using embeddings and configurations
    provided by the EmbeddingsManager and processing the results.
    """

    def __init__(
        self,
        dataset_name,
        corpus,
        id_mapping,
        embedding_manager,
        embedding_time,
        config,
        qa,
        embedding_id,
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
        self._id_mapping = id_mapping
        self._embedding_manager = embedding_manager
        self._embedding_time = embedding_time
        self._config = config
        self._qa = qa
        self._embedding_id = embedding_id

        # Use the embedding generator's identifier for loading Annoy index
        self._embedding_dim = 384  # Should match the dimension of your embeddings
        self._annoy_index = self._load_annoy_index()

        self._similarity_calculator = similarity_calculator

        model_name = config["embedding_model"]
        self._embedding_model = embedding_manager.embedding_model_factory.get_model(
            model_name
        )

        # Metadata for tracking the test run
        self._metadata = TestMetadata(
            dataset_name=dataset_name,
            embedding_model=model_name,
            splitting_methods=config.get("splitting_method", []),
            annoy_trees=10,  # Hardcoded for now, could make configurable
            embedding_time=embedding_time,
        )

        self._results_processor = TestingResultProcessor(corpus)

    def _load_annoy_index(self):
        """
        Load the Annoy index for similarity search.
        """
        index_path = Path(PATH_TO_EMBEDDINGS) / f"{self._embedding_id}.ann"
        annoy_index = AnnoyIndex(self._embedding_dim, "angular")
        annoy_index.load(str(index_path))
        return annoy_index

    def run_test(self):
        """
        Execute the tests and save the results.
        """
        case_by_case_results = []

        for test_case in self._qa.question_answer:
            query = test_case.get("query", "")
            expected_answer = test_case.get("answer", "")

            if not query:
                print(f"Skipping test case: Missing query in {test_case}")
                continue

            # Generate embedding for the query
            embedded_query = self._embedding_model.encode(query, convert_to_tensor=True)

            # Perform similarity search
            annoy_output = self._query_documents(embedded_query)

            # Process and store results
            processed_results = self._results_processor.process(
                annoy_output, self._id_mapping, query, expected_answer
            )
            case_by_case_results.append(processed_results)

        # Save results and metadata
        final_results = {
            "metadata": self._metadata.to_dict(),
            "results": case_by_case_results,
        }
        self._write_results(final_results)

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
            embedded_query, 5, include_distances=True
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
        normalized_name = self._metadata.to_normalized_name()
        return TEST_RESULTS_PATH / f"{normalized_name}.{extension}"
