import os
import json
from sentence_transformers import SentenceTransformer
from factories.embedding_model_factory import EmbeddingModelFactory
from pathlib import Path
from torch import nn
from annoy import AnnoyIndex
from .models import TestMetadata
from .config import TEST_RESULTS_PATH
import numpy as np
from dataclasses import dataclass, field
from Core.results_processors import TestingResultProcessor
import json


class TestRunner:

    def __init__(
        self, qa, corpus, id_mapping, embedding_path, splitting_methods,
        embedding_time, model_name, similarity_calculator=None
    ):
        self._question_answer = qa.question_answer
        self._corpus = corpus

        model_factory = EmbeddingModelFactory()
        self._embedding_model = model_factory.get_model(model_name)

        self._model_name = model_name

        # In case we don't want to use built in annoy similarity
        self._similarity_calculator = similarity_calculator

        self._id_mapping = id_mapping

        self._embedding_dim = 384  # TODO: extract
        metric = "angular"  # TODO: extract
        self._annoy_index = AnnoyIndex(self._embedding_dim, metric)  # TODO: Pass in args
        self._annoy_index.load(str(embedding_path))

        self._metadata = TestMetadata(
            dataset_name=self._corpus.dataset_name,
            embedding_model=model_name,
            splitting_methods=splitting_methods,
            annoy_trees=10,
            embedding_time=embedding_time
        )

        self._results_processor = TestingResultProcessor(self._corpus)

    def run_test(self):
        case_by_case_res = []
        for test_case in self._question_answer:
            query = test_case.get("query", "")
            answer = test_case.get("answer", "")
            if not query:
                print(f"Skipping test case: Missing query in {test_case}")
                continue
            embedded_query = self._embedding_model.encode(query, convert_to_tensor=True)
            annoy_output = self._query_documents(embedded_query)
            results = self._results_processor.process(
                annoy_output, self._id_mapping, query, answer
            )
            case_by_case_res.append(results)
        final_res = {
            "metadata": self._metadata.to_dict(),
            "results": case_by_case_res
        }
        self._write_results(final_res)

    def _write_results(self, final_res):
        file_name = self._generate_test_filepath()
        file_name.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        with open(file_name, "w") as json_file:
            json.dump(final_res, json_file, indent=4)

    def _query_documents(self, embedded_query):
        results = self._annoy_index.get_nns_by_vector(
            embedded_query, 5, include_distances=True
        )
        return results

    def _concat_metadata(self) -> str:
        return self._metadata.to_normalized_name()

    def _generate_test_filepath(self, extension: str = "json") -> str:
        normalized_name = self._concat_metadata()
        filename = Path(f"{normalized_name}.{extension}")
        return TEST_RESULTS_PATH / filename

