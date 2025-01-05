import re
import json
from sentence_transformers import SentenceTransformer
from pathlib import Path
from torch import nn
from annoy import AnnoyIndex
from Models.models import TestMetadata
import numpy as np
from dataclasses import dataclass, asdict, field
from Core.results_processors import TestingResultProcessor
import json


class TestRunner:

    def __init__(
        self, qa, corpus, id_mapping, embedding_path, splitting_methods,
        embedding_time, similarity_calculator=None
    ):
        self._question_answer = qa.question_answer
        self._corpus = corpus
        self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._document_embeddings = {}

        # In case we don't want to use built in annoy similarity
        self._similarity_calculator = similarity_calculator

        self._id_mapping = id_mapping

        self._embedding_dim = 384  # TODO: extract
        metric = "angular"  # TODO: extract
        self._annoy_index = AnnoyIndex(self._embedding_dim, metric)  # TODO: Pass in args
        self._annoy_index.load(str(embedding_path))

        self._metadata = TestMetadata(
            dataset_name=self._corpus.dataset_name,
            embedding_model=self._embedding_model,
            splitting_methods=splitting_methods,
            annoy_trees=10,
            embedding_time=embedding_time
        ).to_dict()

        self._results_processor = TestingResultProcessor(self._corpus)

    def run_test(self):
        case_by_case_res = []
        for test_case in self._question_answer:
            query = test_case.get("query")
            answer = test_case.get("answer")
            embedded_query = self._embedding_model.encode(query, convert_to_tensor=True)
            annoy_output = self._query_documents(embedded_query)
            results = self._results_processor.process(
                annoy_output, self._id_mapping, query, answer
            )
            case_by_case_res.append(results)
        final_res = {
            "metadata": self._metadata,
            "results": case_by_case_res
        }
        self._write_results(final_res)

    def _write_results(self, final_res):
        file_name = Path(
            "C:\\Users\\Djhay\\OneDrive\\Desktop\\Projects\\Hackathon\\Hackathon\\TestRunner\\TestResults\\test_file.json"
        )
        with open(file_name, "w") as json_file:
            json.dump(final_res, json_file, indent=4)

    def _query_documents(self, embedded_query):
        results = self._annoy_index.get_nns_by_vector(
            embedded_query, 5, include_distances=True
        )
        return results


@dataclass
class Results:
    pct_correct: float = 0
    incorrect_guesses: dict = field(default_factory=dict)
    correct_guesses: dict = field(default_factory=dict)


@dataclass
class TestSettings:
    encoding_strategy: str
    annoy_branches: int
