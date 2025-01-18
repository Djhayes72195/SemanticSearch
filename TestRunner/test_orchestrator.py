from .config import (
    DATASETS_PATH,
    GRID_SEARCH_CONFIG_PATH,
    NON_MUTUALLY_EXCLUSIVE_CONFIGS
)
from pathlib import Path
from factories.corpus_factory import create_corpus
import os
import json
import itertools
import spacy
from .factory import create_test_runner
from factories.corpus_factory import create_corpus
from factories.embedding_generator_factory import create_embedding_generator
from Core.embeddings_manager import EmbeddingsManager


class TestOrchestrator:

    def __init__(self, dataset_name, mode, embedding_manager, single_config_path=None, grid_config_path=None):
        self._dataset_name = dataset_name
        self._mode = mode
        self._single_config_path = single_config_path
        data_path = DATASETS_PATH / Path(self._dataset_name)
        self._corpus = create_corpus(data_path)

        self._embedding_manager = embedding_manager

    def orchestrate(self):
        if self._mode == "grid":
            config_space = self._load_configs()
            configs = self._generate_combinations(config_space)
            for config in configs:
                self.run_test(config)
        elif self._mode == "single" and self._single_config_path:
            with open(self._single_config_path, "r") as f:
                config = json.load(f)
            self.run_test(config)

    def run_test(self, config):
        """
        Run a single test with the given configuration.
        """
        id_mapping, embedding_time, embedding_id = (
            self._embedding_manager.generate_or_load_embeddings(config, self._corpus)
        )
    # dataset_name: str,
    # corpus: CorpusData,
    # id_mapping: dict,
    # embedding_path: str,
    # embedding_time,
    # config
        # Create and execute the test runner
        test_runner = create_test_runner(
            self._dataset_name,
            self._corpus,
            id_mapping,
            embedding_time,
            config,
            self._embedding_manager,
            embedding_id
        )
        test_runner.run_test()

    def _load_configs(self):
        """
        Load grid search configuration from a JSON file.
        """
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # full_path = os.path.join(script_dir, GRID_SEARCH_CONFIG_PATH)
        with open(GRID_SEARCH_CONFIG_PATH, "r") as f:
            return json.load(f)

    def _powerset(self, iterable):
        """
        Generate all subsets of an iterable, including the empty set and full set.
        """
        s = list(iterable)
        return [
            list(subset)
            for subset in itertools.chain.from_iterable(
                itertools.combinations(s, r) for r in range(1, len(s) + 1)
            )
        ]

    def _generate_combinations(self, config_space):
        """
        Generate all combinations of configuration options,
        including multi-value attributes like splitting_method.
        """
        updated_config_space = {}

        for key, values in config_space.items():
            if isinstance(values, list) and key in NON_MUTUALLY_EXCLUSIVE_CONFIGS:
                updated_config_space[key] = self._powerset(values)
            else:
                updated_config_space[key] = values

        keys, values = zip(*updated_config_space.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]
