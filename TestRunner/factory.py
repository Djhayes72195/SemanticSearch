# TestRunner/factory.py

from pathlib import Path
from .models import QuestionAnswer
from Models.models import CorpusData
from TestRunner.test_runner import TestRunner


def create_test_runner(
    dataset_name: str,
    corpus: CorpusData,
    id_mapping: dict,
    config,
    embedding_manager,
) -> TestRunner:
    """
    Factory function to create a TestRunner instance.

    Args:
        dataset_name (str): The name of root folder of the data
        id_mapping (dict): Mapping of document IDs to embeddings.
        embedding_path (str): Path to embeddings.

    Returns:
        TestRunner: An initialized TestRunner instance.
    """
                # self._dataset_name,
            # self._corpus,
            # id_mapping,
            # config,
            # self._embedding_manager,
    question_answer = QuestionAnswer(dataset_name)
    return TestRunner(
        dataset_name=dataset_name,
        corpus=corpus,
        id_mapping=id_mapping,
        embedding_manager=embedding_manager,
        config=config,
        qa=question_answer,
   )
