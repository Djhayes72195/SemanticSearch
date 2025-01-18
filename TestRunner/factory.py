# TestRunner/factory.py

from pathlib import Path
from .models import QuestionAnswer
from Models.models import CorpusData
from TestRunner.test_runner import TestRunner


def create_test_runner(
    dataset_name: str,
    corpus: CorpusData,
    id_mapping: dict,
    embedding_time: float,
    config,
    embedding_manager,
    embedding_id
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
    question_answer = QuestionAnswer(dataset_name)
    return TestRunner(
        dataset_name=dataset_name,
        corpus=corpus,
        id_mapping=id_mapping,
        embedding_manager=embedding_manager,
        embedding_time=embedding_time,
        config=config,
        qa=question_answer,
        embedding_id=embedding_id
   )
