# TestRunner/factory.py

from pathlib import Path
import json
from Models.models import CorpusData, QuestionAnswer
from TestRunner.test_runner import TestRunner

def create_test_runner(data_path: str, corpus: CorpusData, id_mapping: dict, embedding_path: str) -> TestRunner:
    """
    Factory function to create a TestRunner instance.

    Args:
        data_path (str): Path to the data directory.
        id_mapping (dict): Mapping of document IDs to embeddings.
        embedding_path (str): Path to embeddings.

    Returns:
        TestRunner: An initialized TestRunner instance.
    """
    path = Path(data_path)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Invalid data path: {data_path}")

    question_answer = QuestionAnswer(data_path)
    return TestRunner(question_answer, corpus, id_mapping, embedding_path)

