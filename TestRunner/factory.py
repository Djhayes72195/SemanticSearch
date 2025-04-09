from .models import QuestionAnswer
from Core.corpus_data import CorpusData
from TestRunner.test_runner import TestRunner


def create_test_runner(
    dataset_name: str,
    corpus: CorpusData,
    processed_corpus_id,
    config,
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
        processed_corpus_id=processed_corpus_id,
        config=config,
        qa=question_answer,
   )
