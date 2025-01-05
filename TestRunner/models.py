from .config import QUESTION_ANSWER_PATH
import json
from dataclasses import dataclass, asdict
from pathlib import Path


class QuestionAnswer:
    """
    A class for managing question-answer configurations.

    This class reads a JSON configuration file and retrieves
    question-answer data associated with a given test data path.

    The testing suite will evaluate the success of the run
    based on how well the answers to a given question are
    retrieved.

    Expected JSON Format
    --------------------
    {
        "test_data_path_1": {"question": "What is X?", "answer": "X is ..."},
        "test_data_path_2": {"question": "What is Y?", "answer": "Y is ..."}
    }
    """

    def __init__(self, dataset_name: str):
        """
        Initialize the QuestionAnswer class.

        Parameters
        ----------
        dataset_name : str
            The name of the root folder of the
            data to be queried.

        Attributes
        ----------
        _question_answer : dict or None
            The question-answer data for the specified test data path.

        Raises
        ------
        FileNotFoundError
            If the `question_answer.json` file is not found.
        JSONDecodeError
            If the JSON file is malformed.
        """
        qa_path = QUESTION_ANSWER_PATH / f"{dataset_name}.json"

        try:
            with qa_path.open("r") as f:
                self._question_answer = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Question and Answer file not found at {qa_path}."
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Question and Answer JSON at {qa_path} is invalid."
            ) from e

    @property
    def question_answer(self) -> dict:
        """
        Retrieve the question-answer data.

        Returns
        -------
        dict or None
            The question-answer data for the specified test data path,
            or None if no data is found.
        """
        return self._question_answer


@dataclass
class TestMetadata:
    """
    Metadata for a test run.

    This class contains metadata related to a single run of the testing suite,
    including details about the dataset, embedding model, and processing methods.

    Attributes
    ----------
    dataset_name : str
        The name of the parent folder containing the test data.
    embedding_model : str
        A string representation of the model used to create the text embeddings.
    splitting_methods : list
        The methods used to split the text before embedding (e.g., by sentence).
    annoy_trees : int
        The number of trees used to create the annoy index.
    embedding_time : float
        The time taken to generate the embeddings, in seconds.
    """
    dataset_name: str
    embedding_model: str
    splitting_methods: list
    annoy_trees: int
    embedding_time: float
    
    def to_dict(self) -> dict:
        """
        Return a copy of the dataclass as a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the metadata.
        """
        return self.__dict__.copy()

    def to_normalized_name(self) -> str:
        components = []
        for key, value in self.to_dict().items():
            if isinstance(value, list):
                value = "_".join(value)
            components.append(f"{key}={value}")
        return "__".join(components)
