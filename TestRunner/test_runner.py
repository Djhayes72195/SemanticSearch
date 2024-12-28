import re
import json
from sentence_transformers import SentenceTransformer
from pathlib import Path
from torch import nn
from annoy import AnnoyIndex
import numpy as np
from dataclasses import dataclass, asdict, field
import json

class CorpusData:

    def __init__(self, path):
        self.data = self.crawl_markdown_files(path)

    def crawl_markdown_files(self, root_dir):
        """
        Crawls through directory and extracts text from markdown files.
        Returns a dict with filepath as key and content as value.
        """
        results = {}

        for path in Path(root_dir).rglob('*.md'):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    results[str(path)] = f.read()
            except Exception as e:
                print(f"Error reading {path}: {e}")

        return results

class QuestionAnswer:

    def __init__(self, test_data_path):
        with open(Path('TestRunner\\question_answer.json'), 'r') as f:
            config_data = json.load(f)
        self._test_set_config = config_data.get(str(test_data_path))
        self.question_answer = self._test_set_config.get('question_answer')


class TestRunner:

    def __init__(self, qa, corpus, id_mapping, embedding_path, similarity_calculator=None):
        self._question_answer = qa.question_answer
        self._corpus = corpus
        self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._document_embeddings = {}

        # In case we don't want to use built in annoy similarity
        self._similarity_calculator = similarity_calculator

        self._results = Results()
        self._id_mapping = id_mapping

        embedding_dim = 384
        metric = 'angular'
        self._annoy_index = AnnoyIndex(embedding_dim, metric) # TODO: Pass in args
        self._annoy_index.load(
            str(embedding_path)
        )

    def run_test(self):
        is_correct_list = []
        for test_case in self._question_answer:
            query = test_case.get('query')
            answer = test_case.get('answer')
            embedded_query = self._embedding_model.encode(
                query, convert_to_tensor=True
            )
            results = self._query_documents(embedded_query)
            correct, guess, off_by = self._analyze_answer(results, answer)
            is_correct_list.append(correct)
            if correct:
                self._results.correct_guesses.update(
                    {
                        query: {
                            "Answer": answer,
                            "OffBy": off_by,
                        }
                    }
                )
            else:
                guess_text = self._corpus.data[guess]
                answer_text = self._corpus.data[answer]
                self._results.incorrect_guesses.update(
                    {
                        query: {
                            "Answer": answer,
                            "OurGuess": guess,
                            "GuessText": guess_text,
                            "AnswerText": answer_text,
                            "OffBy": off_by,
                        }
                    }
                )
        self._results.pct_correct = np.mean(is_correct_list)
        self._write_results()

    def _write_results(self):
        file_name = Path(
            "C:\\Users\\Djhay\\OneDrive\\Desktop\\Projects\\Hackathon\\Hackathon\\TestRunner\\TestResults\\test_file.json"
        )
        with open(file_name, 'w') as json_file:
            json.dump(asdict(self._results), json_file, indent=4)

    def _analyze_answer(self, results, answer):
        # TODO: Left off here. Need to adapt to annoy output
        guess_id = results[0][0]
        guess_data = self._id_mapping[guess_id]
        guess = guess_data['location']
        correct = self._is_correct(answer, guess)
        if not correct:
            answer_score = self._find_answer_similarity(answer)
        answer_score = results[answer]
        guess_score = results[1][0]
        if correct:
            off_by = results[1][1] - guess_score 
        else:
            off_by = guess_score - answer_score
        return correct, guess, off_by

    def _find_answer_similarity(self, answer):
        target_entries = [
            {"id": key, **value}
            for key, value in self.id_mapping.items()
            if value["location"].endswith(answer)
        ]

    def _is_correct(self, answer, guess):
        """
        Determine if the guess is correct.
        
        Guesses are stored with the full location path, whereas
        answers are stored as relative paths. endswith is used
        to determine if the 
        """
        correct = guess.endswith(answer)
        return correct

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


