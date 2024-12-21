import re
import json
from sentence_transformers import SentenceTransformer
from pathlib import Path
from torch import nn
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
        with open(Path('TestRunner\\test_config.json'), 'r') as f:
            config_data = json.load(f)
        self._test_set_config = config_data.get(str(test_data_path))
        self.question_answer = self._test_set_config.get('question_answer')


class TestRunner:

    def __init__(self, qa, corpus, similarity_calculator):
        self._question_answer = qa.question_answer
        self._corpus = corpus
        self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._document_embeddings = {}
        self._similarity_calculator = similarity_calculator
        self._results = Results()
        x = 2

    def run_test(self):
        self._encode_corpus()
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
        highest_to_lowest_scores = sorted(
            results.items(),
            key=lambda item: item[1],
            reverse=True
        )
        guess = highest_to_lowest_scores[0][0]
        correct = guess == answer
        answer_score = results[answer]
        guess_score = highest_to_lowest_scores[0][1]
        if correct:
            off_by = guess_score - highest_to_lowest_scores[1][1]
        else:
            off_by = guess_score - answer_score
        return correct, guess, off_by

    def _query_documents(self, embedded_query):
        results = {}
        for file, contents in self._document_embeddings.items():
            similarity_score = cos(embedded_query, contents.unsqueeze(0))
            results[file] = float(similarity_score)
        return results

    def _encode_corpus(self):
        crawled_result_embeddings = self._embedding_model.encode(
            list(self._corpus.data.values()), convert_to_tensor=True
        )
        self._document_embeddings = {
            key: value for key, value in zip(
            self._corpus.data.keys(), crawled_result_embeddings
            )
        }

@dataclass
class Results:
    pct_correct: float = 0
    incorrect_guesses: dict = field(default_factory=dict)
    correct_guesses: dict = field(default_factory=dict)

from torch import nn
cos = nn.CosineSimilarity(dim=1)
path = Path("test_data\Mitosis")
qa = QuestionAnswer(path)
corpus = CorpusData(path)
tr = TestRunner(qa, corpus, cos)
tr.run_test()

