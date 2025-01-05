from pathlib import Path
import json
from dataclasses import asdict, dataclass


class CorpusData:

    def __init__(self, path):
        self.dataset_name = path.name
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

    def find_passage(self, file_name, char_range):
        start_char = char_range[0]
        end_char = char_range[1]
        return self.data.get(file_name)[start_char:end_char]

class QuestionAnswer:

    def __init__(self, test_data_path):
        with open(Path('TestRunner\\question_answer.json'), 'r') as f:
            config_data = json.load(f)
        self._question_answer = config_data.get(str(test_data_path))

    @property
    def question_answer(self):
        return self._question_answer

@dataclass
class TestMetadata:
    def __init__(self, dataset_name, embedding_model, splitting_methods, annoy_trees, embedding_time):
        self.dataset_name = dataset_name
        self.embedding_model = repr(embedding_model)
        self.splitting_methods = splitting_methods
        self.annoy_trees = annoy_trees
        self.embedding_time = embedding_time

    def to_dict(self):
        return asdict(self)

