from pathlib import Path
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
        self._question_answer = config_data.get(str(test_data_path))

    @property
    def question_answer(self):
        return self._question_answer