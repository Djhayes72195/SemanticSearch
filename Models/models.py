from pathlib import Path


class CorpusData:
    """
    A class for managing a text corpus.

    This class crawls through markdown files in the directory specified by
    the `path` parameter and stores their contents in a dictionary. It also
    provides functionality to retrieve specific passages based on file names
    and character ranges.

    Currently, only markdown files are supported. Future versions may
    support additional file types.
    """

    def __init__(self, path: Path):
        """
        Initialize the CorpusData class.

        Parameters
        ----------
        path : Path
            The root directory containing the markdown files to be analyzed.

        Attributes
        ----------
        dataset_name : str
            The name of the parent folder of the test data
        data : dict
            The corpus where keys are file paths and values are the file contents.
        """
        self.dataset_name = path.name
        self.data = self.crawl_markdown_files(path)

    def crawl_markdown_files(self, root_dir):
        """
        Crawls a directory and extracts text from markdown files.

        Parameters
        ----------
        root_dir : Path
            The root directory to search for markdown files.

        Returns
        -------
        dict
            A dictionary where keys are file paths (as strings) and values
            are the file contents (as strings).
        """
        results = {}

        for path in Path(root_dir).rglob("*.md"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    results[str(path)] = f.read()
            except Exception as e:
                print(f"Error reading {path}: {e}")

        return results

    def find_passage(self, file_name: str, char_range: list) -> str:
        """
        Retrieve a passage from the corpus.

        Parameters
        ----------
        file_name : str
            The name of the file containing the passage.
        char_range : list[int]
            A list of two integers specifying the start and end character positions.

        Returns
        -------
        str
            The extracted passage from the specified character range.

        Raises
        ------
        KeyError
            If the file_name is not found in the corpus.
        IndexError
            If the character range is invalid for the specified file.
        """
        start_char = char_range[0]
        end_char = char_range[1]
        return self.data.get(file_name)[start_char:end_char]
