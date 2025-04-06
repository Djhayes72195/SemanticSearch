import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .config import RECURSIVE_SPLITTER_CONFIG


class TextSplitter:
    def __init__(self, methods: list, nlp):
        self._methods = methods if isinstance(methods, list) else [methods]
        self._nlp = nlp
        self._recursive_splitters = self._create_recursive_splitters()
        self._method_map = {
            "by_sentence": self._by_sentence,
            "recursive_split": self._recursive_split
        }

    def _create_recursive_splitters(self):
        return {
            "large": [
                RecursiveCharacterTextSplitter(
                **RECURSIVE_SPLITTER_CONFIG["large_w_overlap"]
                ),
            ],
            "small": [
                RecursiveCharacterTextSplitter(
                    **RECURSIVE_SPLITTER_CONFIG["small_w_overlap"]
                ),
            ]
        }

    def split(self, document):
        """
        Creates and return the splits for a given document.

        The `splits` object is a dictionary where each key corresponds to a splitting method
        (e.g., "by_sentence", "by_paragraph"). The value for each key is a list of dictionaries,
        with each dictionary representing a text segment and its metadata.

        Data Structure:
        ---------------
        splits = {
            "by_sentence": [
                {
                    "text": str,  # The text of the sentence split
                    "range": [int, int]  # Character range of the split in the original document (start, end)
                },
                ...
            ],
            "by_paragraph": [
                {
                    "text": str,  # The text of the paragraph split
                    "range": [int, int] or None  # Character range of the split in the original document, or None if unavailable
                },
                ...
            ],
            ...
        }

        Example:
        --------
        splits = {
            "by_sentence": [
                {
                    "text": "This is the first sentence.",
                    "range": [0, 26]
        """
        """
        Splits the document using specified methods and returns a flat list of results.
        Each entry includes the split text, character range, and splitting method.
        """
        splits = []

        for method in self._methods:
            if method not in self._method_map:
                raise ValueError(f"Split method '{method}' not available.")
            
            raw_splits = self._method_map[method](document)
            # Implement usefulness filter
            for split in raw_splits:
                split["method"] = method

            splits.extend(raw_splits)

        return splits

    def _recursive_split(self, document):
        """
        Implements multi-granularity chunking:
        1. Large chunks are first extracted.
        2. Each large chunk is then broken into smaller chunks.
        3. Both large and small chunks are stored in the output.
        4. Small chunks maintain a reference to their parent large chunk.
        """
        all_splits = []

        for splitter in self._recursive_splitters["large"]:
            large_chunks = splitter.split_text(document)

            large_ranges = [
                [document.find(chunk), document.find(chunk) + len(chunk)]
                for chunk in large_chunks
            ]

            for large_chunk, large_range in zip(large_chunks, large_ranges):
                large_chunk_entry = {
                    "text": large_chunk,
                    "range": large_range,
                    "granularity": "large"
                }
                all_splits.append(large_chunk_entry)

                large_start = large_range[0]  # Offset for small chunk ranges
                for small_splitter in self._recursive_splitters["small"]:
                    small_chunks = small_splitter.split_text(large_chunk)

                    small_ranges = [
                        [large_start + large_chunk.find(chunk), large_start + large_chunk.find(chunk) + len(chunk)]
                        for chunk in small_chunks
                    ]

                    for small_chunk, small_range in zip(small_chunks, small_ranges):
                        all_splits.append({
                            "text": small_chunk,
                            "range": small_range,
                            "granularity": "small",
                            "parent_large_chunk": large_chunk_entry
                        })

        return all_splits

    def _usefulness_filter(self):
        # This function will probably become it's own module
        # There are lots of potential ways we could filter before embedding
        # and it might not be easy to determine what is best.
        filtered = []

        for split in self._splits:
            # Rule 1: Remove empty or whitespace-only chunks
            split_text = split['text']
            if not split_text.strip():
                continue
            
            # Rule 2: Exclude very short chunks (less than 5 characters)
            if len(split_text) < 5:
                continue

            # Rule 3: Remove non-alphanumeric chunks
            if not any(char.isalnum() for char in split_text):
                continue

            # Add to the filtered list if it passes all checks
            filtered.append(split)

        return filtered

    def _by_sentence(self, document):
        """
        Splits the input text into segments based on the following rules:
        - Split on ., ?, !
        - Split on new lines
        """
        doc = self._nlp(document)
        splits = []
        ranges = []

        for sent in doc.sents:
            splits.append(sent.text)
            ranges.append([sent.start_char, sent.end_char])

        return [{"text": split, "range": r} for split, r in zip(splits, ranges)]
