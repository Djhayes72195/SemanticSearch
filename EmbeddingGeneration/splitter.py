import re
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .config import HEADERS_TO_SPLIT_ON, RECURSIVE_SPLITTER_CONFIG


class TextSplitter:
    def __init__(self, methods: list, nlp):
        self._methods = methods
        self._nlp = nlp
        self._markdown_splitter = self._create_md_splitter()
        self._recursive_splitter = self._create_recursive_splitter()
        self._method_map = {
            "by_sentence": self._by_sentence,
            "by_paragraph": self._by_paragraph,
            # "markdown_split": self._markdown_split,
            "recursive_split": self._recursive_split
            # There is ae LangChain splitter that may be good here.
            # Also found this from Pinecone docs.
            # Here are the steps that make semantic chunking work:

# Break up the document into sentences.
# Create sentence groups: for each sentence, create a group containing some sentences before and after the given sentence. The group is essentially “anchored” by the sentence use to create it. You can decide the specific numbers before or after to include in each group - but all sentences in a group will be associated with one “anchor” sentence.
# Generate embeddings for each sentence group and associate them with their “anchor” sentence.
# Compare distances between each group sequentially: When you look at the sentences in the document sequentially, as long as the topic or theme is the same - the distance between the sentence group embedding for a given sentence and the sentence group preceding it will be low. On the other hand, higher semantic distance indicates that the theme or topic has changed. This can effectively delineate one chunk from the next.
        }

    def _create_md_splitter(self):
        return MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON
        )

    def _create_recursive_splitter(self):
        return RecursiveCharacterTextSplitter(
            **RECURSIVE_SPLITTER_CONFIG
        )

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

    def _markdown_split(self, document):
        """
        Splits on markdown headers (#, ##, etc)

        TODO: Implement this later if I decide we need it
        """
        splits = [
            split.page_content for split in
                self._markdown_splitter.split_text(
                    document
                )
            ]  # we have the option to return the name of the headers we split on.
            # For now I won't worry about that to make the splitting methods consistent,
            # but we will may want to use them eventually.

    def _recursive_split(self, document):
        ranges = []
        splits = self._recursive_splitter.split_text(
            document
        )
        start_chars = [
            document.find(split) for
            split in splits
        ]  # `find` for finding splitting indexes might not be ideal. Consider alt method.
        ranges = [
            [start_char, start_char + len(split)]
            for start_char, split in
            zip(start_chars, splits)
        ]
        return [{"text": split, "range": r} for split, r in zip(splits, ranges)]

    def _by_paragraph(self, document):
        # TODO: Think about how to do by paragraph splits and iff we need it.
        # splits = []
        # ranges = []
        # paragraphs = re.split(r'\n\s*\n', document)
        return [{'text': 'dummy text', 'range': [1,2]}]

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

    def OLD_IMPLEMENTATION_by_sentence(self, document):
        """
        Splits the input text into segments based on the following rules:
        - Split on ., ?, !
        - Split on new lines
        """
        splits = re.split(r'[.?!\n]+', document)
        
        return [split.strip() for split in splits if split.strip()]

    def OLD_IMPLEMENTATION_get_char_ranges(self, document, splits, is_testing=False):
        """
        I'm using Spacy's functionality to find ranges. I'll keep
        this around for now in case I find use for it.
        """
        starting_char_idx = 0
        char_ranges = []
        for split in splits:
            ending_char_idx = len(split) + starting_char_idx
            char_ranges.append(
                [starting_char_idx, ending_char_idx]
            )
            starting_char_idx = ending_char_idx + 1
        if is_testing:
            for i, char_range in enumerate(char_ranges):
                start_idx, end_idx = char_range[0], char_range[1]
                if document[start_idx:end_idx] != splits[i]:
                    raise ValueError("Split char ranges do not accuratly map to document")