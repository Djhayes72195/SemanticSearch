import re

class TextSplitter:
    def __init__(self, method):
        self._method = method

    def split(self, document):
        if self._method == "by_sentence":
            splits = self._by_sentence(document)
        else:
            raise ValueError(f"Split method '{self._method}' not available.")  # Replace print with exception

        splits = self._usefulness_filter(splits)  # TODO: Make usefulness filter variable.
        return splits

    def _usefulness_filter(self, splits):
        # This function will probably become it's own module
        # There are lots of potential ways we could filter before embedding
        # and it might not be easy to determine what is best.
        filtered = []

        for split in splits:
            # Rule 1: Remove empty or whitespace-only chunks
            if not split.strip():
                continue
            
            # Rule 2: Exclude very short chunks (less than 5 characters)
            if len(split) < 5:
                continue

            # Rule 3: Remove non-alphanumeric chunks
            if not any(char.isalnum() for char in split):
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
        # Use a regular expression to split on ., ?, !, or new lines (\n)
        splits = re.split(r'[.?!\n]+', document)
        
        # Remove any empty strings or leading/trailing whitespace
        return [split.strip() for split in splits if split.strip()]