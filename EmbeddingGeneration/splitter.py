class TextSplitter:
    def __init__(self, method):
        self._method = method
        self._splits = []

    def split(self, document):
        if self._method == "sentence_by_sentence":
            self._by_sentence(document)

    def _by_sentence(self):
        pass