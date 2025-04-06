import unicodedata
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

class Tokenizer:
    _shared_state = {}  # Shared state for all instances

    def __new__(cls, *args, **kwargs):
        obj = super(Tokenizer, cls).__new__(cls)
        obj.__dict__ = cls._shared_state  # Share state across instances
        return obj

    def __init__(self, remove_stopwords=True, use_stemming=False, use_lemmatization=True):
        if not hasattr(self, "initialized"):  # Ensure it only runs once
            self.remove_stopwords = remove_stopwords
            self.use_stemming = use_stemming
            self.use_lemmatization = use_lemmatization
            self.stopwords = set(stopwords.words("english")) if remove_stopwords else set()
            self.stemmer = PorterStemmer() if use_stemming else None
            self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
            self.initialized = True  # Mark initialization as done

    def normalize(self, text):
        """Removes accents and converts to lowercase."""
        text = "".join(
            c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
        )
        return text.lower()

    def tokenize(self, text):
        """Tokenizes text with the current settings."""
        text = self.normalize(text)
        tokens = re.findall(r"\b\w+\b", text)

        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stopwords]

        if self.use_stemming:
            tokens = [self.stemmer.stem(word) for word in tokens]

        if self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        return tokens
