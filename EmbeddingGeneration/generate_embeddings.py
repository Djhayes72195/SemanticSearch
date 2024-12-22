from annoy import AnnoyIndex
import re
from EmbeddingGeneration.splitter import TextSplitter
from pathlib import Path
from sentence_transformers import SentenceTransformer, util


class EmbeddingGenerator:
    def __init__(self, corpus, data_path, splitter):
        self._corpus = corpus
        self._data_path = data_path
        self._embedding_path = self._data_path / Path("embedding.ann")
        self._splitter = splitter
        self._embeddings = {}
        self._annoy_index = self._set_up_annoy()
        self._model = SentenceTransformer('all-MiniLM-L6-v2') # TODO: Take this out so model can be passed in

        self.id_counter = 0
        self.id_mapping = {}

    def _set_up_annoy(self):
        embedding_dim = 384  # TODO: Take this out
        metric = 'angular'  # TODO: Take this out
        annoy_index = AnnoyIndex(embedding_dim, metric)
        return annoy_index

    def generate_embeddings(self):
        for path, doc in self._corpus.items():
            splits = self._splitter.split(doc)
            self._encode_and_store(splits, path)
        self._annoy_index.save(self._embedding_path)

    def _encode_and_store(self, splits, path):
        for txt in splits:
            embedding = self._model.encode(txt)
            self._id_mapping.update({self._id_counter: str(path)})
            self._annoy_index.add_item(self._id_counter, embedding)
            self._id_counter += 1
