from annoy import AnnoyIndex
import re
from EmbeddingGeneration.splitter import TextSplitter
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


class EmbeddingGenerator:
    def __init__(self, corpus, data_path, splitter):
        self._corpus = corpus
        self._data_path = data_path
        self._splitter = splitter
        self._annoy_index = self._set_up_annoy()
        self._model = SentenceTransformer('all-MiniLM-L6-v2') # TODO: Take this out so model can be passed in

        self.id_counter = 0
        self.id_mapping = {}
        self.embedding_path = Path("Embeddings") / Path(str(self._data_path).split("\\")[-1] + ".ann")

    def _set_up_annoy(self):
        embedding_dim = 384  # TODO: Take this out
        metric = 'angular'  # TODO: Take this out
        annoy_index = AnnoyIndex(embedding_dim, metric)
        return annoy_index

    def generate_embeddings(self):
        print("Working on generating embeddings")
        print("This may take a little while")
        for path, doc in tqdm(self._corpus.data.items(), desc="Processing documents"):
            splits = self._splitter.split(doc) # You should parallelize this later on.
            self._encode_and_store(splits, path)
        n_trees = 10 # TODO: Pass in
        self._annoy_index.build(n_trees=n_trees)
        self._annoy_index.save(
            str(self.embedding_path)
        )
        print(f"Embeddings generated and saved at {self.embedding_path}.")

    def _encode_and_store(self, splits, path):
        for txt in splits:
            embedding = self._model.encode(txt)
            self.id_mapping.update({self.id_counter: str(path)})
            self._annoy_index.add_item(self.id_counter, embedding)
            self.id_counter += 1
