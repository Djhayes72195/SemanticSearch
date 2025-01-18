import time
from config import PATH_TO_EMBEDDINGS
from annoy import AnnoyIndex
from EmbeddingGeneration.splitter import TextSplitter
import json
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


class EmbeddingGenerator:
    def __init__(self, corpus, dataset_name, splitter, model, embedding_id):
        self._corpus = corpus
        self._data_path = dataset_name
        self._splitter = splitter
        self._annoy_index = self._set_up_annoy()
        self._model = model

        self.id_counter = 0
        self.id_mapping = {}

        self.embedding_id = embedding_id
        self.embedding_time = None

    def _set_up_annoy(self):
        embedding_dim = 384  # TODO: Take this out
        metric = 'angular'  # TODO: Take this out
        annoy_index = AnnoyIndex(embedding_dim, metric)
        return annoy_index

    def generate_embeddings(self):
        print("Working on generating embeddings")
        print("This may take a little while")
        embedding_path = PATH_TO_EMBEDDINGS / Path(f"{self.embedding_id}.ann")
        # Start timing
        start_time = time.time()

        for path, doc in tqdm(self._corpus.data.items(), desc="Processing documents"):
            splits = self._splitter.split(doc)  # You should parallelize this later on.
            self._encode_and_store(splits, path)

        n_trees = 10  # TODO: Pass in
        self._annoy_index.build(n_trees=n_trees)
        self._annoy_index.save(str(embedding_path))

        end_time = time.time()
        self.embedding_time = end_time - start_time

        # Print timing result
        print(f"Embedding generation took {self.embedding_time:.2f} seconds.")
        print(f"Embeddings generated and saved at {embedding_path}.")

        return self.id_mapping, self.embedding_time, self.embedding_id

    def _encode_and_store(self, splits, path):
        for split in splits:
            embedding = self._generate_embedding(split)
            split_metadata = self._extract_split_metadata(split, path)
            self.id_mapping.update({self.id_counter: split_metadata})
            self._annoy_index.add_item(self.id_counter, embedding)
            self.id_counter += 1

    def _generate_embedding(self, split):
        split_txt = split['text']
        embedding = self._model.encode(split_txt)
        return embedding

    def _extract_split_metadata(self, split, path):
        return {
            "location": str(path),
            "char_range": split["range"],
            "splitting_method": split["method"]
        }
