import time
from logger import logger
from config import PATH_TO_EMBEDDINGS
from annoy import AnnoyIndex
from pathlib import Path
from tqdm import tqdm


class EmbeddingGenerator:
    def __init__(self, model, chunks):
        self._annoy_index = self._set_up_annoy()
        self._model = model

    def _set_up_annoy(self):
        embedding_dim = 384  # TODO: Take this out
        metric = 'angular'  # TODO: Take this out
        annoy_index = AnnoyIndex(embedding_dim, metric)
        return annoy_index

    # def generate_embeddings(self, preprocessed_chunks):
    #     # logger.info("Working on generating embeddings")
    #     # logger.info("This may take a little while")
    #     # embedding_path = PATH_TO_EMBEDDINGS / Path(f"{self.embedding_id}.ann")
    #     # start_time = time.time()

    #     # for path, doc in tqdm(self._corpus.data.items(), desc="Processing documents"):
    #     #     splits = self._splitter.split(doc)  # TODO: You should parallelize this later on.
    #     # self._encode_and_store(splits, path)

    #     n_trees = 10  # TODO: Pass in
    #     self._annoy_index.build(n_trees=n_trees)
    #     self._annoy_index.save(str(embedding_path))

    #     end_time = time.time()
    #     self.embedding_time = end_time - start_time

    #     logger.info(f"Embedding generation took {self.embedding_time:.2f} seconds.")
    #     logger.info(f"Embeddings generated and saved at {embedding_path}.")

    #     return self.id_mapping, self.embedding_time, self.embedding_id

    # def _encode_and_store(self, splits, path):
    #     for split in splits:
    #         embedding = self._generate_embedding(split)
    #         split_metadata = self._extract_split_metadata(split, path)
    #         self.id_mapping.update({self.id_counter: split_metadata})
    #         self._annoy_index.add_item(self.id_counter, embedding)
    #         self.id_counter += 1

    def generate_and_store_embedding(self, split, id):
        split_txt = split['text']
        embedding = self._model.encode(split_txt)
        self._annoy_index.add_item(id, embedding)

    # def _extract_split_metadata(self, split, path):
    #     return {
    #         "location": str(path),
    #         "char_range": split["range"],
    #         "splitting_method": split["method"],
    #         "parent_chunk_range": (
    #             split.get("parent_large_chunk").get("range")
    #             if "parent_large_chunk" in split else None
    #         )
    #     }  # This is where I should add parent chunk info
