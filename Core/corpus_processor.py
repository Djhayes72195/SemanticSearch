import json
import os
import unicodedata
import re
import time
import hashlib
from config import PROCESSED_DATA_PATH
from pathlib import Path
import spacy
from Core.tokenizer import Tokenizer
from EmbeddingGeneration.splitter import TextSplitter


class CorpusProcessor:
    def __init__(
        self, corpus, config, dataset_name, embedding_manager, keyword_manager
    ):
        self._corpus = corpus
        self._config = config
        self.dataset_name = dataset_name
        self.embedding_manager = embedding_manager
        self.keyword_manager = keyword_manager
        self._tokenizer = Tokenizer()

        # Initialize NLP & text splitter
        self.nlp = spacy.load("en_core_web_sm")
        self.text_splitter = TextSplitter(
            methods=config["splitting_method"], nlp=self.nlp
        )

    def process(self):
        """ """
        processed_corpus_id = self.generate_processed_data_identifier()
        processed_data_dir = PROCESSED_DATA_PATH / Path(processed_corpus_id)

        if self._check_required_files(processed_data_dir):
            return processed_corpus_id

        processed_corpus_id = self._encode_corpus(processed_corpus_id, processed_data_dir)

        return processed_corpus_id

    def _encode_corpus(self, processed_corpus_id, processed_data_dir):
        tokenized_chunks = []
        id_mapping = {}
        chunk_id_mapping = {}

        chunk_id_counter = 0  # Unique ID for each chunk
        start_time = time.perf_counter()
        print(f"Processing started for dataset: {self.dataset_name}")

        for doc_id, doc_text in self._corpus.data.items():
            chunks = self.text_splitter.split(doc_text)

            for chunk in chunks:
                chunk_text = chunk["text"]
                parent_chunk = chunk.get("parent_large_chunk", "")

                # **BM25 Tokenization**
                tokenized_chunks.append(
                    self._tokenizer.tokenize(chunk_text)
                )

                self.embedding_manager.generate_and_store_embedding(
                    chunk_id_counter, chunk_text
                )

                # **id_mapping**
                id_mapping[chunk_id_counter] = {
                    "location": doc_id,
                    "text": chunk_text,
                    "char_range": chunk["range"],
                    "splitting_method": chunk["method"],
                    "parent_chunk_range": (
                        parent_chunk.get("range") if parent_chunk else None
                    ),
                }

                chunk_id_mapping[chunk_id_counter] = doc_id
                chunk_id_counter += 1

        end_time = time.perf_counter()
        processing_time = end_time - start_time
        metadata = {
            "dataset_name": self.dataset_name,
            "processed_corpus_id": processed_corpus_id,
            "processing_time": processing_time,
            "config": self._config,
        }
        self._save_results(
            processed_corpus_id,
            processed_data_dir,
            tokenized_chunks,
            id_mapping,
            metadata,
        )

        return processed_corpus_id

    def _save_results(
        self,
        processed_corpus_id,
        processed_data_dir,
        tokenized_chunks,
        id_mapping,
        metadata,
    ):
        processed_data_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_manager.save_embeddings(processed_data_dir)
        self.keyword_manager.save_index(
            tokenized_chunks, processed_data_dir
        )
        self._save_id_mapping(id_mapping, processed_data_dir)
        self._save_metadata(processed_data_dir, metadata)


    def _save_metadata(self, processed_data_dir, metadata):
        metadata_path = processed_data_dir / Path("metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def _save_id_mapping(self, id_mapping, processed_data_dir):
        """
        Save id mapping file
        
        id_mapping.json maps the id of a chunk, as stored in bm25
        and annoy, to critical fields:
            - Source document location.
            - Chunk text
            - splitting method used to generate the chunk
            - character range, as an integer range
            - parent chunk range
                - Range of int if chunk as a parent
                - None otherwise

        id_mapping.json is associated with annoy and bm25
        resources according to a unique key. See
        `generate_processed_data_identifier`.
        """
        id_mapping_path = processed_data_dir / Path("id_mapping.json")
        with open(id_mapping_path, "w") as f:
            json.dump(id_mapping, f, indent=4)

    def generate_processed_data_identifier(self):
        """
        Generates a unique hashed ID for the processed corpus.

        We use this ID to identify embeddings and keyword indexes
        associated with a particular dataset and configuration.
        
        Only configurations which would require the corpus to
        be re-processed are included. This allows us to reuse
        embeddings and keyword indexes across configurations.

        Example:
            - Changing splitting method requires recomputation of
            indexes ----> included in ID
            - Changing semantic vs keyword similarity weights
            does not require recomputation of indexes
            ----> not included in ID.
        """
        key_settings = (  # Only include configs that would require re-processing the corpus
            self.dataset_name,
            self._config["splitting_method"],
            self._config["embedding_model"],
            self._config["cleaning_method"],
            self._config["split_filtering"],
        )
        unique_string = "__".join(map(str, key_settings))
        return hashlib.md5(unique_string.encode()).hexdigest()

    def _check_required_files(self, directory: str):
        dir_path = Path(directory)

        if not dir_path.is_dir():
            return False

        # Required filenames
        required_files = {"id_mapping.json", "metadata.json", "embeddings.ann", "bm25_index.pkl"}
        existing_files = {file.name for file in dir_path.iterdir() if file.is_file()}

        return required_files.issubset(existing_files) 
