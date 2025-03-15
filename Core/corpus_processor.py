from pathlib import Path
import hashlib
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
    def __init__(self, corpus, config, dataset_name, embedding_manager, keyword_manager, testing=False):
        self._corpus = corpus
        self._config = config
        self.dataset_name = dataset_name
        self.embedding_manager = embedding_manager
        self.keyword_manager = keyword_manager
        self.testing = testing

        self._tokenizer = Tokenizer()
        # Initialize NLP & text splitter
        self.nlp = spacy.load("en_core_web_sm")
        self.text_splitter = TextSplitter(
            methods=config["split_methods"], nlp=self.nlp
        )

        if testing:
            self.processed_corpus_id = self.generate_processed_data_identifier()
            self.processed_data_dir = PROCESSED_DATA_PATH / Path("Testing") / self.processed_corpus_id
        else:
            self.processed_corpus_id = None  # Not needed for production
            self.processed_data_dir = PROCESSED_DATA_PATH / Path("Production")  # Always the same

    def process(self):
        """Processes the corpus and saves the results."""
        if self.testing and self.processed_data_dir.exists():
            print("Test embeddings already exist. Skipping processing.")
            return self.processed_corpus_id

        print(f"Processing corpus: {self.dataset_name}")
        return self._encode_corpus()

    def _encode_corpus(self):
        """Handles tokenization, embedding generation, and saving."""
        tokenized_chunks = []
        id_mapping = {}

        chunk_id_counter = 0
        start_time = time.perf_counter()

        for doc_id, doc_text in self._corpus.data.items():
            chunks = self.text_splitter.split(doc_text)

            for chunk in chunks:
                chunk_text = chunk["text"]
                self.embedding_manager.generate_and_store_embedding(chunk_id_counter, chunk_text)

                id_mapping[chunk_id_counter] = {
                    "location": doc_id,
                    "text": chunk_text,
                    "char_range": chunk["range"],
                    "splitting_method": chunk["method"]
                }

                # Tokenized chunks will be used to create bm25 index downstream
                tokenized_chunk = self._tokenizer.tokenize(
                    chunk_text
                )
                tokenized_chunks.append(tokenized_chunk)

                chunk_id_counter += 1

        end_time = time.perf_counter()
        processing_time = end_time - start_time

        metadata = {
            "dataset_name": self.dataset_name,
            "processing_time": processing_time,
            "config": self._config,
        }

        self._save_results(tokenized_chunks, id_mapping, metadata)
        return self.processed_corpus_id if self.testing else None  # Return for testing mode

    def _save_results(self, tokenized_chunks, id_mapping, metadata):
        """Saves embeddings, metadata, and keyword index."""
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_manager.save_embeddings(self.processed_data_dir)
        self.keyword_manager.save_index(tokenized_chunks, self.processed_data_dir)
        self._save_json("id_mapping.json", id_mapping)
        self._save_json("metadata.json", metadata)

    def _save_json(self, filename, data):
        path = self.processed_data_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    def generate_processed_data_identifier(self):
        """Generates a unique identifier for the processed corpus (for testing mode)."""
        key_settings = (self.dataset_name, self._config["splitting_method"], self._config["embedding_model"])
        unique_string = "__".join(map(str, key_settings))
        return hashlib.md5(unique_string.encode()).hexdigest()
