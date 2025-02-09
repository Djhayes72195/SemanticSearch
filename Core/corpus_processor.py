import json
import hashlib
from pathlib import Path
import spacy
from EmbeddingGeneration.splitter import TextSplitter

class CorpusProcessor:
    def __init__(self, corpus, config, dataset_name, embedding_manager, keyword_manager):
        self._corpus = corpus
        self._config = config
        self.dataset_name = dataset_name
        self.embedding_manager = embedding_manager
        self.keyword_manager = keyword_manager

        # Initialize NLP & text splitter
        self.nlp = spacy.load("en_core_web_sm")
        self.text_splitter = TextSplitter(methods=config["splitting_method"], nlp=self.nlp)

    def process(self):
        """
        Splits text, generates metadata, and returns processed chunks.
        """
        tokenized_chunks = []
        chunk_metadata = {}
        chunk_id_mapping = {}

        chunk_id_counter = 0  # Unique ID for each chunk

        for doc_id, doc_text in self._corpus.data.items():
            chunks = self.text_splitter.split(doc_text)

            for chunk in chunks:
                chunk_text = chunk["text"]
                parent_chunk = chunk.get("parent_large_chunk", "")

                # **BM25 Tokenization**
                tokenized_chunks.append(chunk_text.lower().split())

                self.embedding_manager.generate_and_store_embedding(chunk_id_counter, chunk_text)

                # **Metadata**
                chunk_metadata[chunk_id_counter] = {
                    "location": doc_id,
                    "text": chunk_text,
                    "char_range": chunk["range"],
                    "splitting_method": chunk["method"],
                    "parent_chunk_range": parent_chunk.get("range") if parent_chunk else None,
                }

                chunk_id_mapping[chunk_id_counter] = doc_id
                chunk_id_counter += 1

        processed_corpus_id = self.generate_processed_data_identifier()
        self.embedding_manager.save_embeddings(processed_corpus_id)
        self.keyword_manager.save_index(tokenized_chunks, processed_corpus_id)

        return {
            "tokenized_chunks": tokenized_chunks,
            "chunk_metadata": chunk_metadata,
            "chunk_id_mapping": chunk_id_mapping,
            "processed_corpus_id": processed_corpus_id
        }

    def generate_processed_data_identifier(self):
        """
        Generates a unique hashed ID for the processed corpus.
        """
        key_settings = (
            self.dataset_name,
            self._config["splitting_method"],
            self._config["embedding_model"],
            self._config["cleaning_method"],
            self._config["split_filtering"],
        )
        unique_string = "__".join(map(str, key_settings))
        return hashlib.md5(unique_string.encode()).hexdigest()
