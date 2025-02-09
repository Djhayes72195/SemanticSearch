from EmbeddingGeneration.splitter import TextSplitter
import spacy


class CorpusProcessor:
    def __init__(
        self, corpus, config, embedding_manager, keyword_manager, dataset_name
    ):
        self._corpus = corpus
        self._config = config
        self.embedding_manager = embedding_manager
        self.keyword_manager = keyword_manager
        self.dataset_name = dataset_name

        splitting_methods = self._config["splitting_method"]
        self.nlp = spacy.load("en_core_web_sm")
        self.text_splitter = TextSplitter(methods=splitting_methods, nlp=self.nlp)

    def process(self):
        """
        Process the corpus into chunks and generate metadata.
        """
        tokenized_chunks = []
        embeddings = []
        chunk_metadata = {}
        chunk_id_mapping = {}

        chunk_id_counter = 0  # Unique ID for each chunk

        for doc_id, doc_text in self._corpus.data.items():
            # Apply the same chunking strategy for BM25 & Embeddings
            chunks = self.text_splitter.split(doc_text)
                            # "text": small_chunk,
                            # "range": small_range,
                            # "granularity": "small",
                            # "parent_large_chunk": large_chunk_entry
            for chunk in chunks:
                chunk_text = chunk["text"]
                parent_chunk = chunk.get("parent_large_chunk", "")

                # **BM25 Tokenization** TODO: Might want to make sure this tokenization is reasonable
                tokenized_chunks.append(chunk_text.lower().split())

                # **Embedding Computation**
                embedding = self.embedding_manager.compute_embedding(chunk_text)
                embeddings.append(embedding)

                # **Metadata**
                chunk_metadata[chunk_id_counter] = {
                    "location": doc_id,
                    "text": chunk_text,
                    "char_range": chunk["range"],
                    "file_path": chunk.get("file_path", "N/A"),
                    "splitting_method": chunk["method"],
                    "parent_chunk_range": (
                        parent_chunk.get("range") if parent_chunk else None
                    ),
                }

                # Store ID Mapping
                chunk_id_mapping[chunk_id_counter] = doc_id
                chunk_id_counter += 1

        return {
            "tokenized_chunks": tokenized_chunks,
            "embeddings": embeddings,
            "chunk_metadata": chunk_metadata,
            "chunk_id_mapping": chunk_id_mapping,
        }
