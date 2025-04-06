# README

### Local Semantic Search Engine

A lightweight, local-first semantic search engine for context-aware querying over a user-supplied Markdown corpus. Everything is self-contained — no external APIs, cloud dependencies, or hosting required.

### Technologies

This project was built with:

- **Annoy** – A local-first vector index for fast approximate nearest neighbor search. Used to store and retrieve document embeddings efficiently.

- **BM25 (via `rank_bm25`)** – An information retrieval algorithm that builds an index over tokenized document chunks. Scores results based on term frequency, inverse document frequency, and document length normalization.

- **SentenceTransformers** –  Provides pre-trained models that generate dense vector embeddings from document chunks and queries, encoding their semantic meaning.

- **spaCy / NLTK** – Used for basic NLP tasks such as tokenization, stopword removal, and sentence-level chunking.

- **LangChain** – Utilized for its recursive chunking utility to generate semantically coherent text segments.

- **NumPy / Pandas** – Used for efficient vector math and intermediate data handling during preprocessing and scoring.

---

### ⚙️ In Progress (as of 2025-04-05)

- **FastAPI** – Adding a high-performance API interface for querying, embedding, and scoring workflows.

- **Uvicorn** – Used to serve FastAPI endpoints locally with auto-reload support.


### How It Works

This project supports two distinct workflows: corpus processing and querying.

##### Corpus Processing

1. The user supplies the path to a directory which contains a set of Markdown files. The Markdown
files may be at the root level of the directory, or nested within subfolders.
2. The contents of each Markdown file is split into chunks.
    - Two splitting methods are currently implemented.
        1. By sentence: Each sentence becomes its own chunk. Sentence boundaries are detected using spaCy.
        2. Recursive: A target chunk length is specified. We will then attempt
        to reach that chunk length by appending 