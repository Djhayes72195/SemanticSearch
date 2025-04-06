# README

### Local Semantic Search Engine

A lightweight semantic search engine for context-aware querying over a user-supplied Markdown corpus. Everything is self-contained — no external APIs, cloud dependencies, or hosting required.


### Installation



### Motivation

This search engine was built for:
- Students/Professionals who want a smarter way to search their personal notes when Ctrl+F doesn't cut it.
- Organizations who want to make their internal documentation more searchable.

### Technologies

This project was built with:

- **Annoy** – A local-first vector index for fast approximate nearest neighbor search. Used to store and retrieve document embeddings efficiently.

- **BM25 (via `rank_bm25`)** – An information retrieval algorithm that builds an index over tokenized document chunks. Scores results based on term frequency, inverse document frequency, and document length normalization.

- **SentenceTransformers** –  Provides pre-trained models that generate dense vector embeddings from document chunks and queries, encoding their semantic meaning.

- **spaCy / NLTK** – Used for basic NLP tasks such as tokenization, stopword removal, and sentence-level chunking.

- **LangChain** – Utilized for its recursive chunking utility to generate semantically coherent text segments.

- **NumPy / Pandas** – Used for vector math and intermediate data handling during preprocessing and scoring.

---

### ⚙️ In Progress (as of 2025-04-05)

- **FastAPI** – Adding API interface for querying, embedding, and scoring workflows.

- **Uvicorn** – Used to serve FastAPI endpoints locally with auto-reload support.


### How It Works

This project supports two distinct workflows: corpus processing and querying.


#### Corpus Processing

1. The user supplies the path to a directory containing a set of Markdown files. Files may be at the root level of the directory or nested within subfolders.
2. The contents of each Markdown file are split into chunks. Two splitting methods are currently implemented. Either or both of these methods may be used depending on runtime configuration:
   - **By sentence**: Each sentence becomes its own chunk. Sentence boundaries are detected using spaCy.
   - **Recursive**: A target chunk length is specified. The text is split recursively, aiming to match the specified length while preserving paragraph and sentence structure.
3. Each chunk is:
   - Embedded and stored in Annoy. Currently using `all-MiniLM-L6-v2` from SentenceTransformers.
   - Tokenized and processed with optional stopword removal, stemming, and/or lemmatization. The processed tokens are added to the BM25 index.
4. Results are saved:
   - Both the BM25 and Annoy indexes are saved, along with metadata that allows us to map entries back to the corresponding passage.

---

#### Querying

1. A query is specified by the user.
   - The query may be a natural language phrase, a single keyword, or a set of keywords.
2. Two representations of the query are created, reflecting the encoding process in **Corpus Processing**:
   - The query is encoded via the same language model.
   - The query is tokenized and optionally filtered (stopwords, etc.).
3. The processed representations are evaluated against the corresponding indexes:
   - The top `k` nearest neighbors to the LLM-encoded query are returned from the Annoy index.
   - The top `k` BM25 matches are returned based on keyword similarity.
4. The scores from both methods are:
   - Normalized to a 0–1 scale.
   - Combined using a weighted sum.
     - Example: `0.7 * SemanticScore + 0.3 * BM25Score`
   - Ranked by combined score.
   - **Note**: Scoring and ranking logic is still evolving. I'm considering:
     - Using score distributions instead of blindly normalizing.
     - Letting strong scores from either method override weak results from the other.
     - Adjusting weights dynamically depending on the nature of the query and/or corpus.
       - For example, BM25 might work better on short keyword queries or bullet-heavy docs, while semantic search performs better on longer, paragraph-based content.
5. The passages with the top `X` combined scores are returned to the user.
   - Currently, these are printed to the console. Eventually, they’ll be integrated with a text editor and allow navigation to the relevant file and location.


#### Test Runner vs Production Runner

In addition to the core functionality, this repo includes a dedicated TestRunner module for automated testing and evaluation. I have been using SQuAD, a benchmark question and answer dataset, for this purpose. A test run incorporates the following functionality.

- A set of candidate test configurations can be automatically generated from a baseline config file.
- Each generated configuration is used to embed/process the corpus under test.
- Each processed corpus is evaluated against a curated test set to measure search effectiveness. 
- For each test case, a set of effectiveness metrics is calculated and recorded in a JSON file.

        
### Architecture

The system is organized into three main subpackages:

- `Core/`: Shared logic used by both production and testing pipelines.
- `SearchApp/`: The production runner used for actual semantic search tasks.
- `TestRunner/`: A test runner for evaluating search quality across multiple configurations using benchmark data.

---

#### `core/`

Holds the main building blocks of the system. Key modules include:

- **`embeddings_manager.py`**  
  Generates dense vector embeddings and saves them to an Annoy index along with metadata.

- **`keyword_manager.py`**  
  Accepts tokenized, preprocessed chunks and builds a BM25 index using `rank_bm25`.

- **`corpus_processor.py`**  
  Orchestrates the full corpus preparation workflow. Internally manages:
  - An `EmbeddingManager`
  - A `KeywordManager`  
  Handles chunking, encoding, indexing, and serialization.

- **`query_runner.py`**
  Orchestrates querying of annoy and bm25 indexes.
  Handles score normalization internally.

- **`ranker.py`**
  Combines and ranks bm25 and annoy similarity scores.
  - This module somewhat hollow at the moment. It will expand when reevaluating ranking and scoring logic.

---

#### `SearchApp/`

Handles query-time logic for production use.

- Loads saved indexes and metadata.
- Accepts user queries and computes relevance scores using both semantic and keyword-based representations.
- Performs hybrid scoring and returns ranked results.
---

#### `test_runner/`

Supports automated evaluation of the system using benchmark data (e.g., SQuAD).

- Generates multiple test configurations from a baseline config.
- Runs corpus processing and query evaluation for each config.
- Computes effectiveness metrics for each test case.
- Outputs results to a JSON file for review or comparison.

    

