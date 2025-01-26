from pathlib import Path

PATH_TO_EMBEDDINGS = Path(
    "C:\\Users\\Djhay\\OneDrive\\Desktop\\Projects\\Hackathon\\Hackathon\\Embeddings"
)
HASH_RECORD_PATH = Path(
    "C:\\Users\\Djhay\\OneDrive\\Desktop\\Projects\\Hackathon\\Hackathon\\TestRunner\\embedding_hashes.json"
)

HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

RECURSIVE_SPLITTER_CONFIG = {
    "separators": ["\n\n", "\n", " ", ""],
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "length_function": len
}