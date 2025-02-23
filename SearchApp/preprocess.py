import argparse
from pathlib import Path
from SearchApp.config import DEFAULT_DATA_DIR
from Core.corpus_processor import CorpusProcessor
from Core.embeddings_manager import EmbeddingManager
from Core.keyword_manager import KeywordManager
from Core.corpus_data import CorpusData


def preprocess(data_dir=DEFAULT_DATA_DIR):
    print(f"🚀 Preprocessing data from: {data_dir}")

    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    corpus = CorpusData(data_dir)

    embedding_manager = EmbeddingManager()
    keyword_manager = KeywordManager(dataset_name=corpus.dataset_name)

    config = {
        "splitting_method": ["recursive_split"],
        "embedding_model": "all-MiniLM-L6-v2",
        "cleaning_method": ["no_cleaning"],
        "split_filtering": ["no_filtering"],
    }

    corpus_processor = CorpusProcessor(
        corpus=corpus,
        config=config,
        dataset_name=corpus.dataset_name,
        embedding_manager=embedding_manager,
        keyword_manager=keyword_manager,
        testing=False
    )

    corpus_processor.process()
    print(f"Processing complete! Corpus is ready to query.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess markdown files into embeddings.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing markdown files.")
    args = parser.parse_args()

    preprocess(args.data_dir)
