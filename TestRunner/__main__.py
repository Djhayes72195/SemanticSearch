import argparse
from pathlib import Path
from .factory import create_test_runner
from factories.corpus_factory import create_corpus
from factories.embedding_generator_factory import create_embedding_generator
from torch import nn

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Run the TestRunner pipeline.")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the directory containing test data.",
    )
    parser.add_argument( # Might want to allow multiple split methods.
        "--split_method",
        type=str,
        default=None,
        help="Method or methods used to split documents.",
    )
    parser.add_argument(
        "--embed-docs",
        type=bool,
        default=True,
        help="Switch to control if we re-embed the docs."
    )
    parser.add_argument(
        "--embedding-path",
        type=str,
        default=None,
        help="Path to embeddings to test."
    )
    args = parser.parse_args()

    # Prepare objects
    data_path = Path(args.data_path)
    splitting_method = args.split_method
    embedding_path = args.embedding_path
    embed_docs = args.embed_docs
    if embed_docs and embedding_path:
        raise ValueError("--embed-docs should be False if --embedding-location is provided.")
    corpus = create_corpus(data_path)
    if embed_docs:
        eg = create_embedding_generator(
            corpus,
            data_path,
            splitting_method
        )
        eg.generate_embeddings()
        id_mapping = eg.id_mapping
        embedding_path = eg.embedding_path
    else:
        # TODO: gather id_mapping from previous run
        pass
    # Run tests
    tr = create_test_runner(data_path, corpus, id_mapping, embedding_path)
    tr.run_test()

if __name__ == "__main__":
    main()
