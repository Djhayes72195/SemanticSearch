import argparse
from pathlib import Path
from .factory import create_test_runner
from factories.corpus_factory import create_corpus
from factories.embedding_generator_factory import create_embedding_generator
import spacy
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
    parser.add_argument(
        "--split_methods",
        type=str,
        default=None,
        help="Comma-separated list of methods used to split documents (e.g., 'by_sentence,by_paragraph').",
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
    splitting_methods = args.split_methods.split(",") if args.split_methods else []
    splitting_methods = [method.strip() for method in splitting_methods]
    embedding_path = args.embedding_path
    embed_docs = args.embed_docs
    if embed_docs and embedding_path:
        raise ValueError("--embed-docs should be False if --embedding-location is provided.")
    nlp = spacy.load("en_core_web_sm")
    corpus = create_corpus(data_path)
    if embed_docs:
        eg = create_embedding_generator(
            corpus,
            data_path,
            splitting_methods,
            nlp
        )
        eg.generate_embeddings()
        id_mapping = eg.id_mapping
        embedding_path = eg.embedding_path
        embedding_time = eg.embedding_time  # TODO: return instead of directly accessing
    else:
        # TODO: gather id_mapping and metadata from previous run
        pass
    # Run tests
    tr = create_test_runner(data_path, corpus, id_mapping, embedding_path, splitting_methods, embedding_time)
    tr.run_test()

if __name__ == "__main__":
    main()
