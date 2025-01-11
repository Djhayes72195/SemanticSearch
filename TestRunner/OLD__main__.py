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
        "--mode", choices=["grid", "single"], default="grid", help="Run mode"
    )
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
    # SWITCH TO NORMALIZED EMBEDDING NAME, EMBED IF NO FILE
    # parser.add_argument(
    #     "--embed-docs",
    #     type=bool,
    #     default=True,
    #     help="Switch to control if we re-embed the docs.",
    # )
    # parser.add_argument(
    #     "--embedding-path", type=str, default=None, help="Path to embeddings to test."
    # )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="The model used to generate embeddings",
    )
    args = parser.parse_args()

    # Prepare objects
    data_path = Path(args.data_path)
    dataset_name = data_path.name
    splitting_methods = args.split_methods.split(",") if args.split_methods else []
    splitting_methods = [method.strip() for method in splitting_methods]
    embedding_path = args.embedding_path
    embed_docs = args.embed_docs
    model_name = args.embedding_model
    if not data_path.exists():
        raise FileNotFoundError(f"Data path {data_path} does not exist.")
    if not embedding_path and not embed_docs:
        raise ValueError("Either --embedding-path or --embed-docs must be provided.")
    nlp = spacy.load("en_core_web_sm")
    corpus = create_corpus(data_path)
    if embed_docs:
        eg = create_embedding_generator(
            corpus, dataset_name, splitting_methods, nlp, model_name
        )
        id_mapping, embedding_path, embedding_time = eg.generate_embeddings()
    else:
        # TODO: gather id_mapping and metadata from previous run
        pass
    # Run tests
    tr = create_test_runner(
        dataset_name,
        corpus,
        id_mapping,
        embedding_path,
        splitting_methods,
        embedding_time,
        model_name,
    )
    tr.run_test()


if __name__ == "__main__":
    main()
