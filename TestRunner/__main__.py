import argparse
from pathlib import Path
from TestRunner.models import CorpusData, QuestionAnswer
from TestRunner.test_runner import TestRunner
from EmbeddingGeneration.generate_embeddings import EmbeddingGenerator, TextSplitter
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
        "--split_method",
        type=str,
        default=[],
        help="Method or methods used to split documents.",
    )
    parser.add_argument(
        "--embed-docs",
        type=bool,
        default=True,
        help="Switch to control if we re-embed the docs."
    )
    args = parser.parse_args()

    # Prepare objects
    data_path = Path(args.data_path)
    splitting_method = args.splitting_method
    corpus = CorpusData(data_path)
    embed_docs = args.embed_docs
    if embed_docs:
        ts = TextSplitter(
            method=splitting_method
        )
        eg = EmbeddingGenerator(
            corpus,
            data_path,
            ts
        )
        eg.generate_embeddings()
    qa = QuestionAnswer(data_path)

    # Run tests
    tr = TestRunner(qa)
    tr.run_test()

if __name__ == "__main__":
    main()
