import argparse
from logger import logger
from TestRunner.test_orchestrator import TestOrchestrator
from Core.embeddings_manager import EmbeddingManager
from Core.keyword_manager import KeywordManager
from Core.tokenizer import Tokenizer


def main():
    logger.info("Parsing arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["grid", "single"], default="grid", help="Run mode"
    )
    parser.add_argument(
        "--single-config-path",
        type=str,
        default=None,
        help="Path to specific config file for single run",
    )
    parser.add_argument(
        "--dataset-name",
        help="Name of dataset used for testing"
    )
    args = parser.parse_args()
    single_config_path = args.single_config_path
    dataset_name = args.dataset_name
    mode = args.mode

    em = EmbeddingManager()
    km = KeywordManager(
        dataset_name=dataset_name,
    )
    logger.info("Creating Test Orchestrator")
    to = TestOrchestrator(
        dataset_name=dataset_name,
        mode=mode,
        embedding_manager=em,
        keyword_manager=km,
        single_config_path=single_config_path,
    )
    logger.info("Begining testing process")
    to.orchestrate()
    logger.info("Test process complete")

if __name__ == "__main__":
    main()
