import argparse
from TestRunner.test_orchestrator import TestOrchestrator
from Core.embeddings_manager import EmbeddingsManager




def main():
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

    em = EmbeddingsManager(
        dataset_name=dataset_name
    )
    to = TestOrchestrator(
        dataset_name=dataset_name,
        mode=mode,
        embedding_manager=em,
        single_config_path=single_config_path,
    )
    to.orchestrate()

if __name__ == "__main__":
    main()
