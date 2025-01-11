import itertools
import json
import hashlib
import argparse
import os
from .factory import create_test_runner
from factories.corpus_factory import create_corpus
from factories.embedding_generator_factory import create_embedding_generator
import spacy
from pathlib import Path
from .config import (
    NON_MUTUALLY_EXCLUSIVE_CONFIGS,
    DATASETS_PATH,
    HASH_RECORD_PATH,
    EMBEDDINGS_PATH
)


def load_configs(config_path="grid_search_config.json"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, config_path)
    with open(full_path, "r") as f:
        return json.load(f)


def powerset(iterable):
    """
    Generate all subsets of an iterable, including the empty set and full set.
    """
    s = list(iterable)
    return [
        list(subset)
        for subset in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(1, len(s) + 1)
        )
    ]


def generate_combinations(config_space):
    """
    Generate all combinations of configuration options,
    including multi-value attributes like splitting_method.
    """
    updated_config_space = {}

    for key, values in config_space.items():
        if (
            isinstance(values, list) and key in NON_MUTUALLY_EXCLUSIVE_CONFIGS
        ):  # Handle non-mutually exclusive case
            updated_config_space[key] = powerset(values)
        else:
            updated_config_space[key] = values

    keys, values = zip(*updated_config_space.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def run_test(config, dataset_name):

    data_path = DATASETS_PATH / Path(dataset_name)
    corpus = create_corpus(data_path)
    embedding_identifier = generate_embedding_identifier(
        dataset_name,
        config
    )
    embedding_path = EMBEDDINGS_PATH / Path(embedding_identifier + ".ann")
    embedding_exists = check_if_embedding_exists(
        embedding_path
    )
    if embedding_exists:
        # TODO: gather id_mapping and metadata from previous run
        pass
    else:
        nlp = spacy.load("en_core_web_sm")
        eg = create_embedding_generator(
            corpus, dataset_name, nlp, config, embedding_path
        )
        id_mapping, embedding_time = eg.generate_embeddings()

    tr = create_test_runner(
        dataset_name,
        corpus,
        id_mapping,
        embedding_path,
        embedding_time,
        config,
    )
    tr.run_test()

def check_if_embedding_exists(embedding_path):
    if not os.path.exists(embedding_path):
        print(f"File does not exist: {embedding_path}")
        return False

    if os.path.getsize(embedding_path) == 0:
        print(f"File is empty: {embedding_path}")
        return False

    # TODO: We may want to add additional checks

    return True

def generate_embedding_identifier(dataset_name, config):
    """
    Generate a hashed identifier for the embeddings based on the dataset name and config.
    
    Parameters
    ----------
    dataset_name : str
        The name of the dataset.
    config : dict
        The configuration dictionary for the test run.

    Returns
    -------
    str
        A unique hashed identifier for the embeddings.
    """
    # Combine key settings into a unique string
    key_settings = (
        dataset_name,
        config["splitting_method"],
        config["embedding_model"],
        config["cleaning_method"],
        config["split_filtering"]
    )
    unique_string = "__".join(map(str, key_settings))

    # Generate a hash for the unique string
    hash_object = hashlib.md5(unique_string.encode())
    embedding_hash = hash_object.hexdigest()

    # Save hash to record for traceability
    save_hash_record(embedding_hash, unique_string)

    return embedding_hash

def save_hash_record(embedding_hash, unique_string):
    """
    Save a mapping of the embedding hash to the unique string for traceability.

    Parameters
    ----------
    embedding_hash : str
        The generated hash.
    unique_string : str
        The unique string that generated the hash.
    """
    # Load existing records or create a new dictionary
    if os.path.exists(HASH_RECORD_PATH):
        with open(HASH_RECORD_PATH, "r") as f:
            hash_record = json.load(f)
    else:
        hash_record = {}

    # Update the record with the new hash
    if embedding_hash not in hash_record:
        hash_record[embedding_hash] = unique_string
        with open(HASH_RECORD_PATH, "w") as f:
            json.dump(hash_record, f, indent=4)

def load_hash_record():
    """
    Load the hash record to check for duplicates or trace settings.

    Returns
    -------
    dict
        A dictionary mapping hashes to their unique strings.
    """
    if os.path.exists(HASH_RECORD_PATH):
        with open(HASH_RECORD_PATH, "r") as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["grid", "single"], default="grid", help="Run mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to specific config file for single run",
    )
    parser.add_argument(
        "--dataset-name",
        help="Name of dataset used for testing"
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    if args.mode == "grid":
        config_space = load_configs()
        configs = generate_combinations(config_space)
        for config in configs:
            run_test(config, dataset_name)
    elif args.mode == "single" and args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
        run_test(config, dataset_name)
    else:
        print("Invalid arguments. Use --help for details.")


if __name__ == "__main__":
    main()
