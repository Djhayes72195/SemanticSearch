import os
import hashlib
from TestRunner.config import HASH_RECORD_PATH
import json

class EmbeddingsManager:

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
        EmbeddingsManager.save_hash_record(embedding_hash, unique_string)

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