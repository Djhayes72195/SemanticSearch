import os
import hashlib
from pathlib import Path
import spacy
from factories.embedding_model_factory import EmbeddingModelFactory
from EmbeddingGeneration.config import PATH_TO_EMBEDDINGS, HASH_RECORD_PATH
from EmbeddingGeneration.generate_embeddings import EmbeddingGenerator, TextSplitter
import json



class EmbeddingsManager:
    def __init__(self, dataset_name, nlp=None, embedding_model_factory=None):
        self._dataset_name = dataset_name
        self._nlp = nlp or spacy.load("en_core_web_sm")
        self.embedding_model_factory = (
            embedding_model_factory or EmbeddingModelFactory()
        )

        self.embedding_generator = None

    def generate_or_load_embeddings(self, config, corpus):
        """
        Creates embeddings associated with a config if they don't
        exist. Either way, it returns the id mapping, embedding time,
        and embedding id.
        """
        embedding_id = self.generate_embedding_identifier(config)
        metadata = self.get_metadata_if_exists(config)
        if metadata:
            return ( 
                {int(k): v for k, v in metadata["id_mapping"].items()},
                metadata["embedding_time"],
                embedding_id
            )

        self.create_embedding_generator(config, corpus)

        id_mapping, embedding_time, embedding_id = self.embedding_generator.generate_embeddings()
        self.save_metadata(embedding_id, {"id_mapping": id_mapping, "embedding_time": embedding_time})
        return id_mapping, embedding_time, embedding_id

    def create_embedding_generator(self, config, corpus):
        try:
            model_name = config["embedding_model"]
            splitting_method = config["splitting_method"]
        except KeyError as e:
            raise ValueError(f"Missing required config key: {e}")

        model = self.embedding_model_factory.get_model(model_name)
        text_splitter = TextSplitter(methods=splitting_method, nlp=self._nlp)

        embedding_identifier = self.generate_embedding_identifier(config)
        self.embedding_generator = EmbeddingGenerator(
            corpus, self._dataset_name, text_splitter, model, embedding_identifier
        )

    def generate_embedding_identifier(self, config):
        key_settings = (
            self._dataset_name,
            config["splitting_method"],
            config["embedding_model"],
            config["cleaning_method"],
            config["split_filtering"],
        )
        unique_string = "__".join(map(str, key_settings))
        embedding_hash = hashlib.md5(unique_string.encode()).hexdigest()
        self.save_hash_record(embedding_hash, unique_string)
        return embedding_hash

    @staticmethod
    def get_metadata_path(embedding_identifier):
        return (
            PATH_TO_EMBEDDINGS
            / Path("EmbeddingMetadata")
            / f"{embedding_identifier}.json"
        )

    @staticmethod
    def save_metadata(embedding_id, metadata):
        metadata_path = EmbeddingsManager.get_metadata_path(embedding_id)
        os.makedirs(metadata_path.parent, exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    def get_metadata_if_exists(self, config):
        """
        Check if the embedding exists and load its metadata if available.
        """
        embedding_id = self.generate_embedding_identifier(config)
        if not EmbeddingsManager.check_if_embedding_exists(embedding_id):
            return None
        return EmbeddingsManager.load_metadata(embedding_id)

    @staticmethod
    def load_metadata(embedding_identifier):
        metadata_path = EmbeddingsManager.get_metadata_path(embedding_identifier)
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Corrupted metadata at {metadata_path}")
                return None
        else:
            print(f"Metadata file not found: {metadata_path}")
            return None

    def check_if_embedding_exists(embedding_identifier):
        embedding_path = PATH_TO_EMBEDDINGS / Path(f"{embedding_identifier}.ann")
        if not os.path.exists(embedding_path):
            print(f"File does not exist: {embedding_path}")
            return False

        if os.path.getsize(embedding_path) == 0:
            print(f"File is empty: {embedding_path}")
            return False

        # TODO: We may want to add additional checks

        return True

    def generate_embedding_identifier(self, config):
        """
        Generate a hashed identifier for the embeddings based on the dataset name and config.

        Parameters
        ----------
        config : dict
            The configuration dictionary for the test run.

        Returns
        -------
        str
            A unique hashed identifier for the embeddings.
        """
        # Combine key settings into a unique string
        key_settings = (
            self._dataset_name,
            config["splitting_method"],
            config["embedding_model"],
            config["cleaning_method"],
            config["split_filtering"],
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
        if os.path.exists(HASH_RECORD_PATH) and os.path.getsize(HASH_RECORD_PATH) > 0:
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
