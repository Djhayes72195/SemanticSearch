from Models.models import CorpusData
from .embedding_model_factory import EmbeddingModelFactory
from EmbeddingGeneration.generate_embeddings import EmbeddingGenerator, TextSplitter

def create_embedding_generator(
    corpus: CorpusData, dataset_name: str, nlp, config, embedding_identifier
) -> EmbeddingGenerator:
    """
    Factory method for EmbeddingGenerator class.

    Parameters
    ----------
    corpus : CorpusData
        The corpus data object.
    dataset_name : str
        The name of the dataset.
    split_methods : list
        Methods to split documents (e.g., by sentence).
    nlp : Spacy NLP Model
        A loaded Spacy model for processing.
    model_name : str
        Name of the embedding model.

    Returns
    -------
    EmbeddingGenerator
        An initialized embedding generator.
    """
    model_name = config["embedding_model"]
    splitting_method = config["splitting_method"]
    embedding_model_factory = EmbeddingModelFactory()
    model = embedding_model_factory.get_model(model_name)
    text_splitter = TextSplitter(methods=splitting_method, nlp=nlp)
    return EmbeddingGenerator(corpus, dataset_name, text_splitter, model, embedding_identifier)
