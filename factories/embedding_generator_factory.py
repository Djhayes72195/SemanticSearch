from Models.models import CorpusData
from EmbeddingGeneration.generate_embeddings import EmbeddingGenerator, TextSplitter

def create_embedding_generator(corpus: CorpusData, data_path: str, split_methods: list, nlp) -> EmbeddingGenerator:
    text_splitter = TextSplitter(methods=split_methods, nlp=nlp)
    return EmbeddingGenerator(corpus, data_path, text_splitter)
