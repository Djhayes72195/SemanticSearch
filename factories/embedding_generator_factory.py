from Models.models import CorpusData
from EmbeddingGeneration.generate_embeddings import EmbeddingGenerator, TextSplitter

def create_embedding_generator(corpus: CorpusData, data_path: str, split_method: str) -> EmbeddingGenerator:
    text_splitter = TextSplitter(method=split_method)
    return EmbeddingGenerator(corpus, data_path, text_splitter)
