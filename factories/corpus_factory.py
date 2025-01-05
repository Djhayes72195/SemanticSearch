from pathlib import Path
from Models.models import CorpusData

def create_corpus(data_path: str) -> CorpusData:
    """Factory method for Corpus class"""
    path = Path(data_path)
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Invalid data path: {data_path}")
    return CorpusData(path)

