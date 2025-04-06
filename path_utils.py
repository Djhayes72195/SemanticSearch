from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = ROOT_DIR / "RawData"
TEST_DATA_PATH = ROOT_DIR / "TestData"
DEFAULT_DATA_PATH = ROOT_DIR / "TestData" / "SQuAD"
QUESTION_ANSWER_PATH = ROOT_DIR / "TestRunner" / "QuestionAnswer"
NOTEBOOKS_PATH = ROOT_DIR / "Notebooks"
TEST_RESULTS_PATH = ROOT_DIR / "TestRunner" / "TestResults"
EMBEDDINGS_PATH = ROOT_DIR / "Embeddings"
GRID_SEARCH_CONFIG_PATH = ROOT_DIR / "TestRunner" / "grid_search_config.json"
PROCESSED_DATA_PATH = ROOT_DIR / "ProcessedData"

