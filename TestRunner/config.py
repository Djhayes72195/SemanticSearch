from pathlib import Path

# TODO: This should all be changed so it works on any computer
QUESTION_ANSWER_PATH = Path(
    "C:\\Users\\Djhay\\OneDrive\\Desktop\\Projects\\Hackathon\\Hackathon\\TestRunner\\QuestionAnswer"
)
TEST_RESULTS_PATH = Path(
    "C:\\Users\\Djhay\\OneDrive\\Desktop\\Projects\\Hackathon\\Hackathon\\TestRunner\\TestResults"
)
DATASETS_PATH = Path(
    "C:\\Users\\Djhay\\OneDrive\\Desktop\\Projects\\Hackathon\\Hackathon\\TestData"
)
EMBEDDINGS_PATH = Path(
    "C:\\Users\\Djhay\\OneDrive\\Desktop\\Projects\\Hackathon\\Hackathon\\Embeddings"
)
GRID_SEARCH_CONFIG_PATH = Path(
    "C:\\Users\\Djhay\\OneDrive\\Desktop\\Projects\\Hackathon\\Hackathon\\TestRunner\\grid_search_config.json"
)


NON_MUTUALLY_EXCLUSIVE_CONFIGS = [
    "splitting_method",
    "split_filtering",
    "cleaning_method",
]

