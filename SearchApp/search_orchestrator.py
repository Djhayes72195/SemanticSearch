import logging
from path_utils import DEFAULT_DATA_PATH
from Core.query_runner import QueryRunner
from Core.ranker import Ranker
from Core.corpus_data import CorpusData

logger = logging.getLogger(__name__)

class SearchOrchestrator:
    """
    Centralized search execution class.
    
    This class is responsible for handling live search queries by:
    - Running semantic search with vector embeddings
    - Running keyword search
    - Ranking and formatting results
    """

    def __init__(self, config, id_mapping):
        logger.info(f"Initializing SearchOrchestrator")

        self.corpus = CorpusData(DEFAULT_DATA_PATH)
        self._id_mapping = id_mapping

        self.query_runner = QueryRunner("Production", config)
        self.ranker = Ranker(config)

    def search(self, query):
        """
        Processes a search query and returns ranked results.

        Args:
            query (str): The user's search query.

        Returns:
            list[dict]: List of ranked search results with metadata.
        """
        logger.info(f"Processing query: {query}")

        annoy_scores, keyword_scores = self.query_runner.query(query)

        ranking_matrix = self.ranker.rank(annoy_scores, keyword_scores)

        formatted_results = self._format_results(ranking_matrix)

        return formatted_results

    def _format_results(self, ranking_matrix):
        """
        Formats the ranked search results into a structured list.

        Args:
            ranking_matrix (dict): Contains document IDs and their similarity scores.

        Returns:
            list[dict]: List of ranked search results.
        """
        top_hits_ids = list(ranking_matrix["ID"])
        combined_similarity = list(ranking_matrix["Combined_Score"])

        top_hits = [self._id_mapping[str(x)] for x in top_hits_ids]

        results = []
        for i, hit in enumerate(top_hits):
            results.append({
                "rank": i + 1,
                "text": hit["text"],
                "score": combined_similarity[i],
                "file": hit["location"],
                "char_range": hit["char_range"]
            })

        return results
