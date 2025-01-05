class ResultProcessor:
    def process(self, annoy_output, id_mapping):
        raise NotImplementedError("Subclasses must implement this method")


class TestingResultProcessor(ResultProcessor):
    def __init__(self, corpus):
        self._corpus = corpus

    def process(self, annoy_output, id_mapping, query, answer):
        top_hits_data = self._format_top_hits_data(annoy_output, id_mapping)
        return self._create_results_dict(top_hits_data, query, answer)

    def _format_top_hits_data(self, annoy_output, id_mapping):
        top_hits_ids, similarities = annoy_output[0], annoy_output[1]
        top_hits_data = [id_mapping[id] for id in top_hits_ids]

        for i, res in enumerate(top_hits_data):
            res.update({"similarity": similarities[i]})
            res.update(
                {
                    "text": self._corpus.find_passage(
                        res.get("location"), res.get("char_range")
                    )
                }
            )
        return top_hits_data

    def _create_results_dict(self, top_hits_data, query, answer):
        """
        {
            "query": "query text",
            "answer_doc": "Mitosis/xxx",
            "answer_position": "chars 1 through 15 OR NA if doc only test",
            "got_top_hit_doc_correct": "true or false",
            "ordered_results": [
                {
                    "text": "",
                    "source": "",
                    "char_range": [1, 5],
                    "similarity": 0.345,
                },
                {
                    ...
                }
            ]
        }
        """
        return {
            "query": query,
            "answer": answer,
            "answer_position": None,  # TODO: Include answer char range if available,
            "got_top_hit_correct": self._is_correct(
                answer, top_hits_data[0].get("location")
            ),
            "ordered_results": top_hits_data,
        }

    def _is_correct(self, answer, guess):
        """
        Determine if the guess matches the expected answer.

        Args:
            answer (str): The expected answer.
            guess (str): The guessed answer.

        Returns:
            bool: True if the guess is correct, False otherwise.
        """
        return guess.endswith(answer)


class ProductionResultProcessor(ResultProcessor):
    """
    NOT DEVELOPED YET

    skeleton for later use in production
    """

    def __init__(self, text_editor_integration):
        self.text_editor_integration = text_editor_integration

    def process(self, annoy_output, id_mapping):
        top_hits_ids = annoy_output[0]
        best_match_id = top_hits_ids[0]
        best_match_data = id_mapping[best_match_id]
        self.text_editor_integration.open_file_at_location(
            best_match_data["location"], best_match_data["char_range"]
        )
        return best_match_data
