class ResultProcessor:
    def process(self, annoy_output, id_mapping):
        raise NotImplementedError("Subclasses must implement this method")


class TestingResultProcessor(ResultProcessor):
    """
    {
        "metadata": {
            "dataset_name": "Mitosis",
            "embedding_model": "all-MiniLM-L6-v2",
            "splitting_methods": [
                "by_paragraph"
            ],
            "annoy_trees": 10,
            "embedding_time": 0.23560380935668945
        },
        "results": [
            {
                "query": "Sister chromatids separation during anaphase",
                "ground_truth_doc": "Anaphase.md",
                "ground_truth_position": null,
                "ground_truth_text: ...
                "got_correct_doc_any": true/false
                "overlaps_w_true_position_any: true/false
                "ground_truth_is_subset_any": true/false
                "ordered_results": [
                    {
                        "location": "C:\\Users\\Djhay\\OneDrive\\Desktop\\Projects\\Hackathon\\Hackathon\\TestData\\Mitosis\\Anaphase.md",
                        "char_range": [
                            1,
                            2
                        ],
                        "overlap": ...
                    ...
    """

    def __init__(self, corpus):
        self._corpus = corpus

    def process(
        self,
        top_hits,
        query,
        ground_truth,
    ):
        ordered_results, any_metrics = self._evaluate_results(
            top_hits, ground_truth
        )
        return self._create_results_dict(
            ordered_results, any_metrics, query, ground_truth
        )

    def _evaluate_results(self, top_hits_data, ground_truth):
        ordered_results = [
            self._evaluate_hit(hit, ground_truth) for hit in top_hits_data
        ]
        any_metrics = self._aggregate_any_metrics(ordered_results)

        return ordered_results, any_metrics

    def _aggregate_any_metrics(self, ordered_results):
        """
        Aggregates "any" metrics across all hits.

        Args:
            ordered_results (list): A list of dictionaries representing the evaluation of each hit.
            ground_truth (dict): The ground truth data containing the correct document and position.

        Returns:
            dict: A dictionary containing aggregated metrics.
        """
        # Initialize "any" metrics
        got_correct_doc_any = False
        overlaps_w_true_position_any = False
        ground_truth_is_subset_any = False

        # Iterate through all hit results
        for hit in ordered_results:
            got_correct_doc_any = hit["is_correct_doc"] or got_correct_doc_any
            overlaps_w_true_position_any = (
                hit["overlap_length"] > 0 or overlaps_w_true_position_any
            )
            ground_truth_is_subset_any = hit["is_subset"] or ground_truth_is_subset_any

            # Early exit if all "any" metrics are True
            if (
                got_correct_doc_any
                and overlaps_w_true_position_any
                and ground_truth_is_subset_any
            ):
                break

        # Return the aggregated metrics
        return {
            "got_correct_doc_any": got_correct_doc_any,
            "overlaps_w_true_position_any": overlaps_w_true_position_any,
            "ground_truth_is_subset_any": ground_truth_is_subset_any,
        }

    def OLD_evaluate_results(self, top_hits_data, ground_truth):
        """
        "query": "Sister chromatids separation during anaphase",
        "ground_truth_doc": "Anaphase.md",
        "ground_truth_position": null,
        "ground_truth_text: ...
        "got_correct_doc_any": true/false
        "overlaps_w_true_position_any: true/false
        "ground_truth_is_subset_any": true/false
        "ordered_results": [
        """
        results = {
            "got_correct_doc_any": False,
            "overlaps_w_true_position_any": False,
            "ground_truth_is_subset_any": False,
            "ordered_results": [],
        }

        for hit in top_hits_data:
            hit_res = self._evaluate_hit(hit, ground_truth)
            results["ordered_results"].append(hit_res)

            results["got_correct_doc_any"] = (
                hit_res["is_correct_doc"] or results["got_correct_doc_any"]
            )
            results["overlaps_w_true_position_any"] = (
                hit_res["overlap_length"] > 0 or results["overlaps_w_true_position_any"]
            )
            results["ground_truth_is_subset_any"] = (
                hit_res["is_subset"] or results["ground_truth_is_subset_any"]
            )

        return results

    def _format_overall_results(self, results, ground_truth):
        """
            {
        "query": "Sister chromatids separation during anaphase",
        "ground_truth_doc": "Anaphase.md",
        "ground_truth_position": null,
        "ground_truth_text: ...
        "got_correct_doc_any": true/false
        "overlaps_w_true_position_any: true/false
        "got_top_hit_correct": true,
        "top_hit_overlaps_w_true_pos": true/false
        "ordered_results": [
        """

    def _evaluate_hit(self, hit, ground_truth):
        """
        TODO: Fix key names. The differences between ground truth config and
        hit config is confusing.
        """
        is_correct_doc = self._is_correct_document(
            ground_truth.get("doc", ""),
            hit[
                "location"
            ],  # TODO: better to make names match as opposed to "doc" and "location"
        )
        overlap, overlap_length, is_subset = self._calculate_overlap(
            ground_truth["position"], hit["char_range"]
        )
        return {
            "hit_text": hit["text"],
            "is_correct_doc": is_correct_doc,
            "overlap": overlap,
            "overlap_length": overlap_length,
            "is_subset": is_subset,
            "similarity_score": hit["similarity"],
            "splitting_method": hit["splitting_method"],
        }

    def _calculate_overlap(self, gt_pos, hit_pos):
        overlap_start = max(gt_pos[0], hit_pos[0])
        overlap_end = min(gt_pos[1], hit_pos[1])

        if overlap_start >= overlap_end:
            overlap = None
            overlap_length = 0
        else:
            overlap = [overlap_start, overlap_end]
            overlap_length = overlap_end - overlap_start

        is_subset = gt_pos[0] >= hit_pos[0] and gt_pos[1] <= hit_pos[1]

        return overlap, overlap_length, is_subset

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

    def _create_results_dict(self, ordered_results, any_metrics, query, ground_truth):
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
        res_dict = {"query": query}
        res_dict["ground_truth_doc"] = ground_truth["doc"]
        res_dict["ground_truth_pos"] = ground_truth["position"]
        res_dict["ground_truth_text"] = ground_truth["text"]
        res_dict.update(any_metrics)
        res_dict.update({"ordered_results": ordered_results})
        return res_dict

    def _is_correct_document(self, answer, guess):
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
