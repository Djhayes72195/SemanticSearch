import pandas as pd


class Ranker:

    def __init__(self, config, id_mapping, corpus):
        self._config = config
        self._id_mapping = id_mapping
        self._corpus = corpus

        weights = self._config["semantic_vs_keyword_weights"]
        self._semantic_weight = weights[0]
        self._keyword_weight = weights[1]

    def rank(self, annoy_scores, keyword_scores):
        annoy_ids, annoy_similarity = annoy_scores
        keyword_ids, keyword_similarity = keyword_scores

        df_annoy = pd.DataFrame({'ID': annoy_ids, 'Semantic_Score': annoy_similarity})
        df_keyword = pd.DataFrame({'ID': keyword_ids, 'Keyword_Score': keyword_similarity})

        df_ranked = pd.merge(df_annoy, df_keyword, on='ID', how='outer').fillna(0)

        df_ranked['Combined_Score'] = (
            self._semantic_weight * df_ranked['Semantic_Score'] +
            self._keyword_weight * df_ranked['Keyword_Score']
        )

        # Sort by Combined Score in descending order
        df_ranked = df_ranked.sort_values(by='Combined_Score', ascending=False)

        # Reset index
        df_ranked.reset_index(drop=True, inplace=True)

        return df_ranked

