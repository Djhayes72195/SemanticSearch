Current scope:
    - Corpus processing
    - Test running


Main components:

    - CorpusProcessor
        - Inputs:
            - The corpus to be processed
        - Outputs:
            - Populated annoy index
            - Populated bm25 index
            - processing metadata
            - A mapping file that connects chunks to entries in annoy and bm25
        - Has a:
            - Embedding manager
            - Keyword manager
            - Text Splitter

    - Embedding manager:
        - Inputs:
            - Chunks to embed
        - Outputs:
            - Annoy index

    - Keyword manager
        - Inputs:
            - Tokenized chunks
        - Outpus:
            - bm25 index

    - Text Splitter:
        - Inputs:
            - Raw text data
        Outputs:
            - Text chunks