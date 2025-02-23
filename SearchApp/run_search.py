import argparse
from SearchApp.search_orchestrator import SearchOrchestrator

def main():
    """
    Command-line interface for running semantic search.
    """
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    parser.add_argument("--query", type=str, required=True, help="Enter a search query.")
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name for indexing.")
    parser.add_argument("--corpus-id", type=str, required=True, help="Processed corpus ID.")
    args = parser.parse_args()

    # Initialize SearchOrchestrator
    orchestrator = SearchOrchestrator(
        dataset_name=args.dataset_name,
        processed_corpus_id=args.corpus_id,
        config={}
    )

    # Execute search
    results = orchestrator.search(args.query)

    # Display top results
    print("\n **Top Results:**")
    for res in results[:5]:  # Show top 5 results
        print(f"{res['rank']}. {res['text']} (Score: {res['score']:.4f})")

if __name__ == "__main__":
    main()
